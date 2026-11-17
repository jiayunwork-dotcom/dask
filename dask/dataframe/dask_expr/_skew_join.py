"""Skew-aware hash join.

A regular hash shuffle sends every row with the same join key to a single
output partition.  When a single (multi-column) key value -- a *hot key* --
accounts for a large fraction of the rows, that one partition becomes a
bottleneck regardless of the total partition count.

The implementation in this module splits the rows belonging to a set of
caller-provided hot keys into ``S = split_factor`` *salt buckets*, on **both**
sides of the join, and evaluates the hot-key part of the join on an ``S x S``
grid of pairing tasks.  Rows with non-hot keys keep flowing through the
ordinary ``npartitions`` hash partitions and are joined exactly like a normal
hash join.  The two data flows are disjoint, so no pair of records is produced
twice and no pair is missed.

The grid is deliberately symmetric (both sides are salted and every salt pair
is materialized).  An asymmetric "split the big side, replicate the small
side" scheme can use fewer tasks, but choosing the replication factor safely
requires knowing ``Lh`` and ``Rh`` at planning time, which is forbidden here
(the graph is constructed without consuming the sources).  The symmetric grid
guarantees the per-task input bound for any ratio ``Lh / Rh`` using only the
statically known ``S``.

Unmatched rows for outer joins are emitted exactly once per salt bucket by
tasks that consume the small per-side hot-key presence counts.  Counts are
plain integer control information; original rows are never wrapped into
control messages.
"""

from __future__ import annotations

import functools
import os
import shutil
import uuid

import numpy as np
import pandas as pd

from dask._task_spec import Task, TaskRef
from dask.dataframe.core import _concat
from dask.dataframe.dispatch import hash_object_dispatch, partd_encode_dispatch
from dask.dataframe.multi import merge_chunk
from dask.dataframe.shuffle import (
    ensure_cleanup_on_exception,
    maybe_buffered_partd,
)
from dask.tokenize import _tokenize_deterministic

# ---------------------------------------------------------------------------
# Configuration validation and normalization
# ---------------------------------------------------------------------------

#: Keys allowed inside a ``skew_join`` configuration mapping.
_SKEW_CONFIG_FIELDS = frozenset({"hot_keys", "split_factor"})
#: Minimum / maximum number of salt buckets.
_SPLIT_FACTOR_MIN = 2
_SPLIT_FACTOR_MAX = 16


def _is_plain_int(value) -> bool:
    """True for non-boolean Python/NumPy integers."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, np.integer)


def _validate_key_component(value) -> bool:
    # ``bool`` is deliberately excluded even though it is an ``int`` subclass.
    if value is None or isinstance(value, bool):
        return value is None
    if isinstance(value, (int, str, np.integer)):
        return True
    return False


def validate_skew_join(skew_join, key_width: int) -> dict | None:
    """Validate and normalize a user supplied ``skew_join`` configuration.

    The normalization is purely structural and never touches the data.

    Parameters
    ----------
    skew_join:
        The raw mapping (or ``None``).
    key_width:
        Number of join key columns.  Every hot key tuple must have this length.

    Returns
    -------
    dict or None
        ``None`` when ``skew_join`` is ``None`` (feature disabled), otherwise a
        normalized mapping with a sorted tuple of deduplicated hot key tuples
        and a plain Python ``split_factor``.
    """
    if skew_join is None:
        return None
    if not isinstance(skew_join, dict):
        raise ValueError(
            "skew_join must be a dict with keys 'hot_keys' and 'split_factor', "
            f"got {type(skew_join).__name__}"
        )
    unknown = set(skew_join) - _SKEW_CONFIG_FIELDS
    if unknown:
        raise ValueError(
            f"skew_join contains unknown field(s) {sorted(unknown)!r}; "
            f"only {sorted(_SKEW_CONFIG_FIELDS)!r} are allowed"
        )
    if "hot_keys" not in skew_join or "split_factor" not in skew_join:
        raise ValueError(
            "skew_join must define both 'hot_keys' and 'split_factor'"
        )

    raw_hot_keys = skew_join["hot_keys"]
    # The public entry point normalizes inputs to a tuple of tuples; accept
    # both that canonical form and the user-facing list form.
    if isinstance(raw_hot_keys, tuple) and all(
        isinstance(k, tuple) for k in raw_hot_keys
    ):
        # Already normalized; only re-check the split factor bounds.
        hot_keys = raw_hot_keys
        split_factor = skew_join["split_factor"]
        if not _is_plain_int(split_factor) or not (
            _SPLIT_FACTOR_MIN <= int(split_factor) <= _SPLIT_FACTOR_MAX
        ):
            raise ValueError(
                "skew_join['split_factor'] must be a non-boolean integer "
                f"between {_SPLIT_FACTOR_MIN} and {_SPLIT_FACTOR_MAX}; got "
                f"{split_factor!r}"
            )
        if len(hot_keys) == 0:
            raise ValueError("skew_join['hot_keys'] must be a non-empty list")
        return {"hot_keys": hot_keys, "split_factor": int(split_factor)}

    if not isinstance(raw_hot_keys, list) or len(raw_hot_keys) == 0:
        raise ValueError("skew_join['hot_keys'] must be a non-empty list")

    normalized = []
    for idx, hot_key in enumerate(raw_hot_keys):
        if not isinstance(hot_key, tuple):
            raise ValueError(
                "every hot key must be a tuple with one component per join "
                f"column; element {idx} is {type(hot_key).__name__}"
            )
        if key_width and len(hot_key) != key_width:
            raise ValueError(
                f"hot key {hot_key!r} has {len(hot_key)} component(s) but the "
                f"join key has {key_width} column(s)"
            )
        for component in hot_key:
            if not _validate_key_component(component):
                raise ValueError(
                    "hot key components must be Python int, str or None "
                    "(bools are not accepted as ints); got "
                    f"{component!r} ({type(component).__name__})"
                )
        # Normalize NumPy integers to plain Python ints so configs that compare
        # equal really produce the same (tokenized) task graph.
        normalized.append(
            tuple(
                None
                if v is None
                else (int(v) if isinstance(v, np.integer) else v)
                for v in hot_key
            )
        )

    # Deduplicate after normalization, preserving a deterministic order.
    hot_keys = tuple(sorted(set(normalized), key=_hot_key_sort_key))

    split_factor = skew_join["split_factor"]
    if not _is_plain_int(split_factor):
        raise ValueError(
            "skew_join['split_factor'] must be a non-boolean integer between "
            f"{_SPLIT_FACTOR_MIN} and {_SPLIT_FACTOR_MAX}; got "
            f"{split_factor!r} ({type(split_factor).__name__})"
        )
    split_factor = int(split_factor)
    if not (_SPLIT_FACTOR_MIN <= split_factor <= _SPLIT_FACTOR_MAX):
        raise ValueError(
            "skew_join['split_factor'] must be between "
            f"{_SPLIT_FACTOR_MIN} and {_SPLIT_FACTOR_MAX}, got {split_factor}"
        )

    return {"hot_keys": hot_keys, "split_factor": split_factor}


def _hot_key_sort_key(hot_key: tuple):
    # Sort key that tolerates mixed int/str/None component types.
    return tuple(
        (1, "") if v is None else (0, type(v).__name__, str(v)) for v in hot_key
    )


# ---------------------------------------------------------------------------
# Row-level helpers (executed inside tasks)
# ---------------------------------------------------------------------------

# Piece dictionary key used to carry the small hot-key count vector.  It is an
# integer slot that cannot collide with the tuple piece keys used below.
_COUNT_KEY = -1


def _hot_key_mask(df: pd.DataFrame, on: list, hot_key: tuple) -> np.ndarray:
    """Vectorized membership mask for one composite hot key.

    Missing components match missing values, following pandas merge semantics
    where ``NaN == NaN`` (and ``NA == NA``) on join keys.
    """
    mask = np.ones(len(df), dtype=bool)
    for column, value in zip(on, hot_key):
        series = df[column]
        if value is None:
            part = np.asarray(series.isna(), dtype=bool)
        else:
            # Nullable comparisons yield NA for missing rows; treat NA as
            # non-matching for a concrete hot key component.
            part = np.asarray((series == value).fillna(False), dtype=bool)
        mask &= part
    return mask


def _normal_targets(df: pd.DataFrame, on: list, npartitions: int) -> np.ndarray:
    """Hash partition index for the non-hot rows."""
    hashed = hash_object_dispatch(df[on], index=False)
    return (hashed % int(npartitions)).to_numpy()


def skew_split_frame(
    df: pd.DataFrame,
    on: list,
    hot_keys: tuple,
    split_factor: int,
    npartitions: int,
):
    """Split one input partition into hot-salt pieces and normal pieces.

    Returns a dictionary ``pieces`` that also carries the small count vector.
    Row data and control information use disjoint key spaces:

    * ``pieces[(0, h, s)]`` -- the hot rows of hot key ``h`` assigned to salt
      bucket ``s`` (one of ``split_factor`` buckets),
    * ``pieces[(1, p)]`` -- the non-hot rows hashed to normal partition ``p``
      (one of ``npartitions`` partitions),
    * ``pieces[_COUNT_KEY]`` -- ``tuple[int]`` with the number of hot rows of
      each hot key in this input partition (control information only).

    Salt assignment is a per-(partition, hot-key) round-robin driven by the
    running row offset, i.e. purely local.  No side is ever gathered by key
    before splitting.
    """
    empty = df.iloc[0:0]
    if not len(df):
        pieces = {(1, p): empty.copy() for p in range(npartitions)}
        pieces[_COUNT_KEY] = tuple(0 for _ in hot_keys)
        return pieces

    hot_union = np.zeros(len(df), dtype=bool)
    hot_index = np.full(len(df), -1, dtype=np.int64)
    hot_salt = np.full(len(df), -1, dtype=np.int64)
    counts = []
    for h, hot_key in enumerate(hot_keys):
        mask = _hot_key_mask(df, on, hot_key)
        counts.append(int(mask.sum()))
        positions = np.flatnonzero(mask)
        if positions.size:
            # Local, deterministic round-robin over the matching rows.  Using
            # the global row position means identical rows still spread across
            # all salt buckets instead of collapsing onto one hash.
            hot_salt[positions] = np.arange(positions.size) % split_factor
            hot_index[positions] = h
        hot_union |= mask

    pieces = {}
    hot_frame = df[hot_union]
    if len(hot_frame):
        salt_values = hot_salt[hot_union]
        hot_labels = hot_index[hot_union]
        for h in range(len(hot_keys)):
            for s in range(split_factor):
                sel = (hot_labels == h) & (salt_values == s)
                pieces[(0, h, s)] = hot_frame[sel]

    normal_frame = df[~hot_union]
    if len(normal_frame):
        targets = _normal_targets(normal_frame, on, npartitions)
        for p in range(npartitions):
            pieces[(1, p)] = normal_frame[targets == p]
    else:
        for p in range(npartitions):
            pieces[(1, p)] = empty.copy()

    pieces[_COUNT_KEY] = tuple(counts)
    _instrument_record(
        {
            "kind": "split",
            "on": list(on),
            "n_rows": int(len(df)),
            "counts": list(counts),
        }
    )
    return pieces


def skew_collect_counts(n_hot: int, *split_results) -> tuple:
    """Aggregate the small hot-key count vectors of one side."""
    totals = [0] * n_hot
    for result in split_results:
        vector = result[_COUNT_KEY]
        for h, value in enumerate(vector):
            totals[h] += int(value)
    return tuple(totals)


def skew_concat_pieces(side, key, *split_results):
    """Concatenate the rows gathered for one piece key on one side."""
    frames = [result[key] for result in split_results if key in result]
    if not frames:
        return None
    # Use Dask's concat helper: it drops empty partitions before concatenating
    # so pyarrow-string metadata dtypes survive all-empty pieces.
    out = _concat(frames)
    if not len(out):
        return None
    _instrument_record(
        {
            "kind": "gather",
            "side": side,
            "piece": list(key) if isinstance(key, tuple) else key,
            "n_rows": int(len(out)),
        }
    )
    return out


def _align_to_meta(frame, result_meta):
    """Cast every output column to the result-metadata dtype.

    A pandas merge can produce a different dtype for the same column in
    different partitions (e.g. a categorical join key upcasts to ``object`` in
    unmatched rows but stays categorical in matched ones).  Normalizing against
    the single ``result_meta`` dtype keeps every output partition concatenable
    and makes the collected result match the pandas reference exactly.
    """
    if frame is None or not len(frame):
        return result_meta.iloc[0:0]
    for column in result_meta.columns:
        if column not in frame.columns:
            continue
        target = result_meta[column].dtype
        if frame[column].dtype == target:
            continue
        try:
            frame = frame.assign(**{column: frame[column].astype(target)})
        except (TypeError, ValueError):
            # Integer columns that gain NA rows become float in a pandas merge
            # rather than object; mirror that conversion.
            if pd.api.types.is_integer_dtype(target) and frame[column].isna().any():
                frame = frame.assign(
                    **{column: frame[column].astype("float64")}
                )
            else:
                frame = frame.assign(
                    **{column: frame[column].astype(object)}
                )
    return frame[result_meta.columns]


def skew_pair_partition(left_piece, right_piece, result_meta, **merge_kwargs):
    """Inner pairing of one (left-salt, right-salt) cell for one hot key.

    Only matches are emitted here; outer-join unmatched rows are handled by
    dedicated emission tasks.  Keeping the cell strictly inner avoids
    duplicating unmatched rows across the grid row/column.
    """
    kwargs = dict(merge_kwargs)
    kwargs["how"] = "inner"
    if (
        left_piece is None
        or right_piece is None
        or not len(left_piece)
        or not len(right_piece)
    ):
        return result_meta.iloc[0:0]
    out = merge_chunk(left_piece, right_piece, result_meta=result_meta, **kwargs)
    _instrument_record(
        {
            "kind": "pair",
            "n_left": int(len(left_piece)),
            "n_right": int(len(right_piece)),
            "n_out": int(len(out)),
        }
    )
    return _align_to_meta(out, result_meta)


def skew_emit_unmatched(
    piece,
    other_counts,
    hot_index: int,
    side: str,
    result_meta,
    indicator_name,
    **merge_kwargs,
):
    """Emit the unmatched rows of one salt bucket of one side.

    ``other_counts`` is the aggregated count vector of the opposite side.  The
    bucket emits its original rows (as unmatched output rows) only when the
    opposite side has no row with this hot key at all.  When both sides
    contain the key, every row is matched somewhere in the grid and nothing is
    emitted.

    The output rows are assembled directly against ``result_meta`` so that the
    column set, dtypes and indicator category values always match pandas.
    """
    how = merge_kwargs["how"]
    if side == "left" and how not in ("left", "outer"):
        return result_meta.iloc[0:0]
    if side == "right" and how not in ("right", "outer"):
        return result_meta.iloc[0:0]
    if other_counts[hot_index] > 0:
        return result_meta.iloc[0:0]
    if piece is None or not len(piece):
        return result_meta.iloc[0:0]

    n = len(piece)
    out = result_meta.iloc[0:0].copy()
    # Reindex creates NA rows with the output dtypes for the opposite side.
    out = out.reindex(range(n))
    for column in piece.columns:
        if column in out.columns:
            values = piece[column].reset_index(drop=True)
            target_dtype = out[column].dtype
            # Cast to the output dtype.  A pandas merge upcasts some input
            # dtypes (e.g. a categorical join key becomes object in the
            # result); the emitted rows must follow that output dtype so that
            # downstream concat never has to union mismatched categoricals.
            if values.dtype != target_dtype:
                try:
                    values = values.astype(target_dtype)
                except (TypeError, ValueError):
                    values = values.astype(object)
            out[column] = values
    if indicator_name is not None:
        label = "left_only" if side == "left" else "right_only"
        categories = result_meta[indicator_name].cat.categories
        out[indicator_name] = pd.Categorical(
            [label] * n, categories=categories
        )
    _instrument_record(
        {
            "kind": "emit",
            "side": side,
            "hot_index": int(hot_index),
            "n_rows": int(len(out)),
        }
    )
    return _align_to_meta(out, result_meta)


def skew_normal_merge(
    left_piece,
    right_piece,
    result_meta,
    left_meta,
    right_meta,
    **merge_kwargs,
):
    """Full-semantics merge of one normal (non-hot) hash partition."""
    if left_piece is None:
        left_piece = left_meta.iloc[0:0]
    if right_piece is None:
        right_piece = right_meta.iloc[0:0]
    out = merge_chunk(
        left_piece, right_piece, result_meta=result_meta, **merge_kwargs
    )
    return _align_to_meta(out, result_meta)


# ---------------------------------------------------------------------------
# Optional, dependency-free instrumentation for the per-task bound checks
# ---------------------------------------------------------------------------

# When this environment variable points at a directory, every split / gather /
# pairing / emission task appends a JSON record describing the rows it held.
# The variable is intentionally env-based so that worker processes pick it up
# without embedding mutable state in the (tokenized) task graph.
_INSTRUMENT_ENV = "DASK_SKEW_JOIN_INSTRUMENT"


def _instrument_record(payload: dict) -> None:
    directory = os.environ.get(_INSTRUMENT_ENV)
    if not directory:
        return
    import json

    payload = dict(payload)
    payload["pid"] = os.getpid()
    # One file per task invocation avoids cross-process write contention.
    name = f"{payload['kind']}-{uuid.uuid4().hex}.json"
    with open(os.path.join(directory, name), "w") as fh:
        json.dump(payload, fh)


# ---------------------------------------------------------------------------
# partd (disk shuffle) helpers
# ---------------------------------------------------------------------------


def _skew_barrier(*args):
    # Mirrors dask.dataframe.shuffle.barrier but accepts variadic task refs.
    list(args)
    return 0


def partd_append_pieces(store, df, on, hot_keys, split_factor, npartitions):
    """Split a partition and append its row pieces into a partd store.

    The store only ever holds original rows keyed by piece identifier; the
    count vector is returned to the caller instead of being stored.
    """
    with ensure_cleanup_on_exception(store):
        pieces = skew_split_frame(df, on, hot_keys, split_factor, npartitions)
        counts = pieces.pop(_COUNT_KEY)
        if pieces:
            store.append(pieces)
        return {_COUNT_KEY: counts}


def partd_collect_piece(store, side, piece_key, meta, barrier_token):
    """Read one row piece back from partd.

    The store is a :class:`_RefCountedPartd` wrapper.  The wrapper removes the
    temporary directory once the last expected collection has run, so cleanup
    happens as part of the normal data flow (no unreferenced terminal tasks)
    and also covers failures before the final collection.
    """
    with ensure_cleanup_on_exception(store):
        frame = store.get(piece_key)
        if frame is None or not len(frame):
            return None
        _instrument_record(
            {
                "kind": "gather",
                "side": side,
                "piece": list(piece_key) if isinstance(piece_key, tuple) else piece_key,
                "n_rows": int(len(frame)),
            }
        )
        return frame


class _RefCountedPartd:
    """partd wrapper that drops its on-disk store after N collections.

    Every gathered piece is one collection.  Culling changes the number of
    materialized outputs, so the expected count is supplied by the expression
    when the graph is built.  Once the last collection finishes (or any
    collection raises), the underlying partd directory is removed.

    partd's own ``drop()`` empties and then recreates its directory (its
    ``atexit`` hook is what normally removes it); the wrapper therefore
    removes the now-empty directory explicitly.
    """

    def __init__(self, store, expected_collects: int):
        self._store = store
        self._remaining = int(expected_collects)
        self._dropped = False
        self._path = _partd_file_path(store)

    def append(self, data):
        with ensure_cleanup_on_exception(self._store):
            return self._store.append(data)

    def get(self, key):
        try:
            with ensure_cleanup_on_exception(self._store):
                return self._store.get(key)
        finally:
            self._remaining -= 1
            if self._remaining <= 0 and not self._dropped:
                self._dropped = True
                self._teardown()

    def drop(self):
        if not self._dropped:
            self._dropped = True
            self._teardown()

    def _teardown(self):
        try:
            self._store.drop()
        finally:
            path = self._path
            if path and os.path.isdir(path) and not os.listdir(path):
                shutil.rmtree(path, ignore_errors=True)


def _make_refcounted_partd(factory, expected_collects):
    # ``factory`` is a dask maybe_buffered_partd instance, i.e. a callable
    # that materializes the actual partd store (including its temp directory).
    return _RefCountedPartd(factory(), expected_collects)


def _partd_file_path(store):
    """Walk a partd wrapper chain to the underlying File.path, if any."""
    seen = set()
    node = store
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        path = getattr(node, "path", None)
        if isinstance(path, str):
            return path
        # Encode -> partd, PandasBlocks(.partd) -> Buffer(.slow) -> File
        node = getattr(node, "partd", None) or getattr(node, "slow", None)
    return None


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------

from dask.dataframe.dask_expr._expr import PartitionsFiltered  # noqa: E402
from dask.dataframe.dask_expr._merge import Merge  # noqa: E402


class SkewJoin(Merge, PartitionsFiltered):
    """Physical hash join that disperses caller-identified hot keys.

    The expression owns the complete data movement plan: local splitting,
    exchange, hot-key pairing grid, outer-join unmatched emission and normal
    hash partitions.  It deliberately advertises no partition mapping for the
    join key (``unique_partition_mapping_columns_from_shuffle`` is empty) so
    downstream groupby/drop-duplicates/merge operations cannot assume that
    equal keys are co-located.
    """

    _parameters = [
        "left",
        "right",
        "how",
        "left_on",
        "right_on",
        "left_index",
        "right_index",
        "suffixes",
        "indicator",
        "shuffle_method",
        "_npartitions",
        "broadcast",
        "skew_join",
        "_partitions",
    ]
    _defaults = {
        "how": "inner",
        "left_on": None,
        "right_on": None,
        "left_index": False,
        "right_index": False,
        "suffixes": ("_x", "_y"),
        "indicator": False,
        "shuffle_method": None,
        "_npartitions": None,
        "broadcast": None,
        "skew_join": None,
        "_partitions": None,
    }

    is_broadcast_join = False

    # -- static, planning-time properties ----------------------------------

    @functools.cached_property
    def skew_config(self) -> dict:
        return validate_skew_join(self.skew_join, len(self._left_on_list))

    @functools.cached_property
    def hot_keys(self) -> tuple:
        return self.skew_config["hot_keys"]

    @functools.cached_property
    def split_factor(self) -> int:
        return self.skew_config["split_factor"]

    @functools.cached_property
    def _left_on_list(self) -> list:
        return list(self.left_on) if isinstance(self.left_on, list) else [self.left_on]

    @functools.cached_property
    def _right_on_list(self) -> list:
        return (
            list(self.right_on)
            if isinstance(self.right_on, list)
            else [self.right_on]
        )

    @functools.cached_property
    def n_hot(self) -> int:
        return len(self.hot_keys)

    @functools.cached_property
    def normal_npartitions(self) -> int:
        if self.operand("_npartitions") is not None:
            return self.operand("_npartitions")
        return max(self.left.npartitions, self.right.npartitions)

    @property
    def _n_cells(self) -> int:
        return self.n_hot * self.split_factor * self.split_factor

    @property
    def _n_emissions(self) -> int:
        return 2 * self.n_hot * self.split_factor

    def _hot_pair_index(self, h: int, sl: int, sr: int) -> int:
        s = self.split_factor
        return self.normal_npartitions + h * s * s + sl * s + sr

    def _emission_index(self, side: str, h: int, salt: int) -> int:
        s = self.split_factor
        base = self.normal_npartitions + self._n_cells
        offset = h * s + salt
        if side == "right":
            offset += self.n_hot * s
        return base + offset

    @functools.cached_property
    def _full_npartitions(self) -> int:
        return self.normal_npartitions + self._n_cells + self._n_emissions

    # -- Expr plumbing ------------------------------------------------------

    def _lower(self):
        return None

    @property
    def npartitions(self):
        if self._filtered:
            return len(self._partitions)
        return self._full_npartitions

    def _divisions(self):
        return (None,) * (self._full_npartitions + 1)

    @functools.cached_property
    def unique_partition_mapping_columns_from_shuffle(self):
        # Hot-key rows are spread over salt buckets and the grid output has no
        # key co-location guarantee.  Partition pruning based on the join key
        # must therefore re-shuffle.
        return set()

    @functools.cached_property
    def _plan_token(self) -> str:
        return _tokenize_deterministic(
            self._name,
            self.skew_join,
            self.left_on,
            self.right_on,
            self.normal_npartitions,
            self.shuffle_method,
        )

    # -- graph generation ---------------------------------------------------

    def _out_key(self, global_index: int):
        """Map a full-plan output index to the materialized task key.

        When partitions are culled (``PartitionsFiltered``), the expression's
        ``npartitions`` becomes the number of selected partitions and its
        dask keys use dense local indices.  Materialized keys must follow the
        same convention instead of keeping the global index.
        """
        if self._filtered:
            return (self._name, self._partitions.index(global_index))
        return (self._name, global_index)

    def _layer(self) -> dict:
        method = self.shuffle_method or "tasks"
        if method == "disk":
            return self._layer_disk()
        if method != "tasks":
            # p2p is rejected at construction time; stay defensive anyway.
            raise ValueError(
                f"skew_join does not support shuffle_method={method!r}"
            )
        return self._layer_tasks()

    def _add_split_tasks(
        self,
        dsk: dict,
        split_left: str,
        split_right: str,
        disk: bool,
        partd_left=None,
        partd_right=None,
    ):
        left_refs = []
        right_refs = []
        common = (
            self.hot_keys,
            self.split_factor,
            self.normal_npartitions,
        )
        split_func = partd_append_pieces if disk else skew_split_frame
        for i in range(self.left.npartitions):
            key = (split_left, i)
            if disk:
                task = Task(
                    key,
                    split_func,
                    TaskRef(partd_left),
                    TaskRef((self.left._name, i)),
                    self._left_on_list,
                    *common,
                )
            else:
                task = Task(
                    key,
                    split_func,
                    TaskRef((self.left._name, i)),
                    self._left_on_list,
                    *common,
                )
            dsk[key] = task
            left_refs.append(task.ref())
        for i in range(self.right.npartitions):
            key = (split_right, i)
            if disk:
                task = Task(
                    key,
                    split_func,
                    TaskRef(partd_right),
                    TaskRef((self.right._name, i)),
                    self._right_on_list,
                    *common,
                )
            else:
                task = Task(
                    key,
                    split_func,
                    TaskRef((self.right._name, i)),
                    self._right_on_list,
                    *common,
                )
            dsk[key] = task
            right_refs.append(task.ref())
        return left_refs, right_refs

    def _add_count_tasks(self, dsk, name, refs):
        key = (name, 0)
        task = Task(key, skew_collect_counts, self.n_hot, *refs)
        dsk[key] = task
        return TaskRef(key)

    def _layer_tasks(self) -> dict:
        dsk: dict = {}
        token = self._plan_token
        s = self.split_factor
        n_hot = self.n_hot
        n_normal = self.normal_npartitions
        wanted = set(self._partitions)

        split_left = f"skew-split-left-{token}"
        split_right = f"skew-split-right-{token}"
        gather_left = f"skew-gather-left-{token}"
        gather_right = f"skew-gather-right-{token}"

        left_split_refs, right_split_refs = self._add_split_tasks(
            dsk, split_left, split_right, disk=False
        )

        # Small, row-free count aggregations (control information only).
        count_left_ref = self._add_count_tasks(
            dsk, f"skew-count-left-{token}", left_split_refs
        )
        count_right_ref = self._add_count_tasks(
            dsk, f"skew-count-right-{token}", right_split_refs
        )

        gather_cache: dict = {}

        def gather(side: str, piece_key):
            cache_key = (side, piece_key)
            if cache_key in gather_cache:
                return gather_cache[cache_key]
            name = gather_left if side == "left" else gather_right
            split_refs = left_split_refs if side == "left" else right_split_refs
            key = (name, _piece_token(piece_key))
            task = Task(
                key,
                skew_concat_pieces,
                side,
                piece_key,
                *split_refs,
            )
            dsk[task.key] = task
            gather_cache[cache_key] = task.ref()
            return task.ref()

        self._build_output_tasks(
            dsk, wanted, s, n_hot, n_normal, gather, count_left_ref, count_right_ref
        )
        return dsk

    def _layer_disk(self) -> dict:
        dsk: dict = {}
        token = self._plan_token
        s = self.split_factor
        n_hot = self.n_hot
        n_normal = self.normal_npartitions
        wanted = set(self._partitions)

        split_left = f"skew-split-left-{token}"
        split_right = f"skew-split-right-{token}"
        gather_left = f"skew-gather-left-{token}"
        gather_right = f"skew-gather-right-{token}"

        # Two independent partd stores; keys are the piece tuples.  The
        # reference-counted wrapper removes each directory once the last
        # materialized collection for that side has finished.
        p_left = (f"skew-zpartd-left-{token}",)
        p_right = (f"skew-zpartd-right-{token}",)

        left_split_refs, right_split_refs = self._add_split_tasks(
            dsk,
            split_left,
            split_right,
            disk=True,
            partd_left=p_left,
            partd_right=p_right,
        )

        # Barriers mirror DiskShuffle: collect() must not run before every
        # split task finished appending.
        barrier_left = (f"skew-barrier-left-{token}",)
        barrier_right = (f"skew-barrier-right-{token}",)
        dsk[barrier_left] = Task(barrier_left, _skew_barrier, *left_split_refs)
        dsk[barrier_right] = Task(barrier_right, _skew_barrier, *right_split_refs)

        count_left_ref = self._add_count_tasks(
            dsk, f"skew-count-left-{token}", left_split_refs
        )
        count_right_ref = self._add_count_tasks(
            dsk, f"skew-count-right-{token}", right_split_refs
        )

        # Count how many distinct pieces each side must materialize for the
        # requested output partitions, so each wrapper knows when to drop.
        needed_left, needed_right = self._needed_pieces(wanted, s, n_hot, n_normal)
        dsk[p_left] = Task(
            p_left,
            _make_refcounted_partd,
            maybe_buffered_partd(
                encode_cls=partd_encode_dispatch(self.left._meta)
            ),
            len(needed_left),
        )
        dsk[p_right] = Task(
            p_right,
            _make_refcounted_partd,
            maybe_buffered_partd(
                encode_cls=partd_encode_dispatch(self.right._meta)
            ),
            len(needed_right),
        )

        gather_cache: dict = {}

        def gather(side: str, piece_key):
            cache_key = (side, piece_key)
            if cache_key in gather_cache:
                return gather_cache[cache_key]
            store = p_left if side == "left" else p_right
            barrier_token = barrier_left if side == "left" else barrier_right
            meta = self.left._meta if side == "left" else self.right._meta
            name = gather_left if side == "left" else gather_right
            key = (name, _piece_token(piece_key))
            task = Task(
                key,
                partd_collect_piece,
                TaskRef(store),
                side,
                piece_key,
                meta,
                TaskRef(barrier_token),
            )
            dsk[task.key] = task
            gather_cache[cache_key] = task.ref()
            return task.ref()

        self._build_output_tasks(
            dsk, wanted, s, n_hot, n_normal, gather, count_left_ref, count_right_ref
        )
        return dsk

    def _needed_pieces(self, wanted, s, n_hot, n_normal):
        """Distinct (side, piece) keys consumed by the wanted outputs."""
        left_pieces, right_pieces = set(), set()
        for global_index in wanted:
            if global_index < n_normal:
                left_pieces.add((1, global_index))
                right_pieces.add((1, global_index))
                continue
            cell_base = n_normal
            grid_end = cell_base + self._n_cells
            if global_index < grid_end:
                offset = global_index - cell_base
                h, rem = divmod(offset, s * s)
                sl, sr = divmod(rem, s)
                left_pieces.add((0, h, sl))
                right_pieces.add((0, h, sr))
                continue
            emit_offset = global_index - grid_end
            if emit_offset < self.n_hot * s:
                h, salt = divmod(emit_offset, s)
                left_pieces.add((0, h, salt))
            else:
                emit_offset -= self.n_hot * s
                h, salt = divmod(emit_offset, s)
                right_pieces.add((0, h, salt))
        return left_pieces, right_pieces

    def _build_output_tasks(
        self,
        dsk,
        wanted,
        s,
        n_hot,
        n_normal,
        gather,
        count_left_ref,
        count_right_ref,
    ):
        merge_kwargs = self.kwargs.copy()

        # Normal hash partitions -------------------------------------------
        for p in range(n_normal):
            if p not in wanted:
                continue
            task = Task(
                self._out_key(p),
                skew_normal_merge,
                gather("left", (1, p)),
                gather("right", (1, p)),
                self._meta,
                self.left._meta,
                self.right._meta,
                **merge_kwargs,
            )
            dsk[task.key] = task

        # Hot-key pairing grid ----------------------------------------------
        for h in range(n_hot):
            for sl in range(s):
                left_ref = gather("left", (0, h, sl))
                for sr in range(s):
                    out_index = self._hot_pair_index(h, sl, sr)
                    if out_index not in wanted:
                        continue
                    task = Task(
                        self._out_key(out_index),
                        skew_pair_partition,
                        left_ref,
                        gather("right", (0, h, sr)),
                        self._meta,
                        **merge_kwargs,
                    )
                    dsk[task.key] = task

        indicator_name = self.indicator if isinstance(self.indicator, str) else (
            "_merge" if self.indicator else None
        )

        # Outer-join unmatched emission -------------------------------------
        for h in range(n_hot):
            for salt in range(s):
                left_out = self._emission_index("left", h, salt)
                if left_out in wanted:
                    task = Task(
                        self._out_key(left_out),
                        skew_emit_unmatched,
                        gather("left", (0, h, salt)),
                        count_right_ref,
                        h,
                        "left",
                        self._meta,
                        indicator_name,
                        **merge_kwargs,
                    )
                    dsk[task.key] = task
                right_out = self._emission_index("right", h, salt)
                if right_out in wanted:
                    task = Task(
                        self._out_key(right_out),
                        skew_emit_unmatched,
                        gather("right", (0, h, salt)),
                        count_left_ref,
                        h,
                        "right",
                        self._meta,
                        indicator_name,
                        **merge_kwargs,
                    )
                    dsk[task.key] = task


def _piece_token(piece_key) -> str:
    return _tokenize_deterministic(piece_key)
