"""Distributed cardinality validation for :class:`Merge` expressions.

Pandas' ``merge(validate=...)`` checks that the join keys of the left and/or
right input are globally unique.  Dask's merge is distributed and lazy, so the
guarantee has to be (a) checked only at compute time without reading the
sources during graph construction, (b) evaluated against the *full* inputs of
the merge (even if the output is filtered, projected down to a single
partition or otherwise reduced), and (c) bounded in memory: a single task may
never hold all the distinct keys, let alone all the raw rows, of a checked
side.

The check implemented here mirrors the semantics of
``pandas.core.reshape.merge._validate_validate_kwd`` on pandas 2.2.3:

* ``1:1``  -> both sides' join keys must be unique
* ``1:m``  -> the left join keys must be unique
* ``m:1``  -> the right join keys must be unique
* ``m:m`` / ``None`` -> no check

Equality of keys (including the treatment of ``NaN``/``NA`` as equal) follows
:class:`pandas.MultiIndex` / :meth:`pandas.DataFrame.duplicated`, exactly like
pandas' own validation.

Bounded check
-------------
For every checked side the *join-key columns* are turned into a local
per-partition ``value_counts`` summary (distinct keys + counts).  Only these
small summaries are hash-partitioned by the key, so equal keys -- including
hot keys -- end up on the same check task while the raw rows stay on the
input partitions.  Each check task sums the counts and reports a single
boolean flag.  A tree reduction (constant fan-in) combines the flags into one
scalar, which every output partition of the merge depends on.  The amount of
state held by any new task is bounded by the number of distinct keys of a
single hash bucket, never by the number of input rows.
"""

from __future__ import annotations

import functools

import pandas as pd
from pandas.errors import MergeError

from dask._task_spec import Task, TaskRef
from dask.dataframe.dask_expr._expr import Blockwise, Expr
from dask.dataframe.dask_expr._shuffle import RearrangeByColumn
from dask.typing import Key

_VALIDATE_ALIASES = {
    "one_to_one": "1:1",
    "1:1": "1:1",
    "one_to_many": "1:m",
    "1:m": "1:m",
    "many_to_one": "m:1",
    "m:1": "m:1",
    "many_to_many": "m:m",
    "m:m": "m:m",
}

#: validate values that actually require a uniqueness check
_CHECKED = frozenset({"1:1", "1:m", "m:1"})

_VALIDATE_VALUES = (
    "one_to_one",
    "1:1",
    "one_to_many",
    "1:m",
    "many_to_one",
    "m:1",
    "many_to_many",
    "m:m",
)

#: sides whose join keys must be unique for each validate mode
_CHECKED_SIDES = {
    "1:1": ("left", "right"),
    "1:m": ("left",),
    "m:1": ("right",),
}

#: pandas 2.2.3 MergeError message templates, keyed by (validate, side)
_MERGE_ERROR_MESSAGES = {
    ("1:1", "both"): (
        "Merge keys are not unique in either left or right dataset; "
        "not a one-to-one merge"
    ),
    ("1:1", "left"): (
        "Merge keys are not unique in left dataset; not a one-to-one merge"
    ),
    ("1:1", "right"): (
        "Merge keys are not unique in right dataset; not a one-to-one merge"
    ),
    ("1:m", "left"): (
        "Merge keys are not unique in left dataset; not a one-to-many merge"
    ),
    ("m:1", "right"): (
        "Merge keys are not unique in right dataset; not a many-to-one merge"
    ),
}


def normalize_validate(validate):
    """Normalize the user facing ``validate`` argument.

    Raises
    ------
    ValueError
        For any value that is not one of the eight documented strings.  This
        is a graph-construction time check and never touches source data.
    """
    if validate is None:
        return None
    try:
        return _VALIDATE_ALIASES[validate]
    except (KeyError, TypeError) as err:
        raise ValueError(
            f"{validate!r} is not a valid argument. Valid arguments are:\n"
            + "\n".join(f'- "{v}"' for v in _VALIDATE_VALUES)
        ) from err


def merge_error_message(validate, side):
    """Return the pandas 2.2.3 MergeError message for *side* ("left"/"right"/"both")."""
    return _MERGE_ERROR_MESSAGES[(validate, side)]


# ---------------------------------------------------------------------------
# Key materialization
# ---------------------------------------------------------------------------


def materialize_key_frame(frame, on, index):
    """Return an expression selecting only the join-key columns of *frame*.

    Index keys are turned into a regular column so that the same column based
    summarization and hash partitioning can be used for column-on-column,
    index-on-column and index-on-index merges.  Only single-level indexes are
    supported by the merge API itself.
    """
    if index:
        # Expressions do not expose collection methods; build the reset-index
        # operation explicitly. A single-level index (the only kind merge
        # supports) becomes one column named after the index, or "index".
        from dask.dataframe.dask_expr._expr import ResetIndex

        reset = ResetIndex(frame, drop=False)
        name = frame._meta.index.name or "index"
        return reset[[name]]
    if isinstance(on, list):
        cols = list(on)
    else:
        cols = [on]
    return frame[cols]


# ---------------------------------------------------------------------------
# Task level helpers (pure pandas, run inside worker tasks)
# ---------------------------------------------------------------------------


def local_key_counts(df, key_columns):
    """Summarize one partition: distinct join keys with their local counts.

    ``NaN``/``NA`` are treated as equal keys, matching
    ``MultiIndex.from_arrays(...).is_unique`` in pandas.
    """
    counts = df.value_counts(subset=list(key_columns), dropna=False)
    out = counts.rename("_n").reset_index()
    return out


def sum_key_counts(*frames):
    """Combine per-partition key summaries and flag duplicated keys.

    Returns a one-row, one-column frame holding a single boolean.  Only the
    (already summarized) distinct keys of one hash bucket are concatenated
    here, so a hot key never causes the raw rows to be gathered.
    """
    if not frames:
        return pd.DataFrame({"_duplicates": [False]})
    keys = [f for f in frames if len(f)]
    if not keys:
        return pd.DataFrame({"_duplicates": [False]})
    combined = pd.concat(keys, ignore_index=True)
    key_columns = [c for c in combined.columns if c != "_n"]
    # Sum the per-partition multiplicities of every key; a total > 1 means
    # the same join key occurs on more than one input row.
    totals = (
        combined.groupby(key_columns, dropna=False, observed=True)["_n"]
        .sum()
        .reset_index()
    )
    has_duplicates = bool((totals["_n"] > 1).any())
    return pd.DataFrame({"_duplicates": [has_duplicates]})


def combine_duplicate_flags(*frames):
    """OR-combine the per-bucket boolean flags up the reduction tree."""
    bad = any(bool(f.iloc[0, 0]) for f in frames if f is not None and len(f))
    return pd.DataFrame({"_duplicates": [bad]})


def raise_if_invalid(left_flag, right_flag, validate, error_validate=None):
    """Turn the two boolean flags into the pandas MergeError (if any).

    ``validate`` selects which flag(s) are consulted; ``error_validate``
    selects the wording of the raised error (the original user mode).
    """
    error_validate = error_validate or validate
    left_bad = bool(_flag_value(left_flag))
    right_bad = bool(_flag_value(right_flag))
    if validate == "1:1":
        if left_bad and right_bad:
            raise MergeError(merge_error_message(error_validate, "both"))
        if left_bad:
            raise MergeError(merge_error_message(error_validate, "left"))
        if right_bad:
            raise MergeError(merge_error_message(error_validate, "right"))
    elif validate == "1:m" and left_bad:
        raise MergeError(merge_error_message(error_validate, "left"))
    elif validate == "m:1" and right_bad:
        raise MergeError(merge_error_message(error_validate, "right"))
    return 0


def _flag_value(flag_frame):
    if flag_frame is None:
        return False
    if isinstance(flag_frame, pd.DataFrame):
        if flag_frame.empty:
            return False
        return flag_frame.iloc[0, 0]
    return bool(flag_frame)


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


class LocalKeyCounts(Blockwise):
    """Blockwise local ``value_counts`` summary of the join-key columns."""

    _parameters = ["frame", "key_columns"]
    _is_length_preserving = False

    @functools.cached_property
    def _meta(self):
        cols = list(self.key_columns)
        meta = self.frame._meta[cols].iloc[:0]
        meta = meta.assign(_n=pd.Series(dtype="int64"))
        return make_meta_like(meta)

    def _divisions(self):
        return (None,) * (self.frame.npartitions + 1)

    def _task(self, name: Key, index: int) -> Task:
        return Task(
            name,
            local_key_counts,
            TaskRef((self.frame._name, index)),
            list(self.key_columns),
        )


class KeyCountsPartitioned(Expr):
    """Hash-partition the local key summaries by the join key.

    Equal keys always land in the same output partition, so a per-partition
    sum of the counts is an exact global uniqueness test while every task only
    sees the distinct keys of one hash bucket.
    """

    _parameters = ["frame", "key_columns", "npartitions_out", "shuffle_method"]
    _defaults = {"shuffle_method": None}

    @functools.cached_property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        return (None,) * (self.npartitions_out + 1)

    def _lower(self):
        shuffled = RearrangeByColumn(
            self.frame,
            list(self.key_columns),
            npartitions_out=self.npartitions_out,
            method=self.shuffle_method,
        )
        return shuffled


class DuplicateFlag(Blockwise):
    """Per-partition boolean flag: does this hash bucket hold a duplicated key?"""

    _parameters = ["frame", "key_columns"]
    _is_length_preserving = False

    @functools.cached_property
    def _meta(self):
        return make_meta_like(pd.DataFrame({"_duplicates": pd.array([], dtype=bool)}))

    def _divisions(self):
        return (None,) * (self.frame.npartitions + 1)

    def _task(self, name: Key, index: int) -> Task:
        return Task(
            name,
            sum_key_counts,
            TaskRef((self.frame._name, index)),
        )


class MergeValidateFlag(Expr):
    """Tree-reduce the per-partition flags of one side to a single boolean.

    The reduction has constant fan-in (8), hence its depth and per-task state
    stay logarithmic / constant in the number of input partitions.
    """

    _parameters = ["frame", "split_every"]
    _defaults = {"split_every": 8}

    @functools.cached_property
    def _meta(self):
        return make_meta_like(
            pd.DataFrame({"_duplicates": pd.array([False], dtype=bool)})
        )

    def _divisions(self):
        return (None, None)

    @functools.cached_property
    def split_every(self):
        out = self.operand("split_every")
        if out is None:
            return 8
        return out

    def _lower(self):
        return None

    def _layer(self) -> dict:
        import toolz

        dsk = {}
        keys = [(self.frame._name, i) for i in range(self.frame.npartitions)]
        split_every = self.split_every
        level = 1
        while len(keys) > split_every:
            new_keys = []
            for i, batch in enumerate(toolz.partition_all(split_every, keys)):
                batch = list(batch)
                out_key = (self._name, level, i)
                dsk[out_key] = Task(
                    out_key,
                    combine_duplicate_flags,
                    *[TaskRef(k) for k in batch],
                )
                new_keys.append(out_key)
            level += 1
            keys = new_keys
        root = (self._name, 0)
        dsk[root] = Task(
            root,
            combine_duplicate_flags,
            *[TaskRef(k) for k in keys],
        )
        return dsk


class CardinalityBarrier(Blockwise):
    """Attach the cardinality check to every partition of the merge result.

    The data partitions pass through unchanged, but each one gains a
    broadcast dependency on the reduced left/right duplicate flags.  Because
    the dependency is attached at the *output* of the merge (after any
    projection/filter optimization), computing any single output partition
    still forces the full check.  The flag tasks hold constant-size state and
    the barrier itself only carries the two booleans.
    """

    _parameters = [
        "frame",
        "left_flag",
        "right_flag",
        "validate",
        "error_validate",
    ]
    _defaults = {"left_flag": None, "right_flag": None, "error_validate": None}
    _is_length_preserving = True
    # The check must not be reordered away or have filters/projections pushed
    # through it: it sits above the merge and observes exactly its inputs.
    _filter_passthrough = False
    _projection_passthrough = False

    @functools.cached_property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        return self.frame.divisions

    def _broadcast_dep(self, dep: Expr):
        # The single-partition flag expressions must be broadcast to every
        # output partition rather than partition-aligned.
        return dep is self.left_flag or dep is self.right_flag

    def _simplify_up(self, parent, dependents):
        # Optimization barrier: never let the optimizer remove or reorder this
        # node, otherwise the guarantee could be pruned away (e.g. by a
        # projection that drops the join keys or by a partition filter).
        return

    def _task(self, name: Key, index: int) -> Task:
        args = [TaskRef((self.frame._name, index))]
        if self.left_flag is not None:
            args.append(TaskRef((self.left_flag._name, 0)))
        else:
            args.append(None)
        if self.right_flag is not None:
            args.append(TaskRef((self.right_flag._name, 0)))
        else:
            args.append(None)
        args.append(self.validate)
        args.append(self.error_validate or self.validate)
        return Task(name, _barrier_impl, *args)


def _barrier_impl(part, left_flag, right_flag, validate, error_validate=None):
    raise_if_invalid(
        left_flag, right_flag, validate, error_validate or validate
    )
    return part


def make_meta_like(df):
    from dask.dataframe.dispatch import make_meta

    return make_meta(df)


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def build_duplicate_flag(side_expr, on, index, npartitions, shuffle_method):
    """Build the reduced duplicate-flag expression for one merge side.

    Parameters
    ----------
    side_expr:
        The *full* input expression of the merge at call time.  Filters pushed
        below the merge are already reflected here; filters above the merge
        are not, exactly matching pandas which validates the merge inputs.
    on, index:
        The join key specification for this side.
    npartitions:
        Number of hash buckets for the summary shuffle.
    """
    keyed = materialize_key_frame(side_expr, on, index)
    key_columns = list(keyed.columns)
    local = LocalKeyCounts(keyed, key_columns)
    partitioned = KeyCountsPartitioned(
        local, key_columns, npartitions_out=npartitions, shuffle_method=shuffle_method
    )
    flags = DuplicateFlag(partitioned, key_columns)
    return MergeValidateFlag(flags)


def apply_cardinality_check(
    lowered,
    left,
    right,
    left_on,
    right_on,
    left_index,
    right_index,
    validate,
    shuffle_method,
    error_validate=None,
):
    """Wrap a lowered merge expression with the distributed cardinality check.

    ``lowered`` is the physical merge plan; ``left``/``right`` are the full
    logical inputs that the merge was called with.

    Parameters
    ----------
    validate:
        The mode that determines *which sides* are checked.
    error_validate:
        The mode whose pandas MergeError wording is used when a violation is
        found. Defaults to ``validate``; differs when upstream evidence already
        proves one side unique, so only the other side is scanned but the
        original user-facing cardinality constraint is reported.
    """
    npartitions = lowered.npartitions
    left_flag = right_flag = None
    sides = _CHECKED_SIDES[validate]
    if "left" in sides:
        left_flag = build_duplicate_flag(
            left, left_on, left_index, npartitions, shuffle_method
        )
    if "right" in sides:
        right_flag = build_duplicate_flag(
            right, right_on, right_index, npartitions, shuffle_method
        )
    return CardinalityBarrier(
        lowered, left_flag, right_flag, validate, error_validate or validate
    )


def merge_side_key_sets(side_expr, on, index, output_columns):
    """Map a merge input's uniqueness evidence onto the merge output columns.

    The merge output retains every input data column under its original name
    (join-key columns are never suffixed), so column-key guarantees carry over
    as long as the columns are still present.  Index keys are not propagated
    because the output index does not preserve the input index identity.
    """
    if index:
        return frozenset()
    output = set(output_columns)
    return frozenset(
        key_set for key_set in side_expr.unique_key_sets if key_set <= output
    )


__all__ = [
    "CardinalityBarrier",
    "DuplicateFlag",
    "KeyCountsPartitioned",
    "LocalKeyCounts",
    "MergeValidateFlag",
    "apply_cardinality_check",
    "build_duplicate_flag",
    "materialize_key_frame",
    "merge_error_message",
    "normalize_validate",
    "raise_if_invalid",
]
