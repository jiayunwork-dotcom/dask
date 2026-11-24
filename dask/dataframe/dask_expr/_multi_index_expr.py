"""Expressions implementing the two-level composite ``set_index``.

Public entry point: ``DataFrame.set_index([col0, col1], ...)``.

Planning and execution are deliberately separated:

* **Explicit divisions.** Constructing the expression, computing ``_meta``
  and ``optimize()`` never read the source or the data.
* **Automatic divisions.** The first time the result's divisions are needed
  (``.divisions``, ``optimize()``, ``persist()`` or ``compute()``) a small
  *summary* graph runs on the client.  Each input partition contributes at
  most :data:`COMPOSITE_SAMPLE_LIMIT` raw key tuples plus its low/high
  endpoint and row count.  The planner therefore never materialises a full
  key set.
* The actual exchange/sort is deferred to normal graph evaluation.  No task
  in the graph calls ``compute``/``persist`` or starts a scheduler.
"""

from __future__ import annotations

import functools
import uuid

import numpy as np
import tlz as toolz

from dask._task_spec import DataNode
from dask.dataframe.dask_expr._expr import (
    Blockwise,
    Expr,
    PartitionsFiltered,
    Projection,
    determine_column_projection,
)
from dask.dataframe.dask_expr._multi_index import (
    COMPOSITE_SAMPLE_LIMIT,
    CompositeIndexError,
    normalize_component_array,
    partition_ids,
    plan_divisions,
    summarize_partition,
    within_bounds,
)
from dask.dataframe.dask_expr._repartition import Repartition
from dask.dataframe.dask_expr._shuffle import (
    DiskShuffle,
    SortIndexBlockwise,
    TaskShuffle,
)
from dask.dataframe.dask_expr._skew_join import _RefCountedPartd
from dask.dataframe.dask_expr._util import LRU, _convert_to_list
from dask.dataframe.shuffle import (
    barrier,
    collect,
    maybe_buffered_partd,
)
from dask.dataframe.dispatch import partd_encode_dispatch
from dask.utils import get_default_shuffle_method

#: Diagnostic record for the most recent automatic planning run.  Used by the
#: scale experiments to report sample counts and planner state sizes; not a
#: public API.
LAST_PLAN_STATS: dict = {}

_auto_divisions_lru: LRU = LRU(32)


def _summarize_block(part, columns, limit, kinds):
    return summarize_partition(part, columns, limit, kinds)


def _plan_from_summaries(results, target):
    """Client-side reduction over per-partition summaries."""
    total_rows = 0
    samples: list = []
    low = None
    high = None
    for count, keys, part_low, part_high in results:
        total_rows += int(count)
        samples.extend(keys)
        if part_low is not None and (low is None or part_low < low):
            low = part_low
        if part_high is not None and (high is None or part_high > high):
            high = part_high
    if low is None or high is None:
        return (), 0, 0
    boundaries = plan_divisions(samples, (low, high), target)
    return boundaries, total_rows, len(samples)


def _plain_key(first, second):
    if isinstance(first, np.generic):
        first = first.item()
    if isinstance(second, np.generic):
        second = second.item()
    return (first, second)


class CompositePartitionId(Blockwise):
    """Assign each row its output partition id from the composite key."""

    _parameters = [
        "frame",
        "columns",
        "boundaries",
        "kinds",
        "partition_column",
        "validate_range",
    ]
    _defaults = {"validate_range": True}
    _preserves_partitioning_information = False

    @staticmethod
    def operation(df, columns, boundaries, kinds, partition_column, validate_range):
        first = normalize_component_array(
            df[columns[0]], kinds[0], where=str(columns[0])
        )
        second = normalize_component_array(
            df[columns[1]], kinds[1], where=str(columns[1])
        )
        if len(df):
            mask = within_bounds(first, second, boundaries)
            if validate_range and not mask.all():
                bad = int(np.flatnonzero(~mask)[0])
                raise CompositeIndexError(
                    "the data contains keys outside the explicit divisions; "
                    f"key {_plain_key(first[bad], second[bad])!r} is not in "
                    f"[{boundaries[0]!r}, {boundaries[-1]!r}]. Out-of-range "
                    "rows are never dropped or clamped into the final "
                    "partition."
                )
        ids = partition_ids(first, second, boundaries)
        out = df.copy(deep=False)
        out[partition_column] = ids
        _record_shuffle_bytes(out, partition_column)
        return out


#: Optional, environment-driven instrumentation for the scale experiments.
#: When this points at a directory, each ``CompositePartitionId`` task appends
#: a JSON record with the rows and payload bytes it is about to hand to the
#: exchange.  Environment based so worker processes pick it up without
#: embedding mutable state (or file paths) in the tokenized graph.
_COMPOSITE_INSTRUMENT_ENV = "DASK_COMPOSITE_INDEX_INSTRUMENT"


def _record_shuffle_bytes(frame, partition_column):
    import os

    directory = os.environ.get(_COMPOSITE_INSTRUMENT_ENV)
    if not directory:
        return
    import json
    import uuid

    from dask.sizeof import sizeof

    payload = frame.drop(columns=[partition_column])
    record = {
        "pid": os.getpid(),
        "rows": int(len(frame)),
        "bytes": int(sizeof(payload)),
    }
    name = f"shuffle-bytes-{uuid.uuid4().hex}.json"
    try:
        with open(os.path.join(directory, name), "w") as handle:
            json.dump(record, handle)
    except OSError:
        return


class _DiskRefCountedPartd(_RefCountedPartd):
    """Reference-counted partd wrapper accepting the shuffle ``fsync`` kwarg."""

    def append(self, data, **kwargs):
        from dask.dataframe.shuffle import ensure_cleanup_on_exception

        with ensure_cleanup_on_exception(self._store):
            return self._store.append(data, **kwargs)


def _make_disk_refcounted_partd(factory, expected_collects):
    return _DiskRefCountedPartd(factory(), expected_collects)


class CompositeDiskShuffle(DiskShuffle):
    """Disk shuffle that deterministically removes its partd store.

    The generic disk shuffle relies on garbage collection/``atexit`` to remove
    its temporary directory.  The composite path counts the materialized
    ``collect`` tasks that were actually built (culling changes that count)
    and removes the directory once the last one has run -- on success and on
    a source error alike.
    """

    @staticmethod
    def _shuffle_group(df, col, _filter, p):
        from dask.dataframe.shuffle import ensure_cleanup_on_exception

        with ensure_cleanup_on_exception(p):
            g = df.groupby(col)
            d = {i: g.get_group(i) for i in g.groups if i in _filter}
            p.append(d, fsync=True)

    def _layer(self):
        column = self.partitioning_index
        df = self.frame
        always_new_token = uuid.uuid1().hex
        p = ("zpartd-" + always_new_token,)
        encode_cls = partd_encode_dispatch(df._meta)
        n_collects = len(self._partitions)
        dsk1 = {
            p: (
                _make_disk_refcounted_partd,
                maybe_buffered_partd(encode_cls=encode_cls),
                n_collects,
            )
        }
        name = "shuffle-partition-" + always_new_token
        dsk2 = {
            (name, i): (self._shuffle_group, key, column, self._partitions, p)
            for i, key in enumerate(df.__dask_keys__())
        }
        barrier_token = ("barrier-" + always_new_token,)
        dsk3 = {barrier_token: (barrier, list(dsk2))}
        dsk4 = {
            (self._name, j): (collect, p, k, df._meta, barrier_token)
            for j, k in enumerate(self._partitions)
        }
        return toolz.merge(dsk1, dsk2, dsk3, dsk4)


class CompositeSetIndexPost(Blockwise):
    """Set the named ``MultiIndex`` on each already-shuffled partition."""

    _parameters = ["frame", "_columns", "drop", "column_dtype", "divisions"]
    _keyword_only = ["divisions"]
    _is_length_preserving = True

    @staticmethod
    def operation(df, columns, drop, column_dtype, divisions=None):
        out = df.set_index(list(columns), drop=drop)
        out.columns = out.columns.astype(column_dtype)
        return out

    def _get_culled_divisions(self, divisions):
        if self.frame.npartitions < len(divisions) - 1:
            part_filter = list(self.frame.find_operations(PartitionsFiltered))
            if len(part_filter) > 0:
                return tuple(
                    [div for i, div in enumerate(divisions) if i in part_filter[0]._partitions]
                    + [divisions[-1]]
                )
            else:
                return self.frame.divisions
        return divisions

    def _divisions(self):
        return self._get_culled_divisions(tuple(self.operand("divisions")))

    def _simplify_up(self, parent, dependents):
        if isinstance(parent, Projection):
            columns = determine_column_projection(
                self,
                parent,
                dependents,
                additional_columns=list(self.operand("_columns")),
            )
            if not columns:
                return
            columns = _convert_to_list(columns)
            if self.frame.columns == columns:
                return
            columns = [col for col in self.frame.columns if col in columns]
            if not set(self.operand("_columns")) <= set(columns):
                return
            return type(parent)(
                type(self)(self.frame[columns], *self.operands[1:]),
                parent.operand("columns"),
            )


class CompositeEmptyIndex(Expr):
    """A single empty partition carrying a two-level ``MultiIndex`` meta."""

    _parameters = ["meta"]

    @functools.cached_property
    def _meta(self):
        return self.operand("meta")

    def _divisions(self):
        return (None, None)

    @property
    def npartitions(self):
        return 1

    def _layer(self):
        name = self._name
        return {(name, 0): DataNode((name, 0), self._meta)}


class SetIndexComposite(Expr):
    """Sort a frame by a two-level composite index."""

    _parameters = [
        "frame",
        "_columns",
        "drop",
        "user_divisions",
        "npartitions",
        "kinds",
        "shuffle_method",
        "options",
    ]
    _defaults = {
        "drop": True,
        "user_divisions": None,
        "npartitions": None,
        "shuffle_method": None,
        "options": None,
    }
    _filter_passthrough = True
    _is_length_preserving = True

    @functools.cached_property
    def _meta(self):
        return self.frame._meta.set_index(list(self._columns), drop=self.drop)

    @property
    def _projection_columns(self):
        return list(self._meta.columns) + list(self._columns)

    @property
    def _target_npartitions(self):
        target = self.operand("npartitions")
        if target is not None:
            return int(target)
        return self.frame.npartitions

    @property
    def npartitions(self):
        divisions = self._divisions()
        if not divisions:
            return 1
        return len(divisions) - 1

    def _divisions(self):
        if self.user_divisions is not None:
            return tuple(self.user_divisions)
        boundaries = self._auto_divisions()
        if not boundaries:
            return (None, None)
        return tuple(boundaries)

    def _auto_divisions(self):
        cache_key = (self._name, self._target_npartitions)
        cached = _auto_divisions_lru.get(cache_key)
        if cached is not None:
            return cached
        import time

        from dask import compute, delayed

        from dask.dataframe.dask_expr._collection import new_collection

        columns = list(self._columns)
        kinds = list(self.operand("kinds"))
        parts = new_collection(self.frame[columns]).to_delayed()
        summaries = [
            delayed(_summarize_block)(part, columns, COMPOSITE_SAMPLE_LIMIT, kinds)
            for part in parts
        ]
        started = time.perf_counter()
        results = compute(*summaries) if summaries else ()
        elapsed = time.perf_counter() - started
        boundaries, total_rows, sample_count = _plan_from_summaries(
            results, self._target_npartitions
        )
        LAST_PLAN_STATS.clear()
        LAST_PLAN_STATS.update(
            {
                "input_partitions": self.frame.npartitions,
                "target_partitions": self._target_npartitions,
                "output_partitions": max(len(boundaries) - 1, 0),
                "rows": total_rows,
                "uploaded_keys": sample_count,
                "planner_state_keys": len(set(boundaries)) if boundaries else 0,
                "planner_sample_bound": self.frame.npartitions
                * (COMPOSITE_SAMPLE_LIMIT + 2),
                "summary_seconds": elapsed,
            }
        )
        _auto_divisions_lru[cache_key] = boundaries
        return boundaries

    def _lower(self):
        boundaries = tuple(self._divisions())
        if not boundaries:
            return CompositeEmptyIndex(self._meta)

        columns = list(self._columns)
        kinds = list(self.operand("kinds"))
        frame = self.frame
        # ``dask.dataframe.shuffle.shuffle_group`` treats a column named
        # exactly ``_partitions`` as an explicit target-partition id (rather
        # than hashing it).  We must use that name, so temporarily rename a
        # user column that already owns it.
        from dask.dataframe.dask_expr._expr import RenameFrame

        partition_column = "_partitions"
        rename = None
        if partition_column in frame.columns:
            temp = "_composite_partitions_user"
            while temp in frame.columns:
                temp += "_"
            rename = (temp, partition_column)
            frame = RenameFrame(frame, {partition_column: temp})

        assigned = CompositePartitionId(
            frame,
            columns,
            boundaries,
            kinds,
            partition_column,
            True,
        )
        n_out = len(boundaries) - 1
        if n_out < assigned.npartitions:
            assigned = Repartition(assigned, new_partitions=n_out)
        if self.shuffle_method == "disk":
            shuffled = CompositeDiskShuffle(
                assigned, partition_column, n_out, True, self.options
            )
        else:
            shuffled = TaskShuffle(
                assigned, partition_column, n_out, True, self.options
            )
        projected = Projection(
            shuffled,
            [c for c in assigned.columns if c != partition_column],
        )
        if rename is not None:
            projected = RenameFrame(projected, {rename[0]: rename[1]})
        index_set = CompositeSetIndexPost(
            projected,
            columns,
            self.drop,
            self.frame._meta.columns.dtype,
            tuple(boundaries),
        )
        return SortIndexBlockwise(index_set)

    def _simplify_up(self, parent, dependents):
        from dask.dataframe.dask_expr._expr import Filter

        if isinstance(parent, Projection):
            columns = determine_column_projection(
                self,
                parent,
                dependents,
                additional_columns=list(self._columns),
            )
            if not columns:
                return
            columns = _convert_to_list(columns)
            columns = [c for c in self.frame.columns if c in columns]
            if not set(self._columns) <= set(columns):
                return
            if self.frame.columns == columns:
                return
            return type(parent)(
                type(self)(self.frame[columns], *self.operands[1:]),
                parent.operand("columns"),
            )
        if isinstance(parent, Filter) and self._filter_passthrough_available(
            parent, dependents
        ):
            return self._filter_simplification(parent)

    def _filter_passthrough_available(self, parent, dependents):
        from dask.dataframe.dask_expr._expr import Index, is_filter_pushdown_available

        if is_filter_pushdown_available(self, parent, dependents):
            predicate = parent.predicate
            return not any(isinstance(x, Index) for x in predicate.walk())
        return False


def resolve_shuffle_method(shuffle_method) -> str:
    """Resolve and validate the exchange method *before* reading the source."""
    resolved = (
        shuffle_method if shuffle_method is not None else get_default_shuffle_method()
    )
    if resolved not in ("tasks", "disk"):
        raise ValueError(
            "composite set_index supports shuffle_method 'tasks' or 'disk'; "
            f"got {shuffle_method!r} (resolved to {resolved!r})"
        )
    return resolved
