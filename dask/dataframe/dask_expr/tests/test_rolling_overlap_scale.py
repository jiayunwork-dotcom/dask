from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import dask.dataframe.dask_expr._expr as _expr_mod
from dask._task_spec import convert_legacy_graph
from dask.dataframe import methods
from dask.dataframe.dask_expr import from_map
from dask.dataframe.dask_expr.tests._util import _backend_library

pd = _backend_library()

ROWS_PER_PART = 2000
GROUPS = ("a", "b", "c", "d")


def scale_gen(i, nparts, rows=ROWS_PER_PART):
    """Deterministically generate one partition of the scale input.

    Groups b/c/d appear throughout; group ``a`` shows up in the first
    partition and then only again in the very last one, so its window
    context has to be carried across the whole input.
    """
    rng = np.random.RandomState(1234 + i)
    idx = np.arange(i * rows, (i + 1) * rows)
    x = rng.uniform(-1.0, 1.0, rows)
    x[rng.rand(rows) < 0.05] = np.nan
    g = np.array(["b", "c", "d"], dtype=object)[rng.randint(0, 3, rows)]
    if i == 0:
        g[:10] = "a"
    if i == nparts - 1:
        g[-10:] = "a"
    return pd.DataFrame({"x": x, "g": g}, index=idx)


def make_scale_ddf(nparts):
    meta = pd.DataFrame(
        {"x": pd.Series([], dtype=float), "g": pd.Series([], dtype=object)}
    )
    divisions = tuple(i * ROWS_PER_PART for i in range(nparts)) + (
        nparts * ROWS_PER_PART - 1,
    )
    return from_map(
        lambda i: scale_gen(i, nparts),
        range(nparts),
        meta=meta,
        divisions=divisions,
        label="scalegen",
    )


def flatten_grouped(result):
    out = result.sort_index(level=-1)
    out.index = out.index.droplevel(list(range(out.index.nlevels - 1)))
    return out


def mirror_plain(parts, window, min_periods):
    """Per-partition plain-rolling reference with running tail context."""
    tails = None
    for i, part in enumerate(parts):
        combined = methods.concat([tails, part]) if tails is not None else part
        ref = combined.rolling(window, min_periods=min_periods).sum()
        n_context = 0 if tails is None else len(tails)
        yield i, ref.iloc[n_context:]
        tails = combined.tail(window - 1)


def mirror_grouped(parts, keys, window, min_periods):
    """Per-partition grouped-rolling reference with running per-group tails."""
    tails = None
    for i, part in enumerate(parts):
        combined = methods.concat([tails, part]) if tails is not None else part
        ref = flatten_grouped(
            combined.groupby(keys, observed=True)
            .rolling(window, min_periods=min_periods)
            .sum()
        )
        n_context = 0 if tails is None else len(tails)
        yield i, ref.iloc[n_context:]
        tails = combined.groupby(keys, observed=True).tail(window - 1)


class OverlapRecorder:
    """Records what the overlap tasks hold, transfer and read."""

    def __init__(self):
        self.peak_held = 0  # max business rows alive inside one new task
        self.chain_out_rows = 0  # rows handed between chain tasks
        self.chain_max_state = 0  # largest context state produced
        self.combined_rows = 0  # rows flowing into the per-partition func
        self.result_rows = 0  # rows kept after trimming

    @property
    def transferred_rows(self):
        return self.chain_out_rows + self.combined_rows


def patch_overlap(recorder):
    orig_tail = _expr_mod._tail_with_context
    orig_head = _expr_mod._head_with_context
    orig_combined = _expr_mod._combined_parts
    orig_chunk = _expr_mod.overlap_chunk

    def tail(prev, current, before, grouper):
        out = orig_tail(prev, current, before, grouper)
        in_rows = (0 if prev is None else len(prev)) + len(current)
        recorder.chain_out_rows += len(out)
        recorder.chain_max_state = max(recorder.chain_max_state, len(out))
        recorder.peak_held = max(recorder.peak_held, in_rows + len(out))
        return out

    def head(current, next_, after, grouper):
        out = orig_head(current, next_, after, grouper)
        in_rows = len(current) + (0 if next_ is None else len(next_))
        recorder.chain_out_rows += len(out)
        recorder.chain_max_state = max(recorder.chain_max_state, len(out))
        recorder.peak_held = max(recorder.peak_held, in_rows + len(out))
        return out

    def combined(prev, current, next_, before, after):
        out = orig_combined(prev, current, next_, before, after)
        in_rows = sum(len(p) for p in (prev, current, next_) if p is not None)
        recorder.peak_held = max(recorder.peak_held, in_rows + len(out[0]))
        return out

    def chunk(func, before, after, *args, **kwargs):
        combined = args[0][0]
        out = orig_chunk(func, before, after, *args, **kwargs)
        recorder.combined_rows += len(combined)
        recorder.result_rows += len(out)
        # func builds a same-length result next to the combined frame
        recorder.peak_held = max(recorder.peak_held, 2 * len(combined))
        return out

    return orig_tail, orig_head, orig_combined, orig_chunk, (tail, head, combined, chunk)


def graph_stats(expr):
    dsk = convert_legacy_graph(expr.__dask_graph__())
    deps = {k: set(v.dependencies) for k, v in dsk.items()}
    nodes = len(deps)
    edges = sum(len(v) for v in deps.values())
    # critical path length (longest dependency chain)
    memo = {}

    def depth(key):
        if key not in memo:
            ds = deps.get(key, ())
            memo[key] = 1 + max((depth(d) for d in ds if d in deps), default=0)
        return memo[key]

    critical = max((depth(k) for k in deps), default=0)
    # how often each input partition of the overlap machinery is read:
    # count references made by the overlap tasks (context chains and
    # per-partition combiners) to partitions outside the overlap layer
    overlap_prefixes = ("overlap-prepend-", "overlap-append-", "createoverlappingpartitions-")
    overlap_tasks = {
        k for k in deps if isinstance(k, tuple) and str(k[0]).startswith(overlap_prefixes)
    }
    reads = Counter()
    for k in overlap_tasks:
        for d in deps[k]:
            if d not in overlap_tasks:
                reads[d] += 1
    return {
        "nodes": nodes,
        "edges": edges,
        "critical_path": critical,
        "max_source_reads": max(reads.values(), default=0),
        "source_partitions": len(reads),
    }


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("window", [17, 257])
@pytest.mark.parametrize("nparts", [16, 64, 256])
def test_rolling_overlap_scale(nparts, window, grouped, tmp_path):
    B = ROWS_PER_PART
    G = len(GROUPS) if grouped else 1
    W = window

    ddf = make_scale_ddf(nparts)
    if grouped:
        result = ddf.groupby("g", observed=True).rolling(W, min_periods=1).sum()
    else:
        result = ddf[["x"]].rolling(W, min_periods=1).sum()

    recorder = OverlapRecorder()
    orig_tail, orig_head, orig_combined, orig_chunk, patched = patch_overlap(recorder)
    stats = graph_stats(result)
    try:
        _expr_mod._tail_with_context, _expr_mod._head_with_context = patched[:2]
        _expr_mod._combined_parts, _expr_mod.overlap_chunk = patched[2], patched[3]
        persisted = result.persist(scheduler="sync")
    finally:
        (
            _expr_mod._tail_with_context,
            _expr_mod._head_with_context,
            _expr_mod._combined_parts,
            _expr_mod.overlap_chunk,
        ) = (orig_tail, orig_head, orig_combined, orig_chunk)

    # check every output partition against the per-partition mirror
    # reference; only one partition is materialized test-side at a time
    parts = (scale_gen(i, nparts) for i in range(nparts))
    if grouped:
        mirror = mirror_grouped(parts, ["g"], W, 1)
    else:
        mirror = mirror_plain((p[["x"]] for p in parts), W, 1)
    for i, expected in mirror:
        out = persisted.partitions[i].compute(scheduler="sync")
        pd.testing.assert_frame_equal(out, expected, rtol=1e-10, atol=1e-10)

    # ---- capacity bounds -------------------------------------------------
    # a new task never holds more than O(B + G*W) business rows (the
    # factor 2 accounts for the input frame plus its result copy)
    bound = 2 * (B + G * W)
    metrics = {
        "nparts": nparts,
        "window": W,
        "grouped": grouped,
        "B_plus_GW": B + G * W,
        "peak_held": recorder.peak_held,
        "chain_max_state": recorder.chain_max_state,
        "transferred_rows": recorder.transferred_rows,
        "combined_rows": recorder.combined_rows,
        "result_rows": recorder.result_rows,
        **stats,
    }
    assert recorder.peak_held <= bound, metrics
    # the chain state is at most G*(W-1) context rows, independent of how
    # many partitions or history rows precede it
    assert recorder.chain_max_state <= G * (W - 1), metrics
    # graph grows linearly with the number of partitions, not quadratically
    assert stats["nodes"] <= 6 * nparts + 10, metrics
    assert stats["edges"] <= 8 * nparts + 10, metrics
    assert stats["critical_path"] <= nparts + 10, metrics
    # every source partition is read at most three times: once by each
    # context chain and once by its own combined task
    assert stats["max_source_reads"] <= 3, metrics
    # transferred rows stay linear in nparts * (B + G*W)
    assert recorder.transferred_rows <= 4 * nparts * (B + G * W), metrics

    print("\nSCALE " + json.dumps(metrics, sort_keys=True))
    out_file = os.environ.get("SCALE_METRICS_OUT")
    if out_file:
        with open(out_file, "a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
