from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dask
from dask.dataframe.dask_expr import from_map, from_pandas
from dask.dataframe.dask_expr._expr import Filter
from dask.dataframe.dask_expr.tests._util import _backend_library, assert_eq

# Set DataFrame backend for this module
pd = _backend_library()


def flatten_grouped(result):
    """Drop pandas' group index level(s) and restore original row order.

    This is how grouped dask rolling results (flat original index) are
    compared against the pandas reference.
    """
    out = result.sort_index(level=-1)
    out.index = out.index.droplevel(list(range(out.index.nlevels - 1)))
    return out


def make_ddf(parts, meta=None):
    """Build a dask dataframe from a list of pandas partitions.

    Divisions are derived from the (globally sorted) index so that they
    are known and trusted; empty partitions get collapsed boundaries.
    """
    if meta is None:
        meta = pd.concat([p.head(0) for p in parts])
    lengths = [len(p) for p in parts]
    total_index = [idx for p in parts for idx in p.index]
    divisions = []
    pos = 0
    for n in lengths:
        divisions.append(total_index[pos] if pos < len(total_index) else 0)
        pos += n
    divisions.append(total_index[-1] if total_index else 0)
    # collapsed boundaries for empty partitions stay non-decreasing
    for i in range(1, len(divisions)):
        if divisions[i] < divisions[i - 1]:
            divisions[i] = divisions[i - 1]
    return from_map(
        lambda i: parts[i],
        range(len(parts)),
        meta=meta,
        divisions=tuple(divisions),
    )


@pytest.fixture
def hand_pdf():
    # The example from the task description: index 0..7, values 1..8,
    # group keys a b b b b b a a, two rows per partition.  Group ``a``
    # disappears for two whole partitions before showing up again.
    return pd.DataFrame(
        {"x": np.arange(1.0, 9.0), "g": list("abbbbbaa")},
        index=pd.RangeIndex(8),
    )


def test_grouped_rolling_sparse_group_handcomputed(hand_pdf):
    # Hand-computed: group a has rows at index 0 (1.0), 6 (7.0), 7 (8.0).
    # Its width-3 window at index 6 must reach back over two partitions
    # that contain no ``a`` row at all, and must not mistake the last two
    # rows of some earlier partition for ``a`` history.
    ddf = from_pandas(hand_pdf, npartitions=4)
    result = ddf.groupby("g", observed=True).rolling(3, min_periods=1).sum()
    out = result.compute()
    assert out.loc[0, "x"] == 1.0
    assert out.loc[6, "x"] == 8.0
    assert out.loc[7, "x"] == 16.0
    # group b rows are 2..6 at index 1..5
    assert out["x"].tolist() == [1.0, 2.0, 5.0, 9.0, 12.0, 15.0, 8.0, 16.0]
    # flat original index, no group level, no group-key column
    assert out.index.equals(hand_pdf.index)
    assert list(out.columns) == ["x"]

    expected = flatten_grouped(
        hand_pdf.groupby("g", observed=True).rolling(3, min_periods=1).sum()
    )
    assert_eq(result, expected)


def test_grouped_rolling_context_not_just_previous_partition():
    # A group whose previous row is two partitions back, and a partition
    # whose last rows belong to a *different* group: the context must be
    # the last same-group rows, not the last rows of the previous block.
    parts = [
        pd.DataFrame({"x": [1.0, 10.0], "g": ["b", "a"]}, index=[0, 1]),
        pd.DataFrame({"x": [2.0, 3.0], "g": ["b", "b"]}, index=[2, 3]),
        pd.DataFrame({"x": [20.0, 4.0], "g": ["a", "b"]}, index=[4, 5]),
        pd.DataFrame({"x": [30.0, 40.0], "g": ["a", "a"]}, index=[6, 7]),
    ]
    pdf = pd.concat(parts)
    ddf = make_ddf(parts)
    result = ddf.groupby("g", observed=True).rolling(2, min_periods=1).sum()
    out = result.compute()
    # group a: 10 (idx1), 20 (idx4), 30 (idx6), 40 (idx7)
    # group b: 1 (idx0), 2 (idx2), 3 (idx3), 4 (idx5)
    assert out["x"].tolist() == [1.0, 10.0, 3.0, 5.0, 30.0, 7.0, 50.0, 70.0]
    expected = flatten_grouped(
        pdf.groupby("g", observed=True).rolling(2, min_periods=1).sum()
    )
    assert_eq(result, expected)


def test_plain_rolling_small_partitions_handcomputed():
    # width-5 window over two-row partitions; no repartition required
    pdf = pd.DataFrame({"x": np.arange(1.0, 9.0)})
    ddf = from_pandas(pdf, npartitions=4)
    result = ddf.rolling(5).sum()
    out = result.compute()
    expected = pdf.rolling(5).sum()
    assert np.isnan(out["x"].tolist()[:4]).all()
    assert out["x"].tolist()[4:] == [15.0, 20.0, 25.0, 30.0]
    assert_eq(result, expected)


def test_map_overlap_supplies_and_keeps_exact_rows():
    # Directly observe which rows the callback receives on both sides and
    # which rows are finally kept, across empty and short partitions.
    sizes = [2, 0, 1, 3, 0, 2]
    parts, start = [], 0
    for n in sizes:
        idx = pd.Index(range(start, start + n), dtype="int64")
        parts.append(
            pd.DataFrame({"x": np.arange(start, start + n, dtype=float)}, index=idx)
        )
        start += n
    pdf = pd.concat(parts)
    ddf = make_ddf(parts)

    received = []

    def func(df):
        received.append(df.index.tolist())
        return df

    result = ddf.map_overlap(func, before=3, after=2)
    received.clear()  # ignore the metadata-inference call
    out = result.compute(scheduler="sync")

    # identity func: every business row is kept exactly once, nothing is
    # trimmed away by mistake even where context is short or missing
    assert out.index.tolist() == pdf.index.tolist()
    pd.testing.assert_frame_equal(out, pdf)

    # what each partition received: up to 3 rows before and 2 rows after,
    # clipped to what actually exists globally
    bounds = np.cumsum([0] + sizes)
    expected_received = []
    for i, n in enumerate(sizes):
        s, e = bounds[i], bounds[i + 1]
        lo = pdf.index[max(0, s - 3)]
        hi = pdf.index[min(len(pdf) - 1, e + 1)] if e + 1 < len(pdf) else pdf.index[-1]
        expected_received.append([v for v in pdf.index if lo <= v <= hi])
    assert sorted(received) == sorted(expected_received)


def test_map_overlap_keeps_rows_with_short_context():
    # before/after larger than the whole frame: only the rows that exist
    # may be dropped (i.e. none), requested length must not delete data
    pdf = pd.DataFrame({"x": np.arange(6, dtype=float)})
    ddf = from_pandas(pdf, npartitions=3)
    result = ddf.map_overlap(lambda df: df, before=100, after=100)
    assert_eq(result, pdf)

    sizes = []

    def count(df):
        out = df.copy()
        out["n"] = len(df)
        sizes.append(len(df))
        return out

    res = ddf.map_overlap(count, before=100, after=100)
    sizes.clear()  # ignore the metadata-inference call
    out = res.compute()
    # every partition sees the whole six-row frame as context+payload
    assert out["n"].tolist() == [6] * 6
    assert sorted(sizes) == [6] * 3


@pytest.mark.parametrize("scheduler", ["sync", "threads"])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("window,min_periods", [(1, None), (2, None), (3, 1), (4, 2), (5, 3), (7, None), (12, 1), (40, 1)])
def test_rolling_integer_window_vs_pandas(window, min_periods, center, scheduler):
    # 25 rows over partitions of sizes 4, 0, 7, 3, 11: windows span blocks
    rng = np.random.RandomState(42)
    sizes = [4, 0, 7, 3, 11]
    parts, start = [], 0
    for n in sizes:
        parts.append(
            pd.DataFrame(
                {"x": rng.randn(n), "y": rng.randn(n)},
                index=pd.RangeIndex(start, start + n),
            )
        )
        start += n
    pdf = pd.concat(parts)
    ddf = make_ddf(parts)

    kwargs = dict(min_periods=min_periods, center=center)
    for how, how_kwargs in [
        ("sum", {}),
        ("count", {}),
        ("mean", {}),
        ("min", {}),
        ("max", {}),
        ("var", {}),
        ("var", {"ddof": 0}),
        ("var", {"ddof": 2}),
        ("std", {}),
        ("std", {"ddof": 0}),
    ]:
        result = getattr(ddf.rolling(window, **kwargs), how)(**how_kwargs)
        expected = getattr(pdf.rolling(window, **kwargs), how)(**how_kwargs)
        assert_eq(result, expected, scheduler=scheduler, rtol=1e-10, atol=1e-10)
        # Series form
        result_s = getattr(ddf.x.rolling(window, **kwargs), how)(**how_kwargs)
        expected_s = getattr(pdf.x.rolling(window, **kwargs), how)(**how_kwargs)
        assert_eq(result_s, expected_s, scheduler=scheduler, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("window", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("center", [False, True])
def test_rolling_integer_window_datetime_index(window, center):
    idx = pd.date_range("2021-01-01", periods=30, freq="h")
    pdf = pd.DataFrame({"x": np.arange(30, dtype=float)}, index=idx)
    ddf = from_pandas(pdf, npartitions=6)
    result = ddf.rolling(window, center=center, min_periods=1).mean()
    expected = pdf.rolling(window, center=center, min_periods=1).mean()
    assert_eq(result, expected, rtol=1e-10, atol=1e-10)
    assert result.divisions == ddf.divisions


def test_rolling_window_longer_than_frame():
    pdf = pd.DataFrame({"x": np.arange(10, dtype=float)})
    ddf = from_pandas(pdf, npartitions=4)
    # default min_periods (= window): everything is NaN
    assert_eq(ddf.rolling(50).sum(), pdf.rolling(50).sum())
    # min_periods=1: expanding-like accumulation across all partitions
    assert_eq(
        ddf.rolling(50, min_periods=1).mean(), pdf.rolling(50, min_periods=1).mean()
    )
    assert_eq(
        ddf.rolling(50, min_periods=1, center=True).sum(),
        pdf.rolling(50, min_periods=1, center=True).sum(),
    )


def test_rolling_nan_metric():
    pdf = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0]}
    )
    ddf = from_pandas(pdf, npartitions=4)
    for how in ["sum", "count", "mean", "min", "max", "var", "std"]:
        assert_eq(
            getattr(ddf.rolling(4, min_periods=1), how)(),
            getattr(pdf.rolling(4, min_periods=1), how)(),
            rtol=1e-10,
            atol=1e-10,
        )


def make_grouped_parts():
    """Partitions where group 'a' vanishes for long stretches."""
    rng = np.random.RandomState(7)
    sizes = [3, 5, 0, 4, 6, 2, 5]
    groups_per_part = [
        ["a", "b", "a"],
        ["b", "c", "b", "c", "b"],
        [],
        ["c", "b", "c", "b"],
        ["b", "c", "b", "c", "b", "c"],
        ["d", "b"],
        ["a", "c", "a", "b", "a"],
    ]
    parts, start = [], 0
    for n, groups in zip(sizes, groups_per_part):
        x = rng.randn(n)
        x[x < -1.5] = np.nan  # a few missing measurements
        parts.append(
            pd.DataFrame(
                {
                    "x": x.astype("float64"),
                    "y": rng.randn(n).astype("float64"),
                    "g": np.asarray(list(groups), dtype=object),
                    "h": np.array(
                        [1 if i % 2 == 0 else 2 for i in range(n)], dtype=np.int64
                    ),
                },
                index=pd.RangeIndex(start, start + n),
            )
        )
        start += n
    return parts


@pytest.mark.parametrize("scheduler", ["sync", "threads"])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("window,min_periods", [(1, None), (2, 1), (3, 2), (4, 1), (6, 3), (30, 1)])
def test_groupby_rolling_integer_vs_pandas(window, min_periods, center, scheduler):
    parts = make_grouped_parts()
    pdf = pd.concat(parts)
    ddf = make_ddf(parts)
    kwargs = dict(min_periods=min_periods, center=center)
    for how, how_kwargs in [
        ("sum", {}),
        ("count", {}),
        ("mean", {}),
        ("min", {}),
        ("max", {}),
        ("var", {}),
        ("var", {"ddof": 0}),
        ("std", {}),
        ("std", {"ddof": 2}),
    ]:
        result = getattr(ddf.groupby("g", observed=True).rolling(window, **kwargs), how)(
            **how_kwargs
        )
        expected = flatten_grouped(
            getattr(pdf.groupby("g", observed=True).rolling(window, **kwargs), how)(
                **how_kwargs
            )
        )
        assert_eq(result, expected, scheduler=scheduler, rtol=1e-10, atol=1e-10)


def test_groupby_rolling_two_keys():
    parts = make_grouped_parts()
    pdf = pd.concat(parts)
    ddf = make_ddf(parts)
    for window, center in [(2, False), (3, True), (4, False), (5, True)]:
        result = ddf.groupby(["g", "h"], observed=True).rolling(
            window, min_periods=1, center=center
        ).mean()
        expected = flatten_grouped(
            pdf.groupby(["g", "h"], observed=True)
            .rolling(window, min_periods=1, center=center)
            .mean()
        )
        assert_eq(result, expected, rtol=1e-10, atol=1e-10)
        out = result.compute()
        assert out.index.nlevels == 1
        assert list(out.columns) == ["x", "y"]


def test_groupby_rolling_int64_keys():
    pdf = pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "k": [3, 1, 3, 1, 2, 2, 3, 1, 3, 2, 1, 3],
        }
    )
    ddf = from_pandas(pdf, npartitions=5)
    result = ddf.groupby("k", observed=True).rolling(3, min_periods=1).sum()
    expected = flatten_grouped(
        pdf.groupby("k", observed=True).rolling(3, min_periods=1).sum()
    )
    assert_eq(result, expected)
    assert result.compute().index.equals(pdf.index)


def test_groupby_rolling_series_slice():
    parts = make_grouped_parts()
    pdf = pd.concat(parts)
    ddf = make_ddf(parts)
    result = ddf.groupby("g", observed=True).x.rolling(3, min_periods=1).mean()
    expected = flatten_grouped(
        pdf.groupby("g", observed=True).x.rolling(3, min_periods=1).mean()
    )
    assert_eq(result, expected, rtol=1e-10, atol=1e-10)
    assert result.name == "x"


def test_groupby_rolling_window_longer_than_group():
    # window exceeds the size of every group, and one longer than the frame
    pdf = pd.DataFrame({"x": np.arange(10, dtype=float), "g": list("aabbaabbaa")})
    ddf = from_pandas(pdf, npartitions=4)
    for window in (4, 25):
        result = ddf.groupby("g", observed=True).rolling(window, min_periods=1).sum()
        expected = flatten_grouped(
            pdf.groupby("g", observed=True).rolling(window, min_periods=1).sum()
        )
        assert_eq(result, expected)
        # default min_periods (= window): only NaN where not enough rows
        result = ddf.groupby("g", observed=True).rolling(window).sum()
        expected = flatten_grouped(pdf.groupby("g", observed=True).rolling(window).sum())
        assert_eq(result, expected)


def test_rolling_empty_partitions_and_empty_frame():
    meta = pd.DataFrame(
        {"x": pd.Series([], dtype=float), "g": pd.Series([], dtype=str)}
    )

    def gen(i):
        if i in (0, 2, 4):
            return meta.copy()
        return pd.DataFrame(
            {"x": [float(i), 2 * float(i)], "g": ["a", "b"]}, index=[2 * i, 2 * i + 1]
        )

    ddf = make_ddf([gen(i) for i in range(6)], meta=meta)
    pdf = pd.concat([gen(i) for i in range(6)])
    assert_eq(ddf.x.rolling(3, min_periods=1).sum(), pdf.x.rolling(3, min_periods=1).sum())
    result = ddf.groupby("g", observed=True).rolling(2, min_periods=1).sum()
    expected = flatten_grouped(
        pdf.groupby("g", observed=True).rolling(2, min_periods=1).sum()
    )
    assert_eq(result, expected)

    # completely empty frame with known divisions
    empty = make_ddf([meta.copy() for _ in range(3)], meta=meta)
    result = empty.x.rolling(3).sum()
    assert_eq(result, meta["x"].rolling(3).sum())
    result_g = empty.groupby("g", observed=True).rolling(3).sum()
    assert result_g.compute().shape == (0, 1)
    assert list(result_g._meta.columns) == ["x"]
    assert result_g._meta.index.nlevels == 1


def test_rolling_partition_membership_and_divisions():
    pdf = pd.DataFrame(
        {"x": np.arange(20, dtype=float), "g": [f"g{i % 3}" for i in range(20)]}
    )
    ddf = from_pandas(pdf, npartitions=5)
    for result in [
        ddf[["x"]].rolling(4, min_periods=1).mean(),
        ddf.groupby("g", observed=True).rolling(3, min_periods=1).sum(),
    ]:
        assert result.npartitions == ddf.npartitions
        assert result.divisions == ddf.divisions
        for i in range(result.npartitions):
            part = result.partitions[i].compute()
            s, e = 4 * i, 4 * (i + 1)
            # every row stays in its original partition with its index
            assert part.index.tolist() == pdf.index.tolist()[s:e]
            assert result.partitions[i].divisions == (
                ddf.divisions[i],
                ddf.divisions[i + 1],
            )


def test_rolling_repartition_cutpoints():
    # the same frame partitioned differently must give the same rows
    pdf = pd.DataFrame(
        {"x": np.arange(30, dtype=float), "g": [f"g{i % 4}" for i in range(30)]}
    )
    expected_plain = pdf[["x"]].rolling(5, min_periods=1).mean()
    expected_grouped = flatten_grouped(
        pdf.groupby("g", observed=True).rolling(4, min_periods=1).sum()
    )
    for npartitions in (1, 2, 3, 7, 30):
        ddf = from_pandas(pdf, npartitions=npartitions)
        assert_eq(ddf[["x"]].rolling(5, min_periods=1).mean(), expected_plain)
        assert_eq(
            ddf.groupby("g", observed=True).rolling(4, min_periods=1).sum(),
            expected_grouped,
        )
    # uneven, hand-picked cut points including empty partitions
    sizes = [1, 16, 0, 12, 1]
    parts, start = [], 0
    for n in sizes:
        parts.append(pdf.iloc[start : start + n])
        start += n
    ddf = make_ddf(parts)
    assert_eq(ddf[["x"]].rolling(5, min_periods=1).mean(), expected_plain)
    assert_eq(
        ddf.groupby("g", observed=True).rolling(4, min_periods=1).sum(),
        expected_grouped,
    )


def test_rolling_expression_does_not_compute():
    calls = []

    def gen(i):
        calls.append(i)
        return pd.DataFrame(
            {"x": [1.0, 2.0], "g": ["a", "b"]}, index=[2 * i, 2 * i + 1]
        )

    meta = pd.DataFrame({"x": pd.Series([], dtype=float), "g": pd.Series([], dtype=str)})
    ddf = from_map(gen, range(4), meta=meta, divisions=(0, 2, 4, 6, 8))
    result = ddf.groupby("g", observed=True).rolling(3, min_periods=1).sum()
    _ = result._meta
    _ = result.divisions
    optimized = result.optimize()
    _ = optimized._meta
    assert calls == []
    # metadata is as expected without touching data
    assert list(result._meta.columns) == ["x"]
    assert result._meta.index.nlevels == 1


def test_rolling_column_projection():
    pdf = pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "y": np.arange(12, dtype=float) * 10,
            "g": [f"g{i % 3}" for i in range(12)],
        }
    )
    ddf = from_pandas(pdf, npartitions=4)
    # plain rolling projection
    result = ddf[["x", "y"]].rolling(3, min_periods=1).sum()["x"]
    expected = pdf[["x", "y"]].rolling(3, min_periods=1).sum()["x"]
    assert_eq(result, expected)
    q = result.simplify()
    eq = ddf["x"].rolling(3, min_periods=1).sum().simplify()
    assert q._name == eq._name

    # grouped: the group key must be supplied to the window and then
    # disappear from the output
    result = ddf.groupby("g", observed=True).rolling(3, min_periods=1).sum()["x"]
    expected = flatten_grouped(
        pdf.groupby("g", observed=True).rolling(3, min_periods=1).sum()
    )["x"]
    assert_eq(result, expected)
    optimal = (
        ddf[["x", "g"]].groupby("g", observed=True).rolling(3, min_periods=1).sum()["x"]
    )
    assert result.optimize()._name == optimal.optimize()._name
    # only the metric column comes out, no group key or internal marker
    out = result.compute()
    assert out.name == "x"
    assert out.index.equals(pdf.index)


def test_rolling_filter_stays_after_window():
    pdf = pd.DataFrame({"x": np.arange(1.0, 13.0)})
    ddf = from_pandas(pdf, npartitions=4)
    result = ddf.rolling(3, min_periods=1).sum()
    filtered = result[result.x > 5]
    expected = pdf.rolling(3, min_periods=1).sum()
    expected = expected[expected.x > 5]
    assert_eq(filtered, expected)
    # the filter must not be pushed below the window: if it were, the
    # window would see a different (filtered) set of rows
    simplified = filtered.simplify()
    assert isinstance(simplified.expr, Filter)
    # filtering the input first is a different, also supported, operation
    filtered_first = ddf[ddf.x > 5].rolling(3, min_periods=1).sum()
    expected_first = pdf[pdf.x > 5].rolling(3, min_periods=1).sum()
    assert_eq(filtered_first, expected_first)


def test_rolling_single_partition_compute():
    pdf = pd.DataFrame(
        {"x": np.arange(24, dtype=float), "g": [f"g{i % 3}" for i in range(24)]}
    )
    ddf = from_pandas(pdf, npartitions=6)
    for result, expected in [
        (
            ddf[["x"]].rolling(5, min_periods=1).sum(),
            pdf[["x"]].rolling(5, min_periods=1).sum(),
        ),
        (
            ddf.groupby("g", observed=True).rolling(4, min_periods=1).mean(),
            flatten_grouped(
                pdf.groupby("g", observed=True).rolling(4, min_periods=1).mean()
            ),
        ),
    ]:
        for i in (0, 2, 5):
            part = result.partitions[i]
            assert part.npartitions == 1
            assert part.divisions == (ddf.divisions[i], ddf.divisions[i + 1])
            out = part.compute()
            pd.testing.assert_frame_equal(
                out, expected.iloc[4 * i : 4 * (i + 1)], check_dtype=False
            )


def test_rolling_co_computation_and_persist():
    pdf = pd.DataFrame(
        {"x": np.arange(20, dtype=float), "g": [f"g{i % 4}" for i in range(20)]}
    )
    ddf = from_pandas(pdf, npartitions=5)
    r3 = ddf[["x"]].rolling(3, min_periods=1).sum()
    r5 = ddf[["x"]].rolling(5, min_periods=1).mean()
    rg = ddf.groupby("g", observed=True).rolling(3, min_periods=1).sum()
    other = ddf.x.mean()

    # different windows and group keys must not share context tasks
    names = {r3.optimize()._name, r5.optimize()._name, rg.optimize()._name}
    assert len(names) == 3
    dsk3 = set(r3.__dask_graph__())
    dsk5 = set(r5.__dask_graph__())
    dskg = set(rg.__dask_graph__())
    # source tasks are shared, overlap context tasks are not
    shared = dsk3 & dsk5
    assert shared
    assert not any("overlap" in str(k) for k in shared)
    assert not any("overlap" in str(k) for k in dsk3 & dskg)

    a, b, c, d = dask.compute(r3, r5, rg, other)
    pd.testing.assert_frame_equal(a, pdf[["x"]].rolling(3, min_periods=1).sum())
    pd.testing.assert_frame_equal(b, pdf[["x"]].rolling(5, min_periods=1).mean())
    pd.testing.assert_frame_equal(
        c,
        flatten_grouped(pdf.groupby("g", observed=True).rolling(3, min_periods=1).sum()),
    )
    assert d == pdf.x.mean()

    # persist works and gives the same result
    persisted = rg.persist()
    assert_eq(persisted, c)


def test_rolling_source_exception_propagates():
    pdf = pd.DataFrame(
        {"x": np.arange(12, dtype=float), "g": [f"g{i % 2}" for i in range(12)]}
    )
    ddf = from_pandas(pdf, npartitions=4)

    def boom(df, partition_info=None):
        if partition_info["number"] == 2:
            raise RuntimeError("boom-2")
        return df

    src = ddf.map_partitions(boom, meta=ddf._meta)
    with pytest.raises(RuntimeError, match="boom-2"):
        src.x.rolling(3).sum().compute()
    with pytest.raises(RuntimeError, match="boom-2"):
        src.groupby("g", observed=True).rolling(3).sum().compute()
    # a legal computation afterwards still works
    assert_eq(ddf.x.rolling(3).sum(), pdf.x.rolling(3).sum())


def test_rolling_single_partition_computes_only_needed_context():
    # computing one output partition pulls in the context chain up to that
    # partition, and nothing beyond it
    import dask.dataframe.dask_expr._expr as _expr_mod

    pdf = pd.DataFrame({"x": np.arange(12, dtype=float)})
    ddf = from_pandas(pdf, npartitions=6)
    result = ddf.rolling(4, min_periods=1).sum()

    calls = []
    orig = _expr_mod._tail_with_context

    def counting(prev, current, before, grouper):
        calls.append(len(current))
        return orig(prev, current, before, grouper)

    _expr_mod._tail_with_context = counting
    try:
        out = result.partitions[3].compute(scheduler="sync")
    finally:
        _expr_mod._tail_with_context = orig
    # the tail chain ran once per preceding partition (0..2), not more
    assert sorted(calls) == [2, 2, 2]
    pd.testing.assert_frame_equal(
        out, pdf.rolling(4, min_periods=1).sum().iloc[6:8]
    )


def test_map_overlap_multiple_empty_partitions():
    # empty partitions at the global head and tail as well as in between
    meta = pd.DataFrame({"x": pd.Series([], dtype=float)})
    parts = [
        meta.copy(),
        meta.copy(),
        pd.DataFrame({"x": [1.0, 2.0]}, index=[0, 1]),
        meta.copy(),
        pd.DataFrame({"x": [3.0]}, index=[2]),
        meta.copy(),
    ]
    ddf = make_ddf(parts, meta=meta)
    pdf = pd.concat(parts)
    received = []

    def func(df):
        received.append(df.index.tolist())
        return df

    result = ddf.map_overlap(func, before=2, after=2)
    received.clear()  # ignore the metadata-inference call
    out = result.compute(scheduler="sync")
    assert out.index.tolist() == pdf.index.tolist()
    # partitions around the empties still see the real neighboring rows
    flat = [idx for call in received for idx in call]
    assert set(flat) == set(pdf.index.tolist())
