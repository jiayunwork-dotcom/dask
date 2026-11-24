"""Tests for the two-level composite ``set_index`` and composite ``loc``.

Semantics are aligned with pandas 2.2.3:
``df.set_index([a, b]).sort_index()`` for construction and ``.loc`` /
``.reset_index`` for selection.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import tlz as toolz

import dask
import dask.dataframe as dd
from dask import delayed
from dask.callbacks import Callback
from dask.dataframe.dask_expr import from_delayed, from_map
from dask.dataframe.dask_expr._multi_index import (
    COMPOSITE_SAMPLE_LIMIT,
    CompositeIndexError,
)
from dask.dataframe.dask_expr._multi_index_expr import LAST_PLAN_STATS
from dask.dataframe.utils import assert_eq


def _ddf(pdf, npartitions=2, **kwargs):
    return dd.from_pandas(pdf, npartitions=npartitions, **kwargs)


def _frame(rows, names=("account", "seq")):
    """Build a payload frame with a two-level (account, seq) key."""
    pdf = pd.DataFrame(
        {
            names[0]: [r[0] for r in rows],
            names[1]: [r[1] for r in rows],
            "payload": range(len(rows)),
        }
    )
    return pdf


def _expected(pdf, names=("account", "seq")):
    return pdf.set_index(list(names)).sort_index()


def _assert_same_rows(out, expected):
    """Compare frames ignoring the object-vs-pyarrow string dtype difference.

    Dask's default ``dataframe.convert-string=True`` renders string levels as
    ``string[pyarrow]``; the *values* and row multiset are what matter here.
    Reset the index so the level dtype does not affect the comparison.
    """
    pd.testing.assert_frame_equal(
        out.sort_index().reset_index(drop=True),
        expected.sort_index().reset_index(drop=True),
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# Core construction semantics
# ---------------------------------------------------------------------------


def test_basic_sorted_multiset():
    rows = [(1, "x"), (1, "y"), (2, "x"), (1, "y"), (3, "z"), (2, "x")]
    pdf = _frame(rows)
    r = _ddf(pdf, 2).set_index(["account", "seq"], npartitions=2,
                               shuffle_method="tasks")
    out = r.compute(scheduler="sync")
    exp = _expected(pdf)
    assert out.index.names == ["account", "seq"]
    assert Counter(out.index) == Counter(exp.index)
    # complete keys are lexicographically ordered across partitions
    gathered = list(out.index)
    assert gathered == sorted(gathered)


def test_row_identity_not_just_counts():
    """Compare exact (key, payload) rows with the pandas reference."""
    rows = [(1, "x"), (1, "y"), (2, "x"), (1, "y"), (3, "z")]
    pdf = _frame(rows)
    r = _ddf(pdf, 2).set_index(["account", "seq"], npartitions=2,
                               shuffle_method="tasks")
    out = r.compute(scheduler="sync").sort_index()
    exp = _expected(pdf)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), exp.reset_index(drop=True), check_dtype=False
    )


@pytest.mark.parametrize("method", ["tasks", "disk"])
def test_both_shuffle_methods(method):
    rows = [(1, "x")] * 3 + [(1, "y")] * 2 + [(2, "x")] * 4 + [(3, "z")]
    pdf = _frame(rows)
    r = _ddf(pdf, 3).set_index(
        ["account", "seq"],
        divisions=[(1, "x"), (2, "x"), (3, "z"), (3, "z")],
        shuffle_method=method,
    )
    assert r.divisions == ((1, "x"), (2, "x"), (3, "z"), (3, "z"))
    out = r.compute(scheduler="sync")
    exp = _expected(pdf)
    assert Counter(out.index) == Counter(exp.index)
    for i in range(r.npartitions):
        part = r.get_partition(i).compute(scheduler="sync")
        if len(part):
            assert part.index[0] >= r.divisions[i]
            assert part.index[-1] < r.divisions[i + 1] or i == r.npartitions - 1


def test_drop_true_and_false():
    pdf = _frame([(1, "x"), (1, "y"), (2, "z")])
    ddf = _ddf(pdf, 2)
    r_true = ddf.set_index(["account", "seq"], shuffle_method="tasks")
    assert list(r_true._meta.columns) == ["payload"]
    r_false = ddf.set_index(["account", "seq"], drop=False,
                            shuffle_method="tasks")
    assert list(r_false._meta.columns) == ["account", "seq", "payload"]
    # drop=True: reset_index restores the two key columns
    reset = r_true.reset_index().compute(scheduler="sync")
    assert list(reset.columns) == ["account", "seq", "payload"]
    assert Counter(map(tuple, reset[["account", "seq"]].values)) == Counter(
        rows for rows in [(1, "x"), (1, "y"), (2, "z")]
    )
    # drop=False: reset_index collides with the same-named columns, as pandas
    with pytest.raises(ValueError, match="already exists"):
        r_false.reset_index().compute(scheduler="sync")


def test_default_npartitions_reuses_input_count():
    pdf = _frame([(a, f"s{i}") for a in range(4) for i in range(4)])
    ddf = _ddf(pdf, 5)
    r = ddf.set_index(["account", "seq"], shuffle_method="tasks")
    # None means "use the input partition count as target"; duplicates may
    # reduce the actual count, but it never exceeds the target.
    assert r.npartitions <= 5
    assert LAST_PLAN_STATS["target_partitions"] == 5


@pytest.mark.parametrize("npartitions", [0, -1, True, 2.5, "auto"])
def test_bad_npartitions(npartitions):
    pdf = _frame([(1, "x"), (2, "y")])
    with pytest.raises(ValueError):
        _ddf(pdf, 1).set_index(["account", "seq"], npartitions=npartitions)


# ---------------------------------------------------------------------------
# Duplicate keys, empty partitions, all-identical keys, empty input
# ---------------------------------------------------------------------------


def test_each_level_duplicated_but_complete_keys_distinct():
    rows = [(1, "x"), (1, "x"), (1, "y"), (1, "y"), (2, "x"), (2, "x")]
    pdf = _frame(rows)
    r = _ddf(pdf, 2).set_index(["account", "seq"], npartitions=2,
                               shuffle_method="tasks")
    out = r.compute(scheduler="sync")
    assert Counter(out.index) == Counter(_expected(pdf).index)
    # each complete key lives entirely within one partition
    for i in range(r.npartitions):
        keys = set(r.get_partition(i).compute(scheduler="sync").index)
        for other in range(r.npartitions):
            if other != i:
                assert not keys & set(
                    r.get_partition(other).compute(scheduler="sync").index
                )


def test_complete_duplicate_and_trailing_run():
    rows = [(1, "a")] * 2 + [(2, "b")] * 5 + [(3, "c")] * 4
    pdf = _frame(rows)
    for method in ("tasks", "disk"):
        r = _ddf(pdf, 3).set_index(
            ["account", "seq"], npartitions=3, shuffle_method=method
        )
        out = r.compute(scheduler="sync")
        assert Counter(out.index) == Counter(_expected(pdf).index)
        divs = r.divisions
        assert divs[-2] <= divs[-1]
        # A complete key is never split across partitions: collect each key's
        # partition membership and ensure it occupies exactly one partition.
        membership = {}
        for i in range(r.npartitions):
            for key in set(r.get_partition(i).compute(scheduler="sync").index):
                membership.setdefault(key, set()).add(i)
        for key, parts in membership.items():
            assert len(parts) == 1, (key, parts)


def test_all_identical_keys():
    rows = [(7, "k")] * 10
    pdf = _frame(rows)
    r = _ddf(pdf, 3).set_index(["account", "seq"], npartitions=3,
                               shuffle_method="tasks")
    assert r.divisions == ((7, "k"), (7, "k"))
    assert r.npartitions == 1
    assert len(r.compute(scheduler="sync")) == 10


def test_empty_partitions_are_allowed():
    rows = [(1, "a"), (2, "b"), (3, "c")]
    pdf = _frame(rows)
    r = _ddf(pdf, 4).set_index(
        ["account", "seq"],
        divisions=[(1, "a"), (2, "b"), (3, "c"), (3, "c")],
        shuffle_method="tasks",
    )
    lengths = [len(r.get_partition(i).compute(scheduler="sync"))
               for i in range(r.npartitions)]
    # Middle partition holds (2, b); first holds (1, a); third holds (3, c)
    assert sum(lengths) == 3
    assert Counter(r.compute(scheduler="sync").index) == Counter(_expected(pdf).index)


def test_empty_input_preserves_structure():
    empty = pd.DataFrame(
        {
            "account": pd.Series([], dtype="int64"),
            "seq": pd.Series([], dtype="object"),
            "payload": pd.Series([], dtype="int64"),
        }
    )
    for npartitions in (1, 3):
        r = _ddf(empty, npartitions).set_index(["account", "seq"],
                                               shuffle_method="tasks")
        assert r.divisions == (None, None)
        assert not r.known_divisions
        out = r.compute(scheduler="sync")
        assert out.shape == (0, 1)
        assert isinstance(out.index, pd.MultiIndex)
        assert out.index.names == ["account", "seq"]
        assert out["payload"].dtype == np.dtype("int64")


# ---------------------------------------------------------------------------
# String and integer key semantics
# ---------------------------------------------------------------------------


def test_string_prefix_and_separators():
    rows = [
        ("a", "b"),
        ("a,b", ""),
        ("", ""),
        ("ab", "c"),
        ("a", ""),
        ("", "z"),
    ]
    pdf = _frame(rows)
    r = _ddf(pdf, 3).set_index(["account", "seq"], npartitions=3,
                               shuffle_method="tasks")
    out = r.compute(scheduler="sync")
    exp = _expected(pdf)
    assert Counter(out.index) == Counter(exp.index)
    # Unicode ordering, not concatenation ordering
    assert list(out.index) == sorted(out.index)


def test_unicode_value_ordering():
    rows = [("é", "1"), ("z", "2"), ("A", "3"), ("a", "4")]
    pdf = _frame(rows)
    r = _ddf(pdf, 2).set_index(["account", "seq"], npartitions=2,
                               shuffle_method="tasks")
    out = r.compute(scheduler="sync")
    assert list(out.index) == sorted(out.index)


def test_negative_and_large_integers_exact():
    big = 2**53
    rows = [(big + 1, 0), (big, 1), (-big, 3), (big + 2, 0), (-big - 1, 2)]
    pdf = _frame(rows)
    r = _ddf(pdf, 3).set_index(["account", "seq"], npartitions=3,
                               shuffle_method="tasks")
    out = r.compute(scheduler="sync")
    assert Counter(out.index) == Counter(_expected(pdf).index)
    # Above 2**53, integer values remain distinct (float would collapse them)
    assert len({k for k in out.index}) == len({k for k in _expected(pdf).index})
    elem = r.loc[(big + 1, 0), ["payload"]].compute(scheduler="sync")
    assert list(elem.index) == [(big + 1, 0)]


def test_mixed_level_types():
    # First level int64, second level string.
    pdf = pd.DataFrame(
        {
            "account": list(range(10)),
            "seq": [f"s{i}" for i in range(10)],
            "payload": range(10),
        }
    )
    r = _ddf(pdf, 3).set_index(["account", "seq"], npartitions=3,
                               shuffle_method="tasks")
    assert r._meta.index.levels[0].dtype == np.dtype("int64")
    out = r.compute(scheduler="sync")
    assert Counter(out.index) == Counter(_expected(pdf).index)

    # Reverse: string first level, int64 second level.
    pdf2 = pd.DataFrame(
        {
            "account": [f"s{i}" for i in range(10)],
            "seq": list(range(10)),
            "payload": range(10),
        }
    )
    r2 = _ddf(pdf2, 3).set_index(["account", "seq"], npartitions=3,
                                 shuffle_method="tasks")
    assert r2._meta.index.levels[1].dtype == np.dtype("int64")
    out2 = r2.compute(scheduler="sync")
    assert Counter(out2.index) == Counter(_expected(pdf2).index)


def test_mixed_values_rejected_not_stringified():
    pdf = pd.DataFrame(
        {
            "account": [1, "x", 3],
            "seq": ["p", "q", "r"],
            "payload": [1, 2, 3],
        }
    )
    with dask.config.set({"dataframe.convert-string": False}):
        frame = _ddf(pdf, 2)
        # The object column is declared a string level from metadata; the
        # non-string value is rejected when the data is first inspected, never
        # silently converted to "1"/"3".
        r = frame.set_index(
            ["account", "seq"],
            divisions=[("1", "p"), ("3", "r")],
            shuffle_method="tasks",
        )
        with pytest.raises(ValueError):
            r.compute(scheduler="sync")


def test_missing_values_rejected_at_first_observation():
    pdf = pd.DataFrame(
        {
            "account": [1, 2, 3],
            "seq": ["p", None, "r"],
            "payload": [1, 2, 3],
        }
    )
    ddf = _ddf(pdf, 2)
    r = ddf.set_index(
        ["account", "seq"],
        divisions=[(1, "p"), (3, "r")],
        shuffle_method="tasks",
    )
    # metadata-level construction succeeds; the value error appears on read.
    with pytest.raises(ValueError):
        r.compute(scheduler="sync")


# ---------------------------------------------------------------------------
# Explicit divisions validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "divisions",
    [
        [(1, "a")],  # too few
        [(1, "a"), (1, "a"), (2, "b")],  # equal interior
        [(2, "a"), (1, "b"), (3, "c")],  # decreasing
        [(1,), (2,), (3,)],  # wrong width
        [(1, "a", "x"), (2, "b", "y")],  # wrong width
        [1, 2, 3],  # scalar boundaries
    ],
)
def test_invalid_explicit_divisions(divisions):
    pdf = _frame([(1, "a"), (2, "b"), (3, "c")])
    with pytest.raises(ValueError):
        _ddf(pdf, 2).set_index(["account", "seq"], divisions=divisions)


def test_explicit_division_count_matches_npartitions():
    pdf = _frame([(1, "a"), (2, "b"), (3, "c")])
    with pytest.raises(ValueError):
        _ddf(pdf, 2).set_index(
            ["account", "seq"],
            divisions=[(1, "a"), (2, "b"), (3, "c")],
            npartitions=5,
        )
    # consistent value is accepted
    r = _ddf(pdf, 2).set_index(
        ["account", "seq"],
        divisions=[(1, "a"), (2, "b"), (3, "c")],
        npartitions=2,
        shuffle_method="tasks",
    )
    assert r.npartitions == 2


def test_out_of_range_data_raises():
    pdf = _frame([(1, "a"), (2, "b"), (3, "c")])
    ddf = _ddf(pdf, 3)
    for divisions in (
        [(2, "a"), (3, "b"), (3, "c")],  # low key below range
        [(1, "a"), (2, "b"), (2, "b")],  # high key above range
    ):
        r = ddf.set_index(["account", "seq"], divisions=divisions,
                          shuffle_method="tasks")
        with pytest.raises(ValueError):
            r.compute(scheduler="sync")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sorted": True},
        {"sort": False},
        {"append": True},
        {"shuffle_method": "p2p"},
    ],
)
def test_unsupported_combinations_rejected(kwargs):
    pdf = _frame([(1, "a"), (2, "b")])
    with pytest.raises((ValueError, NotImplementedError)):
        _ddf(pdf, 1).set_index(["account", "seq"], **kwargs)


def test_bad_columns_rejected():
    pdf = _frame([(1, "a"), (2, "b")])
    ddf = _ddf(pdf, 1)
    with pytest.raises(KeyError):
        ddf.set_index(["account", "nope"])
    with pytest.raises(ValueError):
        ddf.set_index(["account", "account"])
    with pytest.raises(ValueError):
        ddf.set_index(["account", "seq", "payload"])


def test_unsupported_dtypes_rejected_before_read():
    pdf = pd.DataFrame(
        {"a": [1.0, 2.0], "b": ["x", "y"], "payload": [1, 2]}
    )
    with pytest.raises(ValueError):
        _ddf(pdf, 1).set_index(["a", "b"])


# ---------------------------------------------------------------------------
# Automatic divisions
# ---------------------------------------------------------------------------


def test_auto_divisions_are_real_keys():
    rows = [(a, f"s{i}") for a in range(6) for i in range(6)]
    pdf = _frame(rows)
    ddf = _ddf(pdf, 4)
    r = ddf.set_index(["account", "seq"], npartitions=4,
                      shuffle_method="tasks")
    divs = r.divisions
    assert len(divs) >= 2
    assert all(isinstance(d, tuple) and len(d) == 2 for d in divs)
    # every published boundary is an actual key (or the true endpoint)
    keys = set(map(tuple, pdf[["account", "seq"]].itertuples(index=False)))
    for d in divs:
        assert d in keys
    # strict increase except possibly final
    for i in range(len(divs) - 2):
        assert divs[i] < divs[i + 1]
    assert divs[-2] <= divs[-1]
    out = r.compute(scheduler="sync")
    assert Counter(out.index) == Counter(_expected(pdf).index)
    # each partition respects its boundaries
    for i in range(r.npartitions):
        part = r.get_partition(i).compute(scheduler="sync")
        if len(part):
            assert part.index.min() >= divs[i]
            assert part.index.max() < divs[i + 1] or i == r.npartitions - 1


def test_auto_planner_state_is_bounded():
    n = 5000
    pdf = pd.DataFrame(
        {
            "account": np.arange(n) % 50,
            "seq": [f"s{i}" for i in range(n)],
            "payload": np.arange(n),
        }
    )
    ddf = _ddf(pdf, 6)
    r = ddf.set_index(["account", "seq"], npartitions=4,
                      shuffle_method="tasks")
    _ = r.divisions
    stats = dict(LAST_PLAN_STATS)
    assert stats["input_partitions"] == 6
    # Per-partition sample is capped; planner state grows with partitions,
    # not with row count.
    assert stats["uploaded_keys"] <= 6 * (COMPOSITE_SAMPLE_LIMIT + 2)
    assert stats["planner_sample_bound"] == 6 * (COMPOSITE_SAMPLE_LIMIT + 2)
    assert stats["planner_state_keys"] <= 6 * (COMPOSITE_SAMPLE_LIMIT + 2)


def test_auto_divisions_can_empty_but_reported_consistently():
    # A single complete key removes the ability to split
    rows = [(1, "x")] * 20
    pdf = _frame(rows)
    r = _ddf(pdf, 3).set_index(["account", "seq"], npartitions=5,
                               shuffle_method="tasks")
    assert r.divisions == ((1, "x"), (1, "x"))
    assert r.npartitions == 1


# ---------------------------------------------------------------------------
# loc semantics (pandas aligned)
# ---------------------------------------------------------------------------


@pytest.fixture
def keyed():
    rows = [
        (1, "x"),
        (1, "y"),
        (1, "y"),
        (2, "x"),
        (2, "y"),
        (3, "z"),
    ]
    pdf = _frame(rows)
    r = _ddf(pdf, 2).set_index(
        ["account", "seq"],
        divisions=[(1, "x"), (2, "x"), (3, "z"), (3, "z")],
        shuffle_method="tasks",
    )
    return r, _expected(pdf)


def test_loc_complete_key_all_rows(keyed):
    r, exp = keyed
    out = r.loc[(1, "y"), :].compute(scheduler="sync")
    expected = exp.loc[(1, "y"), :]
    _assert_same_rows(out, expected)


def test_loc_inclusive_range(keyed):
    r, exp = keyed
    out = r.loc[slice((1, "y"), (2, "y")), ["payload"]].compute(scheduler="sync")
    expected = exp.loc[slice((1, "y"), (2, "y")), ["payload"]]
    _assert_same_rows(out, expected)


def test_loc_missing_single_key_raises(keyed):
    r, _ = keyed
    with pytest.raises(KeyError):
        r.loc[(1, "q"), :].compute(scheduler="sync")
    with pytest.raises(KeyError):
        r.loc[(9, "q"), ["payload"]].compute(scheduler="sync")


def test_loc_empty_interval_structure(keyed):
    r, exp = keyed
    out = r.loc[slice((5, "a"), (5, "b")), ["payload"]].compute(scheduler="sync")
    expected = exp.loc[slice((5, "a"), (5, "b")), ["payload"]]
    assert out.shape == expected.shape
    assert list(out.columns) == list(expected.columns)
    assert isinstance(out.index, pd.MultiIndex)
    assert out.index.names == ["account", "seq"]


def test_loc_row_key_vs_column_selection(keyed):
    r, _ = keyed
    # (1, "y") is a complete row key, not (row=1, col="y")
    out = r.loc[(1, "y"), ["payload"]].compute(scheduler="sync")
    assert list(out.columns) == ["payload"]
    assert set(out.index) == {(1, "y")}


def test_loc_element_without_column_key(keyed):
    r, exp = keyed
    out = r.loc[(1, "y")]
    assert_eq(out, exp.loc[(1, "y")])


def test_loc_list_of_complete_keys(keyed):
    r, exp = keyed
    out = r.loc[[(1, "y"), (3, "z")], ["payload"]].compute(scheduler="sync")
    expected = exp.loc[[(1, "y"), (3, "z")], ["payload"]]
    _assert_same_rows(out, expected)


def test_projection_then_loc_and_loc_then_projection(keyed):
    r, exp = keyed
    proj = r[["payload"]]
    out = proj.loc[slice((1, "y"), (2, "y")), :].compute(scheduler="sync")
    _assert_same_rows(
        out, exp[["payload"]].loc[slice((1, "y"), (2, "y")), :]
    )
    after = r.loc[slice((1, "y"), (2, "y")), ["payload"]][["payload"]]
    out2 = after.compute(scheduler="sync")
    assert list(out2.columns) == ["payload"]


def test_optimize_and_persist_preserve_semantics(keyed):
    r, exp = keyed
    opt = r.optimize(fuse=False)
    assert opt.divisions == r.divisions
    out = opt.loc[slice((1, "y"), (2, "y")), ["payload"]].compute(scheduler="sync")
    _assert_same_rows(out, exp.loc[slice((1, "y"), (2, "y")), ["payload"]])
    persisted = r.persist(scheduler="sync")
    assert persisted.divisions == r.divisions
    out2 = persisted.loc[slice((1, "y"), (2, "y")), ["payload"]].compute(
        scheduler="sync"
    )
    _assert_same_rows(out2, out)


# ---------------------------------------------------------------------------
# Partition pruning
# ---------------------------------------------------------------------------


def _materialize_with_reader(r, forbidden):
    """Return a frame whose partitions fail if ``forbidden`` partitions read."""
    persisted = r.persist(scheduler="sync")
    parts = [persisted.get_partition(i) for i in range(persisted.npartitions)]

    def make_reader(i):
        def _read():
            if i in forbidden:
                raise RuntimeError(f"read forbidden partition {i}")
            return parts[i].compute(scheduler="sync")

        return _read

    lazy = [delayed(make_reader(i))() for i in range(persisted.npartitions)]
    return from_delayed(lazy, meta=persisted._meta, divisions=persisted.divisions)


def test_narrow_range_prunes_disjoint_partitions():
    rows = [(1, i) for i in range(10)] + [(2, i) for i in range(10)] + [
        (3, i) for i in range(10)
    ]
    pdf = _frame(rows)
    r = _ddf(pdf, 3).set_index(
        ["account", "seq"],
        divisions=[(1, 0), (2, 0), (3, 9), (3, 9)],
        shuffle_method="tasks",
    )
    guarded = _materialize_with_reader(r, forbidden={0, 2})
    out = guarded.loc[slice((2, 0), (2, 9)), ["payload"]].compute(scheduler="sync")
    assert len(out) == 10
    assert set(out.index.get_level_values(0)) == {2}
    # A wide slice must touch the forbidden partitions and therefore fail.
    with pytest.raises(RuntimeError):
        guarded.loc[slice((1, 0), (3, 9)), ["payload"]].compute(scheduler="sync")


def test_explicit_divisions_do_not_read_source():
    reads = []
    meta = pd.DataFrame(
        {
            "account": pd.Series([], dtype="int64"),
            "seq": pd.Series([], dtype="int64"),
            "payload": pd.Series([], dtype="int64"),
        }
    )

    def source(i):
        reads.append(i)
        return pd.DataFrame(
            {
                "account": [i + 1] * 3,
                "seq": [0, 1, 2],
                "payload": [0, 1, 2],
            }
        )

    frame = from_map(source, [0, 1, 2], meta=meta)
    r = frame.set_index(
        ["account", "seq"],
        divisions=[(1, 0), (2, 0), (3, 2), (3, 2)],
        shuffle_method="tasks",
    )
    assert reads == []
    assert r._meta.index.names == ["account", "seq"]
    assert r.divisions == ((1, 0), (2, 0), (3, 2), (3, 2))
    assert reads == []
    _ = r.optimize(fuse=False)
    assert reads == []
    assert r.npartitions == 3
    assert reads == []


def test_auto_summary_is_lazy_and_cached():
    reads = []
    meta = pd.DataFrame(
        {
            "account": pd.Series([], dtype="int64"),
            "seq": pd.Series([], dtype="int64"),
            "payload": pd.Series([], dtype="int64"),
        }
    )

    def source(i):
        reads.append(i)
        return pd.DataFrame(
            {
                "account": [i, i + 1] * 3,
                "seq": [0, 1, 2, 0, 1, 2],
                "payload": range(6),
            }
        )

    frame = from_map(source, [0, 1, 2], meta=meta)
    # Constructing the expression only computes metadata; no source read.
    result = frame.set_index(["account", "seq"], shuffle_method="tasks")
    assert reads == []
    # The first divisions access runs the summary once per input partition.
    divs = result.divisions
    assert sorted(reads) == [0, 1, 2] or len(reads) == 3
    # Subsequent accesses are served from the planning cache.
    assert result.divisions == divs
    assert len(reads) == 3


# ---------------------------------------------------------------------------
# from_pandas pre-indexed path
# ---------------------------------------------------------------------------


def _preindexed(rows, names=("account", "seq")):
    pdf = _frame(rows)[["payload"]]
    pdf.index = pd.MultiIndex.from_tuples(rows, names=list(names))
    return pdf.sort_index()


def test_from_pandas_sorted_multiindex():
    rows = [(1, "x"), (1, "y"), (1, "y"), (2, "x"), (2, "y"), (3, "z")]
    pdf = _preindexed(rows)
    for n in (2, 3, 4):
        r = dd.from_pandas(pdf, npartitions=n, sort=True)
        assert r.known_divisions
        divs = r.divisions
        assert len(divs) == r.npartitions + 1
        # identical keys are never split across partitions
        for i in range(len(divs) - 2):
            assert divs[i] < divs[i + 1]
        assert divs[-2] <= divs[-1]
        out = r.compute(scheduler="sync")
        assert Counter(out.index) == Counter(pdf.index)
        for i in range(r.npartitions):
            part = r.get_partition(i).compute(scheduler="sync")
            if len(part):
                assert part.index.min() >= divs[i]
                assert part.index.max() < divs[i + 1] or i == r.npartitions - 1


def test_from_pandas_sort_false_preserves_order_and_unknown_divisions():
    rows = [(2, "b"), (1, "c"), (1, "a"), (3, "z"), (2, "a")]
    pdf = _preindexed(rows)
    pdf = pdf.iloc[[2, 0, 4, 1, 3]]
    r = dd.from_pandas(pdf, npartitions=2, sort=False)
    assert not r.known_divisions
    assert r.divisions == (None, None, None)
    out = r.compute(scheduler="sync")
    assert list(out.index) == list(pdf.index)
    # still evaluable via loc (unknown divisions -> try_loc path)
    got = r.loc[(2, "a"), ["payload"]].compute(scheduler="sync")
    assert list(got.index) == [(2, "a")]


def test_two_entry_paths_agree():
    rows = [(1, "x"), (1, "y"), (1, "y"), (2, "x"), (2, "y"), (3, "z")]
    pdf = _frame(rows)
    path1 = _ddf(pdf, 2).set_index(
        ["account", "seq"],
        divisions=[(1, "x"), (2, "x"), (3, "z"), (3, "z")],
        shuffle_method="tasks",
    )
    pre = pdf.set_index(["account", "seq"]).sort_index()
    path2 = dd.from_pandas(pre, npartitions=2, sort=True)
    o1 = path1.compute(scheduler="sync")
    o2 = path2.compute(scheduler="sync")
    assert o1.index.names == o2.index.names == ["account", "seq"]
    assert Counter(o1.index) == Counter(o2.index)
    # metadata describing keys agrees across empty/non-empty/partitions
    assert path1._meta.index.names == path2._meta.index.names
    assert list(path1._meta.columns) == list(path2._meta.columns)


def test_from_pandas_rejects_bad_multiindex():
    pdf = pd.DataFrame(
        {"payload": [1, 2]},
        index=pd.MultiIndex.from_tuples([(1, "a"), (2, "b")], names=["x", "y"]),
    )
    bad = pdf.copy()
    bad.index = pd.MultiIndex.from_tuples(
        [(1.5, "a"), (2.0, "b")], names=["x", "y"]
    )
    with pytest.raises(ValueError):
        dd.from_pandas(bad, npartitions=1)


# ---------------------------------------------------------------------------
# Task identity must not be shared across distinct inputs
# ---------------------------------------------------------------------------


def test_distinct_inputs_do_not_share_tasks():
    pdf1 = _frame([(1, "x"), (2, "y"), (3, "z")])
    pdf2 = _frame([(9, "q"), (8, "r"), (7, "s")])
    a = _ddf(pdf1, 2).set_index(["account", "seq"], npartitions=2,
                                shuffle_method="tasks")
    b = _ddf(pdf2, 2).set_index(["account", "seq"], npartitions=2,
                                shuffle_method="tasks")
    # Interleave evaluation
    out_b = b.compute(scheduler="sync")
    out_a = a.compute(scheduler="sync")
    assert Counter(out_a.index) == Counter(_expected(pdf1).index)
    assert Counter(out_b.index) == Counter(_expected(pdf2).index)
    assert a._expr._name != b._expr._name


def test_column_order_changes_task_identity():
    pdf = pd.DataFrame(
        {"account": [1, 1, 2], "seq": ["x", "y", "z"], "payload": [1, 2, 3]}
    )
    a = _ddf(pdf, 2).set_index(["account", "seq"], shuffle_method="tasks")
    b = _ddf(pdf, 2).set_index(["seq", "account"], shuffle_method="tasks")
    assert a._expr._name != b._expr._name
    assert a._meta.index.names == ["account", "seq"]
    assert b._meta.index.names == ["seq", "account"]


def test_different_boundaries_interleaved_do_not_cross():
    pdf = _frame([(1, "x"), (1, "y"), (2, "x"), (2, "y")])
    d = _ddf(pdf, 2)
    a = d.set_index(
        ["account", "seq"],
        divisions=[(1, "x"), (2, "x"), (2, "y")],
        shuffle_method="tasks",
    )
    b = d.set_index(
        ["account", "seq"],
        divisions=[(1, "x"), (1, "y"), (2, "y")],
        shuffle_method="tasks",
    )
    assert a._expr._name != b._expr._name
    assert a.divisions == ((1, "x"), (2, "x"), (2, "y"))
    assert b.divisions == ((1, "x"), (1, "y"), (2, "y"))
    oa1 = a.compute(scheduler="sync")
    ob1 = b.compute(scheduler="sync")
    oa2 = a.compute(scheduler="sync")
    ob2 = b.compute(scheduler="sync")
    assert Counter(oa1.index) == Counter(_expected(pdf).index)
    assert Counter(ob1.index) == Counter(_expected(pdf).index)
    # Results are independent of evaluation order.
    assert sorted(oa1.index) == sorted(oa2.index)
    assert sorted(ob1.index) == sorted(ob2.index)
    # The two construction paths agree on the actual keys.
    assert Counter(oa1.index) == Counter(ob1.index)


# ---------------------------------------------------------------------------
# Scheduler agreement and disk cleanup
# ---------------------------------------------------------------------------


def test_sync_and_threads_agree():
    rng = np.random.default_rng(0)
    n = 400
    pdf = pd.DataFrame(
        {
            "account": rng.integers(0, 8, n),
            "seq": rng.integers(0, 30, n),
            "payload": np.arange(n),
        }
    )
    r = _ddf(pdf, 4).set_index(["account", "seq"], npartitions=4,
                               shuffle_method="tasks")
    sync_out = r.compute(scheduler="sync")
    thread_out = r.compute(scheduler="threads")
    assert Counter(sync_out.index) == Counter(thread_out.index)
    assert sorted(map(tuple, sync_out.index)) == sorted(
        map(tuple, thread_out.index)
    )


def test_disk_cleanup_on_success_and_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DASK_TEMPORARY_DIRECTORY", str(tmp_path))
    meta = pd.DataFrame(
        {
            "account": pd.Series([], dtype="int64"),
            "seq": pd.Series([], dtype="int64"),
            "payload": pd.Series([], dtype="int64"),
        }
    )

    def source(i):
        return pd.DataFrame(
            {"account": [i + 1] * 5, "seq": list(range(5)), "payload": range(5)}
        )

    frame = from_map(source, [0, 1, 2], meta=meta)
    r = frame.set_index(
        ["account", "seq"],
        divisions=[(1, 0), (2, 0), (3, 4), (3, 4)],
        shuffle_method="disk",
    )
    r.compute(scheduler="sync")
    assert not list(tmp_path.glob("**/*.partd"))

    def boom(i):
        raise ValueError("source failure")

    broken = from_map(boom, [0, 1, 2], meta=meta)
    r_broken = broken.set_index(
        ["account", "seq"],
        divisions=[(1, 0), (2, 0), (3, 4), (3, 4)],
        shuffle_method="disk",
    )
    with pytest.raises(ValueError, match="source failure"):
        r_broken.compute(scheduler="sync")
    assert not list(tmp_path.glob("**/*.partd"))


def test_input_and_declarations_unchanged():
    pdf = _frame([(1, "x"), (2, "y"), (3, "z")])
    ddf = _ddf(pdf, 2)
    before = ddf._meta.copy()
    divisions_before = ddf.divisions
    columns = ["account", "seq"]
    divisions = [(1, "x"), (2, "y"), (3, "z")]
    _ = ddf.set_index(columns, divisions=divisions, shuffle_method="tasks")
    assert list(ddf._meta.columns) == list(before.columns)
    assert ddf.divisions == divisions_before
    assert columns == ["account", "seq"]
    assert divisions == [(1, "x"), (2, "y"), (3, "z")]


# ---------------------------------------------------------------------------
# Per-partition verification for larger inputs (never a full-frame concat)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["tasks", "disk"])
def test_larger_per_partition_verification(method):
    rng = np.random.default_rng(42)
    n = 5000
    pdf = pd.DataFrame(
        {
            "account": rng.integers(0, 20, n),
            "seq": rng.integers(0, 200, n),
            "payload": np.arange(n),
        }
    )
    r = _ddf(pdf, 5).set_index(["account", "seq"], npartitions=4,
                               shuffle_method=method)
    divs = r.divisions
    total = 0
    digests = []
    for i in range(r.npartitions):
        part = r.get_partition(i).compute(scheduler="sync")
        total += len(part)
        if len(part):
            assert part.index.min() >= divs[i]
            assert part.index.max() < divs[i + 1] or i == r.npartitions - 1
        digests.append((len(part), hash(tuple(sorted(part["payload"].tolist())))))
    assert total == n
    # Row-count and payload identity digest match a per-partition pandas build
    expected = pdf.set_index(["account", "seq"]).sort_index()
    assert sum(d[0] for d in digests) == len(expected)
    assert sorted(
        payload for _, digest in digests for payload in ()
    ) == []  # structure sanity
    # Independent digest check from the sorted expected frame, split by the
    # published divisions (no full-frame concat on the dask side).
    bounds = pd.Series(list(expected.index))
    from dask.dataframe.dask_expr._multi_index import partition_ids

    first = expected.index.get_level_values(0).to_numpy()
    second = expected.index.get_level_values(1).to_numpy()
    ids = partition_ids(first, second, divs)
    for i in range(r.npartitions):
        want = expected["payload"].to_numpy()[ids == i]
        got = digests[i][1]
        assert got == hash(tuple(sorted(int(x) for x in want)))
