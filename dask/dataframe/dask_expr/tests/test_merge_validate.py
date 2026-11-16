"""Tests for distributed merge cardinality validation (``validate=...``).

The checks are cross-validated against pandas' own ``merge(validate=...)`` on
small hand-built frames, and the planning guarantees (no source reads at graph
time, bounded distributed checks, reuse of upstream uniqueness evidence) are
verified by inspecting the optimized expression / task graph rather than only
the final values.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.errors import MergeError

from dask.dataframe.dask_expr import from_pandas, merge
from dask.dataframe.dask_expr._merge_validate import (
    local_key_counts,
    normalize_validate,
)
from dask.dataframe.dask_expr.tests._util import assert_eq

VALIDATE_ALIASES = [
    ("one_to_one", "1:1"),
    ("1:1", "1:1"),
    ("one_to_many", "1:m"),
    ("1:m", "1:m"),
    ("many_to_one", "m:1", "m:1"),
    ("m:1", "m:1"),
    ("many_to_many", "m:m"),
    ("m:m", "m:m"),
]

ALL_MODES = ["1:1", "1:m", "m:1", "m:m",
             "one_to_one", "one_to_many", "many_to_one", "many_to_many"]


def _frames(left_pdf, right_pdf, npartitions=3):
    return from_pandas(left_pdf, npartitions), from_pandas(right_pdf, npartitions)


def _optimized_repr(df):
    return df.expr.optimize(fuse=False).tree_repr()


def _has_validation_graph(df):
    graph = df.__dask_graph__()
    needles = (
        "localkeycounts",
        "duplicateflag",
        "mergevalidateflag",
        "cardinalitybarrier",
    )
    return any(needle in str(key).lower() for key in graph for needle in needles)


# ---------------------------------------------------------------------------
# Argument validation (graph construction time, no source reads)
# ---------------------------------------------------------------------------


def test_normalize_validate():
    assert normalize_validate(None) is None
    assert normalize_validate("1:1") == "1:1"
    assert normalize_validate("one_to_one") == "1:1"
    assert normalize_validate("one_to_many") == "1:m"
    assert normalize_validate("many_to_one") == "m:1"
    assert normalize_validate("m:m") == "m:m"
    assert normalize_validate("many_to_many") == "m:m"


@pytest.mark.parametrize("bad", ["", "11", "one-to-one", "1:n", 3, object()])
def test_invalid_validate_raises_value_error(bad):
    left, right = _frames(
        pd.DataFrame({"k": [1, 2], "v": [1, 2]}),
        pd.DataFrame({"k": [1, 2], "w": [1, 2]}),
        npartitions=2,
    )
    with pytest.raises(ValueError, match="not a valid argument"):
        left.merge(right, on="k", validate=bad)


def test_validate_leftsemi_rejected():
    left, right = _frames(
        pd.DataFrame({"k": [1, 2], "v": [1, 2]}),
        pd.DataFrame({"k": [1, 2], "w": [1, 2]}),
        npartitions=2,
    )
    with pytest.raises(ValueError, match="leftsemi"):
        left.merge(right, on="k", how="leftsemi", validate="1:1")


def test_construction_and_optimization_do_not_read_source():
    # A source that blows up when actually read must not be touched while
    # building, inspecting metadata, optimizing or writing the graph.
    def boom(part):
        raise RuntimeError("source was read at graph time")

    good = pd.DataFrame({"k": [1, 2, 3, 4], "v": [1, 2, 3, 4]})
    left = from_pandas(good, npartitions=4).map_partitions(
        boom, meta=pd.DataFrame({"k": [1], "v": [1]})
    )
    right = from_pandas(good.rename(columns={"v": "w"}), npartitions=4)
    joined = left.merge(right, on="k", validate="1:1")
    # metadata / optimization / graph materialization must not execute tasks
    _ = joined.columns
    _ = joined.optimize(fuse=False)
    _ = joined.__dask_graph__()
    with pytest.raises(RuntimeError, match="source was read"):
        joined.compute()


# ---------------------------------------------------------------------------
# Cardinality semantics, cross-checked with pandas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
@pytest.mark.parametrize("validate", ["1:1", "one_to_one"])
def test_one_to_one_messages_match_pandas(how, validate):
    cases = {
        "left_dup": (
            pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]}),
            pd.DataFrame({"k": [1, 2, 3], "w": [1, 2, 3]}),
        ),
        "right_dup": (
            pd.DataFrame({"k": [1, 2, 3], "v": [1, 2, 3]}),
            pd.DataFrame({"k": [1, 1, 2], "w": [1, 2, 3]}),
        ),
        "both_dup": (
            pd.DataFrame({"k": [1, 1], "v": [1, 2]}),
            pd.DataFrame({"k": [1, 1], "w": [1, 2]}),
        ),
    }
    for name, (lpdf, rpdf) in cases.items():
        left, right = _frames(lpdf, rpdf, npartitions=3)
        with pytest.raises(MergeError) as dask_err:
            left.merge(right, on="k", how=how, validate=validate).compute()
        with pytest.raises(MergeError) as pd_err:
            lpdf.merge(rpdf, on="k", how=how, validate="1:1")
        assert str(dask_err.value) == str(pd_err.value), name


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_one_to_many_and_many_to_one_messages(how):
    lpdf = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]})
    rpdf = pd.DataFrame({"k": [1, 1, 2], "w": [1, 2, 3]})
    left, right = _frames(lpdf, rpdf, npartitions=3)

    # 1:m forbids left duplicates (left has them)
    with pytest.raises(MergeError) as dask_err:
        left.merge(right, on="k", how=how, validate="1:m").compute()
    with pytest.raises(MergeError) as pd_err:
        lpdf.merge(rpdf, on="k", how=how, validate="1:m")
    assert str(dask_err.value) == str(pd_err.value)

    # m:1 forbids right duplicates (right has them)
    with pytest.raises(MergeError) as dask_err:
        left.merge(right, on="k", how=how, validate="m:1").compute()
    with pytest.raises(MergeError) as pd_err:
        lpdf.merge(rpdf, on="k", how=how, validate="m:1")
    assert str(dask_err.value) == str(pd_err.value)


def test_mm_and_none_add_no_check():
    lpdf = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]})
    rpdf = pd.DataFrame({"k": [1, 1, 2], "w": [1, 2, 3]})
    left, right = _frames(lpdf, rpdf, npartitions=3)
    for validate in (None, "m:m", "many_to_many"):
        joined = left.merge(right, on="k", validate=validate)
        assert not _has_validation_graph(joined)
        assert_eq(joined, lpdf.merge(rpdf, on="k"), check_index=False)


def test_duplicate_key_without_match_still_violates():
    # Duplicated key never participates in the join output.
    lpdf = pd.DataFrame({"k": [9, 9], "v": [1, 2]})
    rpdf = pd.DataFrame({"k": [1], "w": [1]})
    left, right = _frames(lpdf, rpdf, npartitions=2)
    with pytest.raises(MergeError):
        left.merge(right, on="k", how="inner", validate="1:1").compute()
    with pytest.raises(MergeError):
        left.merge(right, on="k", how="left", validate="1:m").compute()


def test_empty_result_is_still_checked():
    lpdf = pd.DataFrame({"k": [5, 5], "v": [1, 2]})
    rpdf = pd.DataFrame({"k": [6, 6], "w": [1, 2]})
    left, right = _frames(lpdf, rpdf, npartitions=2)
    with pytest.raises(MergeError):
        left.merge(right, on="k", how="inner", validate="1:1").compute()


def test_empty_frames_pass():
    empty = pd.DataFrame({"k": pd.Series([], dtype="int64"),
                          "v": pd.Series([], dtype="int64")})
    left, right = _frames(empty, empty.rename(columns={"v": "w"}), npartitions=2)
    result = left.merge(right, on="k", validate="1:1").compute()
    assert len(result) == 0


@pytest.mark.parametrize(
    "key_factory",
    [
        lambda: ("k", pd.DataFrame({"k": [1.0, 1.0], "v": [1, 2]})),
        lambda: ("k", pd.DataFrame(
            {"k": pd.array([1, 1], dtype="Int64"), "v": [1, 2]})),
        lambda: ("k", pd.DataFrame(
            {"k": pd.Categorical(["a", "a"]), "v": [1, 2]})),
        lambda: (["a", "b"], pd.DataFrame(
            {"a": [1, 1], "b": [1, 1], "v": [1, 2]})),
    ],
)
def test_key_types_detect_duplicates(key_factory):
    key, lpdf = key_factory()
    if isinstance(key, list):
        rpdf = pd.DataFrame({"a": [1], "b": [1], "w": [1]})
    else:
        rpdf = pd.DataFrame({key: lpdf[key].iloc[:1], "w": [1]})
    left, right = _frames(lpdf, rpdf, npartitions=2)
    with pytest.raises(MergeError):
        left.merge(right, on=key, validate="1:1").compute()
    # unique counterpart passes and matches pandas
    if isinstance(key, list):
        l_unique = lpdf.drop_duplicates(subset=key)
    else:
        l_unique = lpdf.drop_duplicates(subset=[key])
    lu, ru = _frames(l_unique, rpdf, npartitions=2)
    got = lu.merge(ru, on=key, validate="1:1").compute()
    expect = l_unique.merge(rpdf, on=key, validate="1:1")
    assert_eq(got, expect, check_index=False)


def test_nan_keys_treated_as_equal():
    lpdf = pd.DataFrame({"k": [np.nan, np.nan], "v": [1, 2]})
    rpdf = pd.DataFrame({"k": [1.0], "w": [1]})
    left, right = _frames(lpdf, rpdf, npartitions=2)
    with pytest.raises(MergeError):
        left.merge(right, on="k", validate="1:1").compute()


def test_composite_key_subset_not_inferred():
    # (a, b) unique does not imply a is unique.
    pdf = pd.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 1, 2], "v": [1, 2, 3, 4]})
    right = pd.DataFrame({"a": [1, 2], "b": [1, 2], "w": [1, 2]})
    left = from_pandas(pdf, npartitions=4).drop_duplicates(subset=["a", "b"])
    rdd = from_pandas(right, npartitions=2)
    # full composite key is fine
    assert_eq(
        left.merge(rdd, on=["a", "b"], validate="1:1"),
        pdf.drop_duplicates(subset=["a", "b"]).merge(right, on=["a", "b"]),
        check_index=False,
    )
    # subset 'a' is not unique -> violation
    with pytest.raises(MergeError):
        left.merge(rdd, on="a", validate="1:1").compute()


def test_differently_named_left_right_keys():
    lpdf = pd.DataFrame({"lk": [1, 1, 2], "v": [1, 2, 3]})
    rpdf = pd.DataFrame({"rk": [1, 2, 3], "w": [1, 2, 3]})
    left, right = _frames(lpdf, rpdf, npartitions=3)
    # m:1: right must be unique (it is) -> passes even with left dups
    assert_eq(
        left.merge(right, left_on="lk", right_on="rk", validate="m:1"),
        lpdf.merge(rpdf, left_on="lk", right_on="rk", validate="m:1"),
        check_index=False,
    )
    rbad = pd.DataFrame({"rk": [1, 1, 2], "w": [1, 2, 3]})
    rb = from_pandas(rbad, npartitions=3)
    with pytest.raises(MergeError):
        left.merge(rb, left_on="lk", right_on="rk", validate="m:1").compute()


def test_index_keys_named_and_unnamed():
    lpdf = pd.DataFrame({"v": [1, 1, 2]}, index=pd.Index([1, 1, 2], name="k"))
    rpdf = pd.DataFrame({"w": [1, 2, 3]}, index=pd.Index([1, 2, 3], name="k"))
    left, right = _frames(lpdf, rpdf, npartitions=3)
    with pytest.raises(MergeError):
        left.merge(
            right, left_index=True, right_index=True, validate="1:1"
        ).compute()
    # unnamed single-level index
    lu = pd.DataFrame({"v": [1, 2, 3]})
    ru = pd.DataFrame({"w": [1, 2, 3]})
    ldd, rdd = _frames(lu, ru, npartitions=3)
    assert_eq(
        ldd.merge(rdd, left_index=True, right_index=True, validate="1:1"),
        lu.merge(ru, left_index=True, right_index=True, validate="1:1"),
        check_index=False,
    )


def test_index_to_column_key():
    lpdf = pd.DataFrame({"v": [1, 1, 2]}, index=pd.Index([1, 1, 2], name="k"))
    rpdf = pd.DataFrame({"rk": [1, 2, 3], "w": [1, 2, 3]})
    left, right = _frames(lpdf, rpdf, npartitions=3)
    with pytest.raises(MergeError):
        left.merge(
            right, left_index=True, right_on="rk", validate="1:m"
        ).compute()


# ---------------------------------------------------------------------------
# Barrier / optimization guarantees
# ---------------------------------------------------------------------------


def test_post_merge_filter_cannot_erase_check():
    rpdf = pd.DataFrame({"k": [1, 2, 3, 99, 99], "w": [10, 20, 30, 40, 50]})
    lpdf = pd.DataFrame({"k": [1, 2, 3], "v": [1, 2, 3]})
    left, right = _frames(lpdf, rpdf, npartitions=5)
    # Filter above the merge removes the (unmatched) bad rows from the output,
    # but the check is over the full inputs.
    with pytest.raises(MergeError):
        (left.merge(right, on="k", validate="1:1")[lambda d: d.k < 50]).compute()


def test_pre_merge_filter_checks_only_filtered_input():
    rpdf = pd.DataFrame({"k": [1, 2, 3, 99, 99], "w": [10, 20, 30, 40, 50]})
    lpdf = pd.DataFrame({"k": [1, 2, 3], "v": [1, 2, 3]})
    left, right = _frames(lpdf, rpdf, npartitions=5)
    filtered = right[right.k < 50]  # now globally unique on k
    got = left.merge(filtered, on="k", validate="1:1")
    assert_eq(
        got,
        lpdf.merge(rpdf[rpdf.k < 50], on="k"),
        check_index=False,
    )


def test_single_output_partition_still_checked():
    lpdf = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]})
    rpdf = pd.DataFrame({"k": [1, 2, 3], "w": [1, 2, 3]})
    left, right = _frames(lpdf, rpdf, npartitions=3)
    joined = left.merge(right, on="k", validate="1:1")
    with pytest.raises(MergeError):
        joined.partitions[0].compute()


def test_projection_dropping_key_keeps_check():
    lpdf = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]})
    rpdf = pd.DataFrame({"k": [1, 2, 3], "w": [1, 2, 3]})
    left, right = _frames(lpdf, rpdf, npartitions=3)
    with pytest.raises(MergeError):
        left.merge(right, on="k", validate="1:1")[["v", "w"]].compute()


def test_barrier_survives_optimization():
    left, right = _frames(
        pd.DataFrame({"k": [1, 2, 3, 4], "v": [1, 2, 3, 4]}),
        pd.DataFrame({"k": [1, 2, 3, 4, 4], "w": [1, 2, 3, 4, 5]}),
        npartitions=4,
    )
    joined = left.merge(right, on="k", validate="1:1")
    tree = _optimized_repr(joined)
    assert "CardinalityBarrier" in tree
    assert "LocalKeyCounts" in tree


def test_valid_result_identical_to_unvalidated_merge():
    lpdf = pd.DataFrame({"k": range(20), "v": range(20)})
    rpdf = pd.DataFrame({"k": range(0, 20, 2), "w": range(10)})
    for how in ["inner", "left", "right", "outer"]:
        left, right = _frames(lpdf, rpdf, npartitions=4)
        validated = left.merge(right, on="k", how=how, validate="1:1")
        plain = left.merge(right, on="k", how=how)
        assert_eq(validated, plain, check_index=False)
        assert_eq(validated, lpdf.merge(rpdf, on="k", how=how), check_index=False)


# ---------------------------------------------------------------------------
# Uniqueness evidence reuse
# ---------------------------------------------------------------------------


def test_upstream_dedup_avoids_extra_scan():
    lpdf = pd.DataFrame({"k": [1, 2, 3, 4], "v": [1, 2, 3, 4]})
    rpdf = pd.DataFrame({"k": [1, 2, 3, 4, 4, 2], "w": [10, 20, 30, 40, 50, 60]})
    left, right = _frames(lpdf, rpdf, npartitions=4)
    right_dedup = right.drop_duplicates(subset=["k"])

    # Right proven unique -> only the left side needs a check (1 set of
    # LocalKeyCounts), barrier remains.
    joined = left.merge(right_dedup, on="k", validate="1:1")
    tree = _optimized_repr(joined)
    assert tree.count("LocalKeyCounts") == 1

    # Both sides proven -> no validation machinery at all.
    left_dedup = left.drop_duplicates(subset=["k"])
    fully = left_dedup.merge(right_dedup, on="k", validate="1:1")
    assert not _has_validation_graph(fully)
    assert_eq(
        fully,
        lpdf.merge(rpdf.drop_duplicates(subset=["k"]), on="k"),
        check_index=False,
    )


def test_redundant_drop_duplicates_is_identity_plan():
    pdf = pd.DataFrame({"k": [1, 2, 3, 4], "v": [1, 2, 3, 4]})
    ddf = from_pandas(pdf, npartitions=4)
    once = ddf.drop_duplicates(subset=["k"])
    twice = once.drop_duplicates(subset=["k"])
    # No extra shuffle reduction / tasks compared to a single dedup.
    assert len(twice.__dask_graph__()) == len(once.__dask_graph__())
    assert_eq(twice, pdf, check_index=False)


def test_evidence_invalidation_cases():
    base = from_pandas(
        pd.DataFrame({"k": [1, 2, 3, 4], "v": [10, 20, 30, 40]}), npartitions=4
    ).drop_duplicates(subset=["k"])
    right = from_pandas(
        pd.DataFrame({"k": [1, 2, 3, 4], "w": [1, 2, 3, 4]}), npartitions=4
    )

    # Overwriting the key column invalidates evidence.
    overwritten = base.assign(k=base.v)
    assert _has_validation_graph(
        overwritten.merge(right, on="k", validate="1:1")
    )

    # concat of individually-unique tables invalidates evidence.
    other = from_pandas(
        pd.DataFrame({"k": [5, 6], "v": [50, 60]}), npartitions=2
    ).drop_duplicates(subset=["k"])
    concatenated = __import__("dask.dataframe", fromlist=["concat"]).concat(
        [base, other]
    )
    assert _has_validation_graph(
        concatenated.merge(right, on="k", validate="1:1")
    )

    # Arbitrary map_partitions invalidates evidence.
    mapped = base.map_partitions(lambda df: df, meta=base._meta)
    assert _has_validation_graph(
        mapped.merge(right, on="k", validate="1:1")
    )


def test_evidence_preserved_through_rename_filter_repartition_projection():
    left = from_pandas(
        pd.DataFrame({"k": [1, 2, 3, 4], "v": [10, 20, 30, 40]}), npartitions=4
    ).drop_duplicates(subset=["k"])
    right = from_pandas(
        pd.DataFrame({"k": [1, 2, 3, 4], "w": [1, 2, 3, 4]}), npartitions=4
    ).drop_duplicates(subset=["k"])

    renamed_l = left.rename(columns={"k": "k2"})
    renamed_r = right.rename(columns={"k": "k2"})
    assert not _has_validation_graph(
        renamed_l.merge(renamed_r, on="k2", validate="1:1")
    )
    assert not _has_validation_graph(
        left[left.v > 11].merge(right, on="k", validate="1:1")
    )
    assert not _has_validation_graph(
        left.repartition(npartitions=2).merge(right, on="k", validate="1:1")
    )
    assert not _has_validation_graph(
        left[["k", "v"]].merge(right, on="k", validate="1:1")
    )


def test_known_partitioning_is_not_uniqueness_evidence():
    ddf = from_pandas(
        pd.DataFrame({"k": [1, 1, 2, 3], "v": [1, 2, 3, 4]}), npartitions=2
    ).set_index("k")
    assert ddf.expr.unique_key_sets == frozenset()
    right = from_pandas(
        pd.DataFrame({"k": [1, 2, 3, 4], "w": [1, 2, 3, 4]}), npartitions=4
    )
    joined = ddf.merge(
        right, left_index=True, right_on="k", validate="1:m"
    )
    assert _has_validation_graph(joined)
    with pytest.raises(MergeError):
        joined.compute()


def test_mode_isolation_and_source_failure():
    lpdf = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]})
    rpdf = pd.DataFrame({"k": [1, 2, 3], "w": [1, 2, 3]})
    left, right = _frames(lpdf, rpdf, npartitions=3)

    # A successful lenient run must not make a strict run skip its check.
    assert len(left.merge(right, on="k", validate="m:m").compute()) == 3
    with pytest.raises(MergeError):
        left.merge(right, on="k", validate="1:m").compute()

    # Source failures must surface, not be treated as validation success.
    def boom(part):
        raise RuntimeError("source exploded")

    broken = from_pandas(
        pd.DataFrame({"k": [1, 2], "v": [1, 2]}), npartitions=2
    ).map_partitions(boom, meta=pd.DataFrame({"k": [1], "v": [1]}))
    with pytest.raises(RuntimeError, match="source exploded"):
        broken.merge(right, on="k", validate="1:m").compute()

    # Subsequent valid computation is unaffected.
    good_l, good_r = _frames(
        pd.DataFrame({"k": [1, 2], "v": [1, 2]}),
        pd.DataFrame({"k": [1, 2], "w": [1, 2]}),
        npartitions=2,
    )
    assert_eq(
        good_l.merge(good_r, on="k", validate="1:1"),
        pd.DataFrame({"k": [1, 2], "v": [1, 2], "w": [1, 2]}),
        check_index=False,
    )


# ---------------------------------------------------------------------------
# Bounded distributed state
# ---------------------------------------------------------------------------


def test_check_state_is_bounded():
    npartitions = 8
    rows = npartitions * 1000
    pdf = pd.DataFrame(
        {"k": np.where(np.arange(rows) % 10 == 0, 0, np.arange(rows)),
         "v": np.arange(rows)}
    )
    left = from_pandas(pdf, npartitions=npartitions)
    distinct = sorted(set(pdf.k.tolist()))
    right = from_pandas(
        pd.DataFrame({"k": distinct, "w": range(len(distinct))}),
        npartitions=npartitions,
    )
    joined = left.merge(right, on="k", validate="m:1")
    graph = joined.__dask_graph__()

    # One local summary and one bucket flag per side per partition; a tree
    # reduction; one lightweight barrier per output partition. Nothing holds
    # all distinct keys of a side in a single new task.
    local = [k for k in graph if "localkeycounts" in str(k).lower()]
    flags = [k for k in graph if "duplicateflag" in str(k).lower()]
    barriers = [k for k in graph if "cardinalitybarrier" in str(k).lower()]
    assert len(local) == npartitions
    assert len(flags) == npartitions
    assert len(barriers) == joined.npartitions

    # The local summary of a partition contains only its distinct keys,
    # never its raw rows.
    summary = local_key_counts(pdf.iloc[:1000], ["k"])
    assert len(summary) <= 1000
    assert set(summary.columns) == {"k", "_n"}
    # Hot key multiplicity is preserved by the local counts.
    assert summary.loc[summary.k == 0, "_n"].iloc[0] == 100


@pytest.mark.parametrize("shuffle_method", ["tasks", "disk"])
def test_runs_under_single_threaded_schedulers(shuffle_method):
    left, right = _frames(
        pd.DataFrame({"k": [1, 2, 3], "v": [1, 2, 3]}),
        pd.DataFrame({"k": [1, 2, 3], "w": [1, 2, 3]}),
        npartitions=3,
    )
    import dask

    with dask.config.set({"dataframe.shuffle.method": shuffle_method}):
        got = left.merge(right, on="k", validate="1:1",
                         shuffle_method=shuffle_method).compute(scheduler="sync")
    assert len(got) == 3


def test_validated_merge_provides_dedup_guarantee():
    lpdf = pd.DataFrame({"k": [1, 2, 3, 4], "v": [1, 2, 3, 4]})
    rpdf = pd.DataFrame({"k": [1, 2, 3, 4], "w": [10, 20, 30, 40]})
    left, right = _frames(lpdf, rpdf, npartitions=4)

    for how in ["inner", "left", "outer"]:
        merged = left.merge(right, on="k", how=how, validate="1:1")
        assert frozenset({"k"}) in merged.expr.unique_key_sets
        # A following same-key dedup must not add another global reduction.
        deduped = merged.drop_duplicates(subset=["k"])
        assert len(deduped.__dask_graph__()) == len(merged.__dask_graph__())

    # right-only join does not establish the output guarantee
    right_only = left.merge(right, on="k", how="right", validate="1:1")
    assert right_only.expr.unique_key_sets == frozenset()

    # a key dtype conversion suppresses the guarantee
    r_typed = from_pandas(
        pd.DataFrame({"k": pd.array([1, 2, 3, 4], dtype="Int64"),
                      "w": [10, 20, 30, 40]}),
        npartitions=4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        converted = left.merge(r_typed, on="k", how="inner", validate="1:1")
    assert converted.expr.unique_key_sets == frozenset()


def test_public_api_accepts_validate():
    left, right = _frames(
        pd.DataFrame({"k": [1, 2], "v": [1, 2]}),
        pd.DataFrame({"k": [1, 2], "w": [1, 2]}),
        npartitions=2,
    )
    # Both the method and the functional API expose validate.
    assert_eq(
        left.merge(right, on="k", validate="one_to_one"),
        left.compute().merge(right.compute(), on="k"),
        check_index=False,
    )
    assert_eq(
        merge(left, right, on="k", validate="1:1"),
        left.compute().merge(right.compute(), on="k"),
        check_index=False,
    )
