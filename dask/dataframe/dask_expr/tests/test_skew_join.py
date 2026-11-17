"""Tests for the skew-aware hash join (``skew_join=``).

Coverage groups:

* hand-built small tables with row identifiers and the expected pairing list
  compared against pandas for every join direction,
* two-sided / single-sided / absent hot keys, duplicate identical rows,
  multiple hot keys, empty frames, composite keys with missing components,
  differently named left/right columns, categorical and nullable keys,
  ``suffixes``, ``indicator`` and internal-column-name collisions,
* configuration validation (errors raised while building the graph, without
  running source tasks) and unsupported-combination rejection,
* laziness, projection/filter pushdown, optimization, partition culling,
  ``persist``, joint evaluation and graph-key isolation,
* the per-task input bound verified through the instrumentation probe,
* downstream groupby / drop_duplicates / second merge correctness,
* tasks and disk shuffle methods, plus the synchronous and threaded
  schedulers and a local distributed ``Client``.
"""

from __future__ import annotations

import io
import contextlib
import json
import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

import dask
from dask.dataframe.dask_expr import from_pandas, merge
from dask.dataframe.dask_expr._skew_join import (
    _INSTRUMENT_ENV,
    SkewJoin,
    validate_skew_join,
)
from dask.dataframe.dask_expr.tests._util import _backend_library, assert_eq

pd = _backend_library()

CFG = {"hot_keys": [(1,)], "split_factor": 2}
ALL_HOW = ["inner", "left", "right", "outer"]
ALL_METHODS = ["tasks", "disk"]


def _ddf(pdf, npartitions=3):
    return from_pandas(pdf, npartitions)


def _plan_text(collection):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        collection.optimize().pprint()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Hand-built pairing table with explicit row identifiers
# ---------------------------------------------------------------------------


def test_handbuilt_expected_pairings():
    # Left hot key 1: rows L1, L2 (identical key, distinct ids).
    # Right hot key 1: rows R1, R2.
    # Normal key 2 appears on both sides; key 3 is left-only, key 4 right-only.
    left = pd.DataFrame(
        {
            "k": [1, 1, 2, 3],
            "lid": ["L1", "L2", "L3", "L4"],
            "v": [10, 20, 30, 40],
        }
    )
    right = pd.DataFrame(
        {"k": [1, 1, 2, 4], "rid": ["R1", "R2", "R3", "R4"], "w": [1, 2, 3, 4]}
    )
    dfl, dfr = _ddf(left, 3), _ddf(right, 2)

    expected_inner = {
        ("L1", "R1"),
        ("L1", "R2"),
        ("L2", "R1"),
        ("L2", "R2"),
        ("L3", "R3"),
    }
    for method in ALL_METHODS:
        got = dfl.merge(dfr, on="k", how="inner", skew_join=CFG,
                        shuffle_method=method).compute()
        pairs = set(zip(got["lid"], got["rid"]))
        assert pairs == expected_inner
        # Every qualified pair exactly once (no duplication, no omission).
        assert len(got) == len(expected_inner)

        # Outer result must preserve every original row exactly once.
        outer = dfl.merge(
            dfr, on="k", how="outer", skew_join=CFG, shuffle_method=method
        ).compute()
        assert sorted(outer["lid"].dropna()) == ["L1", "L1", "L2", "L2", "L3", "L4"]
        assert sorted(outer["rid"].dropna()) == ["R1", "R1", "R2", "R2", "R3", "R4"]


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_handbuilt_vs_pandas_all_directions(how, method):
    left = pd.DataFrame({"k": [1, 1, 2, 3], "v": list("abcd")})
    right = pd.DataFrame({"k": [1, 1, 2, 4], "w": list("wxyz")})
    got = _ddf(left, 3).merge(
        _ddf(right, 2), on="k", how=how, skew_join=CFG, shuffle_method=method
    )
    assert_eq(got, left.merge(right, on="k", how=how), check_index=False)


# ---------------------------------------------------------------------------
# Hot key presence scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_hot_key_only_left(how, method):
    left = pd.DataFrame({"k": [1, 1, 1, 2], "v": list("abcd")})
    right = pd.DataFrame({"k": [2, 3], "w": ["x", "y"]})
    got = _ddf(left, 3).merge(
        _ddf(right, 2), on="k", how=how, skew_join=CFG, shuffle_method=method
    )
    assert_eq(got, left.merge(right, on="k", how=how), check_index=False)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_hot_key_only_right(how, method):
    left = pd.DataFrame({"k": [2, 3], "v": ["a", "b"]})
    right = pd.DataFrame({"k": [1, 1, 1, 2], "w": list("wxyz")})
    got = _ddf(left, 3).merge(
        _ddf(right, 2), on="k", how=how, skew_join=CFG, shuffle_method=method
    )
    assert_eq(got, left.merge(right, on="k", how=how), check_index=False)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_hot_key_absent(how, method):
    # A listed hot key that exists nowhere must only affect cost, not result.
    left = pd.DataFrame({"k": [2, 3, 3], "v": list("abc")})
    right = pd.DataFrame({"k": [2, 4], "w": ["x", "y"]})
    cfg = {"hot_keys": [(99,)], "split_factor": 2}
    got = _ddf(left, 3).merge(
        _ddf(right, 2), on="k", how=how, skew_join=cfg, shuffle_method=method
    )
    assert_eq(got, left.merge(right, on="k", how=how), check_index=False)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_duplicate_identical_rows(how, method):
    # Fully identical input rows are still distinct records.
    left = pd.DataFrame({"k": [1, 1, 1, 1], "v": ["a", "a", "b", "b"]})
    right = pd.DataFrame({"k": [1, 1], "w": ["x", "x"]})
    got = _ddf(left, 2).merge(
        _ddf(right, 2), on="k", how=how, skew_join=CFG, shuffle_method=method
    )
    expected = left.merge(right, on="k", how=how)
    assert_eq(got, expected, check_index=False)
    # Inner join yields the full 4 x 2 cartesian product of the hot key.
    if how == "inner":
        assert len(got.compute()) == 8


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_multiple_hot_keys(how, method):
    left = pd.DataFrame({"k": [1, 1, 2, 2, 3], "v": list("abcde")})
    right = pd.DataFrame({"k": [1, 2, 2], "w": ["x", "y", "z"]})
    cfg = {"hot_keys": [(1,), (2,)], "split_factor": 2}
    got = _ddf(left, 3).merge(
        _ddf(right, 2), on="k", how=how, skew_join=cfg, shuffle_method=method
    )
    assert_eq(got, left.merge(right, on="k", how=how), check_index=False)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_empty_frames(how, method):
    empty = pd.DataFrame(
        {"k": pd.Series(dtype="int64"), "v": pd.Series(dtype="object")}
    )
    nonempty = pd.DataFrame({"k": [1, 1], "w": ["a", "b"]})
    for lhs, rhs in ((empty, nonempty), (nonempty, empty), (empty, empty)):
        got = _ddf(lhs, 2).merge(
            _ddf(rhs, 2), on="k", how=how, skew_join=CFG, shuffle_method=method
        )
        assert_eq(got, lhs.merge(rhs, on="k", how=how), check_index=False)


# ---------------------------------------------------------------------------
# Key types: composite, missing components, heterogeneously named columns,
# categorical, nullable / numpy integers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_composite_key_with_none_int(how, method):
    left = pd.DataFrame(
        {
            "k": [1, 1, 1, 2],
            "z": pd.Series([None, None, 1, None], dtype="Int64"),
            "v": list("abcd"),
        }
    )
    right = pd.DataFrame(
        {
            "k": [1, 2],
            "z": pd.Series([None, None], dtype="Int64"),
            "w": ["x", "y"],
        }
    )
    cfg = {"hot_keys": [(1, None)], "split_factor": 2}
    got = _ddf(left, 3).merge(
        _ddf(right, 2), on=["k", "z"], how=how, skew_join=cfg,
        shuffle_method=method,
    )
    assert_eq(got, left.merge(right, on=["k", "z"], how=how), check_index=False)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_composite_key_with_none_string(how, method):
    left = pd.DataFrame(
        {"k": [1, 1, 1, 2], "z": ["a", "a", "b", None], "v": list("abcd")}
    )
    right = pd.DataFrame(
        {"k": [1, 1], "z": ["a", None], "w": ["x", "y"]}
    )
    cfg = {"hot_keys": [(1, None)], "split_factor": 2}
    got = _ddf(left, 3).merge(
        _ddf(right, 2), on=["k", "z"], how=how, skew_join=cfg,
        shuffle_method=method,
    )
    assert_eq(got, left.merge(right, on=["k", "z"], how=how), check_index=False)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_differently_named_columns(how, method):
    left = pd.DataFrame({"lk": [1, 1, 1, 2], "v": list("abcd")})
    right = pd.DataFrame({"rk": [1, 2], "w": ["x", "y"]})
    got = _ddf(left, 3).merge(
        _ddf(right, 2),
        left_on="lk",
        right_on="rk",
        how=how,
        skew_join=CFG,
        shuffle_method=method,
    )
    expected = left.merge(right, left_on="lk", right_on="rk", how=how)
    assert_eq(got, expected, check_index=False)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_categorical_same_domain(how, method):
    left = pd.DataFrame(
        {"k": pd.Categorical(["x", "x", "x", "y"]), "v": [1, 2, 3, 4]}
    )
    right = pd.DataFrame(
        {"k": pd.Categorical(["x", "y", None]), "w": [10, 20, 30]}
    )
    cfg = {"hot_keys": [("x",)], "split_factor": 2}
    got = _ddf(left, 3).merge(
        _ddf(right, 2), on="k", how=how, skew_join=cfg, shuffle_method=method
    )
    expected = left.merge(right, on="k", how=how)
    assert_eq(got, expected, check_index=False)
    # Categorical dtype of the join column must survive.
    assert str(got.compute()["k"].dtype) == str(expected["k"].dtype)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_categorical_different_domains(how, method):
    # Different category domains force pandas to upcast the join key; the
    # skew output partitions must still concatenate without dtype conflicts.
    left = pd.DataFrame(
        {
            "k": pd.Categorical(["x", "x", "y"], categories=["x", "y", "z"]),
            "v": [1, 2, 3],
        }
    )
    right = pd.DataFrame({"k": pd.Categorical(["x", "z"]), "w": [10, 20]})
    cfg = {"hot_keys": [("x",)], "split_factor": 2}
    got = _ddf(left, 2).merge(
        _ddf(right, 2), on="k", how=how, skew_join=cfg, shuffle_method=method
    )
    assert_eq(got, left.merge(right, on="k", how=how), check_index=False)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_nullable_string_key(how, method):
    left = pd.DataFrame(
        {"k": pd.array(["a", "a", None], dtype="string"), "v": [1, 2, 3]}
    )
    right = pd.DataFrame(
        {"k": pd.array(["a", None], dtype="string"), "w": [10, 20]}
    )
    cfg = {"hot_keys": [("a",)], "split_factor": 2}
    got = _ddf(left, 2).merge(
        _ddf(right, 2), on="k", how=how, skew_join=cfg, shuffle_method=method
    )
    assert_eq(got, left.merge(right, on="k", how=how), check_index=False)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_nullable_integer_key(how, method):
    left = pd.DataFrame(
        {"k": pd.array([1, 1, 1, pd.NA, 2], dtype="Int64"), "v": list("abcde")}
    )
    right = pd.DataFrame(
        {"k": pd.array([1, pd.NA], dtype="Int64"), "w": [10, 20]}
    )
    cfg = {"hot_keys": [(1,)], "split_factor": 3}
    got = _ddf(left, 3).merge(
        _ddf(right, 2), on="k", how=how, skew_join=cfg, shuffle_method=method
    )
    assert_eq(got, left.merge(right, on="k", how=how), check_index=False)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_numpy_integer_key_and_config(how, method):
    left = pd.DataFrame(
        {"k": np.array([1, 1, 2], dtype=np.int32), "v": [1, 2, 3]}
    )
    right = pd.DataFrame(
        {"k": np.array([1, 2], dtype=np.int32), "w": [1, 2]}
    )
    cfg = {"hot_keys": [(np.int32(1),)], "split_factor": 2}
    got = _ddf(left, 3).merge(
        _ddf(right, 2), on="k", how=how, skew_join=cfg, shuffle_method=method
    )
    assert_eq(got, left.merge(right, on="k", how=how), check_index=False)


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_string_hot_key(how, method):
    left = pd.DataFrame({"k": ["hot", "hot", "cold"], "v": [1, 2, 3]})
    right = pd.DataFrame({"k": ["hot", "cold"], "w": [10, 20]})
    cfg = {"hot_keys": [("hot",)], "split_factor": 2}
    got = _ddf(left, 3).merge(
        _ddf(right, 2), on="k", how=how, skew_join=cfg, shuffle_method=method
    )
    assert_eq(got, left.merge(right, on="k", how=how), check_index=False)


# ---------------------------------------------------------------------------
# suffixes, indicator, temp column collision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ALL_HOW)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_suffixes_indicator_and_column_collision(how, method):
    left = pd.DataFrame(
        {"a": [1, 1], "_skew_piece": [9, 9], "v": [1, 2]}
    )
    right = pd.DataFrame({"a": [1], "_skew_piece": [8], "w": [3]})
    got = _ddf(left, 2).merge(
        _ddf(right, 1),
        on="a",
        how=how,
        skew_join=CFG,
        shuffle_method=method,
        suffixes=("_L", "_R"),
        indicator=True,
    )
    expected = left.merge(
        right, on="a", how=how, suffixes=("_L", "_R"), indicator=True
    )
    assert_eq(got, expected, check_index=False)
    # User column is retained (suffixed); nothing is overwritten or dropped.
    assert "_skew_piece_L" in got.columns
    assert "_skew_piece_R" in got.columns
    if how in ("left", "outer"):
        ind = got.compute()["_merge"]
        assert set(ind.astype(str)).issubset({"left_only", "right_only", "both"})


@pytest.mark.parametrize("method", ALL_METHODS)
def test_custom_indicator_name(method):
    left = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]})
    right = pd.DataFrame({"k": [1, 2, 2], "w": [1, 2, 3]})
    got = _ddf(left, 3).merge(
        _ddf(right, 2),
        on="k",
        how="outer",
        skew_join=CFG,
        shuffle_method=method,
        indicator="flag",
    )
    expected = left.merge(right, on="k", how="outer", indicator="flag")
    assert_eq(got, expected, check_index=False)
    assert "flag" in got.columns


# ---------------------------------------------------------------------------
# Configuration validation and unsupported combinations
# ---------------------------------------------------------------------------


def test_validate_skew_join_normalization():
    assert validate_skew_join(None, 1) is None
    norm = validate_skew_join(
        {"hot_keys": [(np.int64(1),), (1,)], "split_factor": np.int16(4)}, 1
    )
    assert norm == {"hot_keys": ((1,),), "split_factor": 4}
    assert isinstance(norm["split_factor"], int)


@pytest.mark.parametrize(
    "bad",
    [
        [("x",)],
        {"hot_keys": [(1,)]},
        {"split_factor": 2},
        {"hot_keys": [], "split_factor": 2},
        {"hot_keys": "nope", "split_factor": 2},
        {"hot_keys": [(1,)], "split_factor": 2, "extra": 1},
        {"hot_keys": [[1]], "split_factor": 2},
        {"hot_keys": [(1, 2)], "split_factor": 2},
        {"hot_keys": [(True,)], "split_factor": 2},
        {"hot_keys": [(1.0,)], "split_factor": 2},
        {"hot_keys": [(b"x",)], "split_factor": 2},
        {"hot_keys": [(1,)], "split_factor": 1},
        {"hot_keys": [(1,)], "split_factor": 17},
        {"hot_keys": [(1,)], "split_factor": True},
        {"hot_keys": [(1,)], "split_factor": 2.0},
    ],
)
def test_invalid_configs_raise(bad):
    left = pd.DataFrame({"k": [1, 2], "v": [1, 2]})
    dfl = dfr = _ddf(left, 2)
    with pytest.raises(ValueError):
        dfl.merge(dfr, on="k", skew_join=bad)


def test_validation_does_not_consume_source():
    left = pd.DataFrame({"k": [1, 2], "v": [1, 2]})
    dfl = dfr = _ddf(left, 2)

    def boom(part):
        raise RuntimeError("source task executed during planning")

    broken = dfl.map_partitions(boom, meta=left.iloc[0:0])
    # Graph construction and optimization must succeed without touching data.
    out = broken.merge(dfr, on="k", skew_join=CFG)
    out.__dask_graph__()
    out.optimize()
    # A malformed config is rejected on the same broken source without compute.
    with pytest.raises(ValueError):
        broken.merge(dfr, on="k", skew_join={"hot_keys": [(1, 2)],
                                            "split_factor": 2})


def test_unsupported_combinations_rejected():
    left = pd.DataFrame({"k": [1, 2], "v": [1, 2]})
    dfl = dfr = _ddf(left, 2)
    # Rejected while constructing the graph (lowering), without computing.
    with pytest.raises(ValueError):
        dfl.merge(dfr, on="k", how="leftsemi", skew_join=CFG).__dask_graph__()
    with pytest.raises(ValueError):
        dfl.merge(
            dfr, on="k", broadcast=True, skew_join=CFG
        ).__dask_graph__()
    with pytest.raises(ValueError):
        dfl.merge(
            dfr, on="k", shuffle_method="p2p", skew_join=CFG
        ).__dask_graph__()
    indexed = _ddf(left.set_index("k"), 2)
    with pytest.raises(ValueError):
        indexed.merge(
            indexed, left_index=True, right_index=True, skew_join=CFG
        ).__dask_graph__()


def test_normalized_configs_share_graph_keys_but_configs_are_isolated():
    left = pd.DataFrame({"k": [1, 2], "v": [1, 2]})
    dfl = dfr = _ddf(left, 2)
    a = dfl.merge(
        dfr, on="k", skew_join={"hot_keys": [(np.int64(1),), (1,)],
                                "split_factor": 2}
    )
    b = dfl.merge(dfr, on="k", skew_join=CFG)
    assert set(map(str, a.__dask_graph__())) == set(
        map(str, b.__dask_graph__())
    )

    def skew_keys(coll):
        return {
            k for k in coll.__dask_graph__() if "skew" in str(k[0])
        }

    c = dfl.merge(dfr, on="k", skew_join={"hot_keys": [(1,)],
                                          "split_factor": 3})
    d = dfl.merge(dfr, on="k", skew_join={"hot_keys": [(2,)],
                                          "split_factor": 2})
    assert skew_keys(b).isdisjoint(skew_keys(c))
    assert skew_keys(b).isdisjoint(skew_keys(d))


def test_default_mode_unchanged_without_skew_join():
    left = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]})
    got = _ddf(left, 2).merge(_ddf(left, 2), on="k")
    # No SkewJoin layer appears in the optimized plan.
    assert "SkewJoin" not in _plan_text(got)


# ---------------------------------------------------------------------------
# Lifecycle: metadata, optimize, culling, persist, joint evaluation, schedulers
# ---------------------------------------------------------------------------


@pytest.fixture
def skew_pair():
    left = pd.DataFrame(
        {"k": [1] * 20 + [2, 3, 4, 5], "v": range(24)}
    )
    right = pd.DataFrame({"k": [1, 1, 2, 6], "w": range(4)})
    return _ddf(left, 5), _ddf(right, 2), left, right


def test_metadata_and_npartitions(skew_pair):
    dfl, dfr, left, right = skew_pair
    cfg = {"hot_keys": [(1,)], "split_factor": 3}
    out = dfl.merge(dfr, on="k", how="outer", skew_join=cfg)
    # Metadata is available without computing sources.
    assert list(out.columns) == list(
        left.merge(right, on="k", how="outer").columns
    )
    # normal partitions (5) + H*S*S cells (9) + 2*H*S emissions (6) = 20
    assert out.npartitions == 5 + 9 + 6


def test_optimize_and_projection_filter_pushdown(skew_pair):
    dfl, dfr, left, right = skew_pair
    out = dfl.merge(dfr, on="k", how="left", skew_join=CFG)
    expected = left.merge(right, on="k", how="left")
    assert_eq(out.optimize(), expected, check_index=False)

    proj = dfl.merge(dfr, on="k", how="left", skew_join=CFG)[["v", "w"]]
    assert_eq(
        proj, expected[["v", "w"]], check_index=False,
    )
    filt = dfl[dfl.v > 2].merge(dfr, on="k", how="inner", skew_join=CFG)
    assert_eq(
        filt,
        left[left.v > 2].merge(right, on="k", how="inner"),
        check_index=False,
    )


def test_partition_culling(skew_pair):
    dfl, dfr, left, right = skew_pair
    out = dfl.merge(dfr, on="k", how="outer", skew_join=CFG)
    expected = left.merge(right, on="k", how="outer")
    # Repeated single-partition reads must each resolve to the right key.
    pieces = [out.partitions[i].compute() for i in range(out.npartitions)]
    assert_eq(
        pd.concat(pieces), expected, check_index=False, check_divisions=False
    )
    # Culled graph contains exactly the one requested output key.
    culled = out.partitions[3].optimize()
    keys = [
        k for k in culled.__dask_graph__() if isinstance(k, tuple)
        and "skewjoin" in str(k[0])
    ]
    assert keys == [culled.__dask_keys__()[0]]


def test_persist_and_joint_evaluation(skew_pair):
    dfl, dfr, left, right = skew_pair
    out = dfl.merge(dfr, on="k", how="outer", skew_join=CFG)
    expected = left.merge(right, on="k", how="outer")
    persisted = out.persist()
    assert_eq(persisted, expected, check_index=False)
    assert_eq(persisted, expected, check_index=False)  # reuse

    other = dfl.merge(dfr, on="k", how="inner", skew_join=CFG)
    r1, r2 = dask.compute(out, other)
    assert_eq(
        r1, expected, check_index=False, check_divisions=False
    )
    assert_eq(
        r2,
        left.merge(right, on="k", how="inner"),
        check_index=False,
        check_divisions=False,
    )


@pytest.mark.parametrize("scheduler", ["sync", "threads"])
def test_schedulers(skew_pair, scheduler):
    dfl, dfr, left, right = skew_pair
    for method in ALL_METHODS:
        out = dfl.merge(
            dfr, on="k", how="outer", skew_join=CFG, shuffle_method=method
        )
        got = out.compute(scheduler=scheduler)
        assert_eq(
            got,
            left.merge(right, on="k", how="outer"),
            check_index=False,
            check_divisions=False,
        )


def test_local_distributed_client():
    distributed = pytest.importorskip("distributed")
    from distributed import Client

    left = pd.DataFrame(
        {"k": [1] * 20 + [2, 3, 4, 5, 6], "v": range(25)}
    )
    right = pd.DataFrame({"k": [1, 1, 2, 7], "w": range(4)})
    cfg = {"hot_keys": [(1,)], "split_factor": 3}
    with Client(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
        silence_logs=40,
    ):
        for method in ALL_METHODS:
            out = _ddf(left, 5).merge(
                _ddf(right, 3),
                on="k",
                how="outer",
                skew_join=cfg,
                shuffle_method=method,
            )
            got = out.compute()
            assert len(got) == len(left.merge(right, on="k", how="outer"))


# ---------------------------------------------------------------------------
# Dispersal bound verified through the instrumentation probe
# ---------------------------------------------------------------------------


def _run_with_probe(collection):
    probe_dir = tempfile.mkdtemp(prefix="skew-test-probe-")
    old = os.environ.get(_INSTRUMENT_ENV)
    os.environ[_INSTRUMENT_ENV] = probe_dir
    try:
        (frame,) = dask.compute(collection)
        records = []
        for name in os.listdir(probe_dir):
            with open(os.path.join(probe_dir, name)) as fh:
                records.append(json.load(fh))
    finally:
        if old is None:
            os.environ.pop(_INSTRUMENT_ENV, None)
        else:
            os.environ[_INSTRUMENT_ENV] = old
        shutil.rmtree(probe_dir, ignore_errors=True)
    return frame, records


def test_dispersal_bound_identical_hot_rows():
    # 1024 fully identical hot rows on both sides; even identical content must
    # spread instead of collapsing onto a whole-key hash.
    n = 1024
    S, Pl, Pr = 4, 4, 4
    left = pd.DataFrame({"k": [777] * n, "v": range(n)})
    right = pd.DataFrame({"k": [777] * n, "w": range(n)})
    out = _ddf(left, Pl).merge(
        _ddf(right, Pr),
        on="k",
        how="inner",
        skew_join={"hot_keys": [(777,)], "split_factor": S},
        shuffle_method="tasks",
    )
    frame, records = _run_with_probe(out)
    assert len(frame) == n * n

    pairs = [r for r in records if r["kind"] == "pair"]
    gathers = [r for r in records if r["kind"] == "gather"]
    splits = [r for r in records if r["kind"] == "split"]

    # Exactly Pl + Pr split tasks, each holding only its own partition.
    assert len(splits) == Pl + Pr
    partition_size = n // Pl
    assert all(r["n_rows"] <= partition_size for r in splits)

    # S salt buckets per side, each balanced (round-robin over identical rows).
    left_salts = sorted(
        r["n_rows"]
        for r in gathers
        if r.get("side") == "left" and r["piece"][0] == 0
    )
    right_salts = sorted(
        r["n_rows"]
        for r in gathers
        if r.get("side") == "right" and r["piece"][0] == 0
    )
    assert left_salts == [n // S] * S
    assert right_salts == [n // S] * S

    # S*S pairing cells; each holds n/S rows from each side.
    assert len(pairs) == S * S
    bound = math.ceil(max(n, n) / S) + min(n, n) + Pl + Pr
    assert all(
        r["n_left"] + r["n_right"] <= bound for r in pairs
    )
    # Each original row reaches exactly S pairing tasks on its side.
    assert all(
        r["n_left"] == n // S and r["n_right"] == n // S for r in pairs
    )


def test_dispersal_bound_large_small_skew():
    # 90% hot on the left, only 4 hot rows on the right.
    Pl = Pr = 8
    S = 2
    n_left = 2000 * Pl
    n_hot_l = int(round(n_left * 0.9))
    left = pd.DataFrame(
        {
            "k": [777] * n_hot_l
            + [10 + (i % 500) for i in range(n_left - n_hot_l)],
            "v": range(n_left),
        }
    )
    right = pd.DataFrame(
        {"k": [777] * 4 + [6000 + i for i in range(2000 * Pr - 4)],
         "w": range(2000 * Pr)}
    )
    out = _ddf(left, Pl).merge(
        _ddf(right, Pr),
        on="k",
        how="outer",
        skew_join={"hot_keys": [(777,)], "split_factor": S},
        shuffle_method="tasks",
    )
    _, records = _run_with_probe(out)
    Lh, Rh = n_hot_l, 4
    bound = math.ceil(max(Lh, Rh) / S) + min(Lh, Rh) + Pl + Pr
    for r in records:
        if r["kind"] == "pair":
            assert r["n_left"] + r["n_right"] <= bound
        if r["kind"] == "gather" and r["piece"][0] == 0:
            assert r["n_rows"] <= bound
        if r["kind"] == "emit":
            assert r["n_rows"] <= bound
    # Salt balance on the big side.
    big_salts = [
        r["n_rows"]
        for r in records
        if r["kind"] == "gather"
        and r.get("side") == "left"
        and r["piece"][0] == 0
    ]
    assert max(big_salts) - min(big_salts) <= 1
    assert sum(big_salts) == Lh


def test_split_tasks_never_gather_whole_side():
    # The split exception only permits holding the local source partition.
    Pl = Pr = 16
    left = pd.DataFrame(
        {"k": [777] * (2000 * Pl), "v": range(2000 * Pl)}
    )
    right = pd.DataFrame({"k": [777] * 4, "w": range(4)})
    out = _ddf(left, Pl).merge(
        _ddf(right, Pr),
        on="k",
        how="inner",
        skew_join={"hot_keys": [(777,)], "split_factor": 4},
        shuffle_method="disk",
    )
    _, records = _run_with_probe(out)
    split_rows = [r["n_rows"] for r in records if r["kind"] == "split"]
    # No split task ever sees more than one source partition (2000 rows).
    assert max(split_rows) <= 2000


# ---------------------------------------------------------------------------
# Downstream consumers must re-shuffle because the key mapping is invalidated
# ---------------------------------------------------------------------------


def test_partition_mapping_invalidated(skew_pair):
    dfl, dfr, _, _ = skew_pair
    skew = dfl.merge(dfr, on="k", how="inner", skew_join=CFG)
    normal = dfl.merge(dfr, on="k", how="inner")
    assert skew.expr.unique_partition_mapping_columns_from_shuffle == set()
    assert normal.expr.unique_partition_mapping_columns_from_shuffle == {"k"}


def test_downstream_groupby_dropduplicates_merge(skew_pair):
    from dask.dataframe.dask_expr._reductions import ApplyConcatApply

    dfl, dfr, left, right = skew_pair
    joined = dfl.merge(dfr, on="k", how="inner", skew_join=CFG)
    pandas_joined = left.merge(right, on="k", how="inner")

    # --- groupby: correctness against pandas on the full joined result ---
    gb = joined.groupby("k").agg({"v": "sum", "w": "mean"})
    assert_eq(
        gb,
        pandas_joined.groupby("k").agg({"v": "sum", "w": "mean"}),
        check_index=False,
        check_divisions=False,
    )

    def aca_need_shuffle(coll):
        return [
            e.need_to_shuffle
            for e in coll.expr.walk()
            if isinstance(e, ApplyConcatApply)
        ]

    # The skew result advertises no key co-location, so the split_out groupby
    # must shuffle; the ordinary join can reuse its existing partition mapping.
    skew_gb = joined.groupby("k").agg(
        {"v": "sum"}, split_out=joined.npartitions
    )
    normal_gb = dfl.merge(dfr, on="k", how="inner").groupby("k").agg(
        {"v": "sum"}, split_out=dfl.npartitions
    )
    assert any(aca_need_shuffle(skew_gb))
    assert not any(aca_need_shuffle(normal_gb))
    assert "Shuffle" in _plan_text(skew_gb)

    # --- drop_duplicates ---
    dd = joined.drop_duplicates(subset=["k"])
    assert_eq(
        dd,
        pandas_joined.drop_duplicates(subset=["k"]),
        check_index=False,
        check_divisions=False,
    )
    skew_dd = joined.drop_duplicates(subset=["k"])
    normal_dd = dfl.merge(dfr, on="k", how="inner").drop_duplicates(
        subset=["k"]
    )
    assert any(aca_need_shuffle(skew_dd))
    assert not any(aca_need_shuffle(normal_dd))

    # --- second merge on the same key ---
    third = pd.DataFrame({"k": [1, 2, 3], "z": [100, 200, 300]})
    dthird = _ddf(third, 2)
    m2 = joined.merge(dthird, on="k", how="left")
    assert_eq(
        m2, pandas_joined.merge(third, on="k", how="left"), check_index=False
    )
    # The second merge cannot assume the skew output is key-partitioned, so a
    # shuffle layer appears (on the new input or on the skew result).
    assert "Shuffle" in _plan_text(m2)


def test_disk_partd_cleanup(skew_pair, tmp_path):
    dfl, dfr, _, _ = skew_pair
    os.environ["DASK_TEMPORARY_DIRECTORY"] = str(tmp_path)
    try:
        dfl.merge(
            dfr, on="k", how="outer", skew_join=CFG, shuffle_method="disk"
        ).compute()
        # The explicit reference-counted cleanup removes every partd dir.
        assert not list(tmp_path.glob("**/*.partd"))
    finally:
        os.environ.pop("DASK_TEMPORARY_DIRECTORY", None)


def test_disk_cleanup_after_source_failure(skew_pair, tmp_path):
    dfl, dfr, _, _ = skew_pair
    os.environ["DASK_TEMPORARY_DIRECTORY"] = str(tmp_path)

    def boom(part):
        raise ValueError("source failure")

    broken = dfl.map_partitions(boom, meta=dfl._meta.iloc[0:0])
    try:
        with pytest.raises(ValueError):
            broken.merge(
                dfr, on="k", how="inner", skew_join=CFG,
                shuffle_method="disk",
            ).compute()
    finally:
        os.environ.pop("DASK_TEMPORARY_DIRECTORY", None)
    # ensure_cleanup_on_exception removes the partd directory on append errors
    # whenever it got created; no skew partd directory may remain.
    assert not list(tmp_path.glob("**/*.partd"))


def test_npartitions_controls_normal_partitions_only():
    left = pd.DataFrame({"k": [1] * 100 + list(range(50)), "v": range(150)})
    right = pd.DataFrame({"k": [1] * 4 + list(range(20)), "w": range(24)})
    n_normal = 7
    out = _ddf(left, 4).merge(
        _ddf(right, 3),
        on="k",
        how="inner",
        skew_join={"hot_keys": [(1,)], "split_factor": 2},
        npartitions=n_normal,
    )
    # normal partitions + H*S*S + 2*H*S
    assert out.npartitions == n_normal + 4 + 4
