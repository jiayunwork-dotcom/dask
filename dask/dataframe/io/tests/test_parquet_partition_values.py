"""Regression tests for Hive partition round-tripping.

These cover the "partition value" bugs where a slash (or a percent-encoded
sequence) in a partition value was not preserved, and where all-null
partition columns could not be read back at all.

Reference environment: Python 3.12, pandas 2.2.3, pyarrow 19.0.1.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

import dask.dataframe as dd

try:
    import pyarrow as pa
    import pyarrow.dataset as pa_ds
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None

pytestmark = pytest.mark.skipif(pa is None, reason="pyarrow not installed")


def _collect_ids(path, filters=None, columns=("id",), filesystem=None):
    kwargs = {"columns": list(columns)}
    if filters is not None:
        kwargs["filters"] = filters
    if filesystem is not None:
        kwargs["filesystem"] = filesystem
    return sorted(dd.read_parquet(path, **kwargs).compute()["id"].tolist())


def _normalize(values):
    return [None if pd.isna(v) else v for v in values]


def _assert_partition_values(path, expected):
    """Verify (id -> part) mapping for both read paths and via Arrow."""
    expected = dict(zip(range(len(expected)), _normalize(expected)))
    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        got = dd.read_parquet(path, **kwargs).compute().sort_values("id")
        assert _normalize(got["part"].tolist()) == list(expected.values())
    # A standard, non-Dask Hive reader must agree
    table = pa_ds.dataset(path, format="parquet", partitioning="hive").to_table()
    table = table.sort_by("id")
    assert _normalize(table.column("part").to_pylist()) == list(expected.values())


@pytest.fixture
def special_values_df():
    return pd.DataFrame(
        {"part": ["a/b", "a%2Fb", "a=b", "雪", ""], "id": range(5)}
    )


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_partition_value_slash_and_percent_roundtrip(tmpdir, special_values_df):
    path = str(tmpdir)
    dd.from_pandas(special_values_df, npartitions=2).to_parquet(
        path, partition_on=["part"], write_index=False
    )

    # One directory level per partition value (no extra nesting for "a/b")
    dirs = sorted(os.listdir(path))
    assert dirs == sorted(
        [
            "part=a%2Fb",
            "part=a%252Fb",
            "part=a%3Db",
            "part=%E9%9B%AA",
            "part=",
        ]
    )

    _assert_partition_values(path, special_values_df["part"].tolist())

    # Filters use *logical* values
    for value, expected_id in [
        ("a/b", 0),
        ("a%2Fb", 1),
        ("a=b", 2),
        ("雪", 3),
        ("", 4),
    ]:
        assert _collect_ids(path, [("part", "==", value)]) == [expected_id]
        assert _collect_ids(
            path, [("part", "==", value)], filesystem="arrow"
        ) == [expected_id]


@pytest.mark.parametrize("write_metadata_file", [False, True])
def test_partition_value_slash_metadata_options(tmpdir, write_metadata_file):
    df = pd.DataFrame({"part": ["a/b", "a%2Fb", None], "id": range(3)})
    path = str(tmpdir)
    dd.from_pandas(df, npartitions=2).to_parquet(
        path,
        partition_on=["part"],
        write_index=False,
        write_metadata_file=write_metadata_file,
    )
    for ignore in [False, True]:
        got = (
            dd.read_parquet(path, ignore_metadata_file=ignore)
            .compute()
            .sort_values("id")
        )
        assert _normalize(got["part"].tolist()) == ["a/b", "a%2Fb", None]


def test_null_partition_string_roundtrip(tmpdir):
    df = pd.DataFrame({"part": [None, "x", None, "y"], "id": range(4)})
    path = str(tmpdir)
    dd.from_pandas(df, npartitions=2).to_parquet(
        path, partition_on=["part"], write_index=False
    )
    assert sorted(os.listdir(path)) == sorted(
        ["part=__HIVE_DEFAULT_PARTITION__", "part=x", "part=y"]
    )

    _assert_partition_values(path, df["part"].tolist())

    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        assert sorted(
            dd.read_parquet(
                path, filters=[("part", "is", None)], **kwargs
            ).compute()["id"]
        ) == [0, 2]
        assert sorted(
            dd.read_parquet(
                path, filters=[("part", "is not", None)], **kwargs
            ).compute()["id"]
        ) == [1, 3]
        # Filtering without emitting the filter column
        assert _collect_ids(path, [("part", "is", None)], filesystem=fs) == [0, 2]
        assert _collect_ids(
            path, [("part", "is not", None)], filesystem=fs
        ) == [1, 3]

    # The plain Arrow Hive reader agrees, too
    table = pa_ds.dataset(path, format="parquet", partitioning="hive").to_table()
    assert sorted(
        table.filter(pa_ds.field("part").is_null()).column("id").to_pylist()
    ) == [0, 2]


def test_all_null_partition_single_file(tmpdir):
    """A file whose partition column is *always* null must still be readable."""
    path = os.path.join(str(tmpdir), "part=__HIVE_DEFAULT_PARTITION__")
    os.makedirs(path)
    pq.write_table(pa.table({"id": [0, 1], "v": ["a", "b"]}), os.path.join(path, "f.parquet"))

    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        result = dd.read_parquet(str(tmpdir), **kwargs).compute()
        assert sorted(result["id"].tolist()) == [0, 1]
        assert result["part"].isna().all()
        filtered = dd.read_parquet(
            str(tmpdir), filters=[("part", "is", None)], **kwargs
        ).compute()
        assert sorted(filtered["id"].tolist()) == [0, 1]


def test_two_partition_columns_and_dnf_filters(tmpdir):
    df = pd.DataFrame(
        {
            "p": ["a", "a", "b", "b"],
            "q": ["x", "y", "x", "y"],
            "id": [0, 1, 2, 3],
        }
    )
    path = str(tmpdir)
    dd.from_pandas(df, npartitions=2).to_parquet(
        path, partition_on=["p", "q"], write_index=False
    )
    assert sorted(os.listdir(path)) == ["p=a", "p=b"]

    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        got = dd.read_parquet(path, **kwargs).compute().sort_values("id")
        assert list(zip(got["p"], got["q"])) == [
            ("a", "x"),
            ("a", "y"),
            ("b", "x"),
            ("b", "y"),
        ]
        # AND within a conjunction
        assert _collect_ids(
            path, [[("p", "==", "a"), ("q", "==", "y")]], filesystem=fs
        ) == [1]
        # OR across conjunctions (DNF)
        assert _collect_ids(
            path,
            [[("p", "==", "a"), ("q", "==", "y")], [("p", "==", "b")]],
            filesystem=fs,
        ) == [1, 2, 3]


def test_categorical_partition_missing_and_unused(tmpdir):
    df = pd.DataFrame(
        {
            "part": pd.Categorical(
                ["a", None, "a", "b"], categories=["a", "b", "c", "d"]
            ),
            "id": range(4),
        }
    )
    path = str(tmpdir)
    dd.from_pandas(df, npartitions=2).to_parquet(
        path, partition_on=["part"], write_index=False
    )
    # Unused categories do not create (empty) directories
    assert sorted(os.listdir(path)) == sorted(
        ["part=__HIVE_DEFAULT_PARTITION__", "part=a", "part=b"]
    )

    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        got = dd.read_parquet(path, **kwargs).compute().sort_values("id")
        assert got["part"].isna().tolist() == [False, True, False, False]
        assert got["part"].astype(object).where(got["part"].notna(), None).tolist() == [
            "a",
            None,
            "a",
            "b",
        ]
        assert _collect_ids(path, [("part", "is", None)], filesystem=fs) == [1]
        assert _collect_ids(path, [("part", "==", "b")], filesystem=fs) == [3]


def test_empty_write_partition_is_harmless(tmpdir):
    import dask

    meta = pd.DataFrame({"part": pd.Series([], dtype=object), "id": pd.Series([], dtype="int64")})
    partitions = [
        meta,
        pd.DataFrame({"part": ["a", "a"], "id": [0, 1]}),
        meta,
    ]
    ddf = dd.from_delayed([dask.delayed(p) for p in partitions], meta=meta)
    path = str(tmpdir)
    ddf.to_parquet(path, partition_on=["part"], write_index=False)
    assert sorted(os.listdir(path)) == ["part=a"]
    assert _collect_ids(path) == [0, 1]


def test_arrow_written_hive_dataset_read_by_dask(tmpdir):
    """Cross-check: standard Arrow Hive output must be read correctly."""
    values = ["a/b", "a%2Fb", "a=b", "雪", "", None]
    table = pa.table({"part": values, "id": range(len(values))})
    path = str(tmpdir)
    pa_ds.write_dataset(
        table,
        path,
        format="parquet",
        partitioning=pa_ds.HivePartitioning(pa.schema([("part", pa.string())])),
    )
    expected = _normalize(values)

    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        got = dd.read_parquet(path, **kwargs).compute().sort_values("id")
        assert _normalize(got["part"].tolist()) == expected
        assert _collect_ids(
            path, [("part", "==", "a/b")], filesystem=fs
        ) == [0]
        assert _collect_ids(
            path, [("part", "is", None)], filesystem=fs
        ) == [5]
        assert _collect_ids(
            path, [("part", "is not", None)], filesystem=fs
        ) == [0, 1, 2, 3, 4]


def test_partition_filter_and_projection_pruning(tmpdir):
    """Column projection and file pruning must still work with logical values."""
    import shutil

    df = pd.DataFrame(
        {"part": ["a/b", "a/b", "x", None], "id": range(4), "v": range(100, 104)}
    )
    path = str(tmpdir)
    dd.from_pandas(df, npartitions=2).to_parquet(
        path, partition_on=["part"], write_index=False
    )

    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        # Only project the id column; ensure partition columns are pruned too
        result = dd.read_parquet(
            path, columns=["id"], filters=[("part", "==", "a/b")], **kwargs
        )
        assert sorted(result.compute()["id"].tolist()) == [0, 1]

        # The read task must not request the filter/partition columns as
        # physical columns when they are not in the projection.
        from dask.dataframe.dask_expr.io import ReadParquet

        optimized = result.optimize()
        read_ops = list(optimized.expr.find_operations(ReadParquet))
        assert read_ops
        for read_op in read_ops:
            assert "v" not in (read_op.operand("columns") or [])

    # Prove file pruning: after the plan is built, remove the directories
    # that do not match the filter.  The result must still be correct, which
    # means only the matching file was scheduled.
    pruned = dd.read_parquet(path, filters=[("part", "==", "x")], columns=["id"])
    optimized = pruned.optimize()
    for name in ("part=a%2Fb", "part=__HIVE_DEFAULT_PARTITION__"):
        shutil.rmtree(os.path.join(path, name), ignore_errors=True)
    assert sorted(optimized.compute()["id"].tolist()) == [2]


def test_no_full_table_read_before_filter(tmpdir):
    """Execution must not decode non-matching files."""
    import shutil

    df = pd.DataFrame({"part": ["a", "b", "c"], "id": [0, 1, 2]})
    path = str(tmpdir)
    dd.from_pandas(df, npartitions=1).to_parquet(
        path, partition_on=["part"], write_index=False
    )
    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        result = dd.read_parquet(
            path, filters=[("part", "==", "b")], **kwargs
        ).optimize()
        for name in ("part=a", "part=c"):
            shutil.rmtree(os.path.join(path, name), ignore_errors=True)
        assert sorted(result.compute()["id"].tolist()) == [1]
        # Restore for the next read path
        dd.from_pandas(df.iloc[[0, 2]], npartitions=1).to_parquet(
            path, partition_on=["part"], write_index=False, append=True
        )


def test_calculate_divisions_with_partition_filter(tmpdir):
    df = pd.DataFrame({"part": ["a", "b", "c"], "id": [0, 1, 2]})
    path = str(tmpdir)
    dd.from_pandas(df, npartitions=2).to_parquet(
        path, partition_on=["part"], write_index=False
    )
    result = dd.read_parquet(path, filters=[("part", "is not", None)])
    assert sorted(result.compute()["id"].tolist()) == [0, 1, 2]


def test_no_matching_partition_filter_is_empty(tmpdir):
    df = pd.DataFrame({"part": ["a", "b"], "id": [0, 1]})
    path = str(tmpdir)
    dd.from_pandas(df, npartitions=2).to_parquet(
        path, partition_on=["part"], write_index=False
    )
    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        result = dd.read_parquet(path, filters=[("part", "==", "zzz")], **kwargs)
        assert result.compute()["id"].tolist() == []
        assert list(result.compute().columns) == ["id", "part"]


def test_changing_filters_does_not_reuse_stale_plan(tmpdir):
    """Re-filtering a previously materialized read must not reuse its plan."""
    df = pd.DataFrame({"part": ["a", "b", "a", "b"], "id": [0, 1, 2, 3]})
    path = str(tmpdir)
    dd.from_pandas(df, npartitions=2).to_parquet(
        path, partition_on=["part"], write_index=False
    )
    from dask.dataframe.dask_expr.io import ReadParquet

    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        read = dd.read_parquet(path, filters=[("part", "==", "a")], **kwargs)
        # Materialize the dataset-info / plan cache for this expression
        for read_op in read.expr.find_operations(ReadParquet):
            read_op._dataset_info
        re_filtered = read.substitute_parameters({"filters": [("part", "==", "b")]})
        assert sorted(re_filtered.compute()["id"].tolist()) == [1, 3]
