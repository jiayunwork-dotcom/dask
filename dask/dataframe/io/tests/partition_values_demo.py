"""Reproducible acceptance checks for the Parquet Hive-partition fix.

From the repository root::

    PYTHONPATH=$PWD python dask/dataframe/io/tests/partition_values_demo.py

The pytest regression covering the same behavior is::

    python -m pytest dask/dataframe/io/tests/test_parquet_partition_values.py \
        --no-header -p no:cacheprovider --no-cov

The full affected regression is::

    python -m pytest \
        dask/dataframe/io/tests/test_parquet.py \
        dask/dataframe/dask_expr/io/tests/test_parquet.py \
        dask/dataframe/io/tests/test_parquet_partition_values.py \
        --no-header -p no:cacheprovider --no-cov

The checks use real files on local disk, compare results by ``id`` (never by
scan order), and cover both the fsspec reader and the ``filesystem="arrow"``
reader.  Reference environment: Python 3.12, pandas 2.2.3, pyarrow 19.0.1.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pandas as pd

import dask.dataframe as dd

try:
    import pyarrow as pa
    import pyarrow.dataset as pa_ds
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None


def _read_dir(path, filters=None, columns=None, filesystem=None):
    kwargs = {}
    if filters is not None:
        kwargs["filters"] = filters
    if columns is not None:
        kwargs["columns"] = columns
    if filesystem is not None:
        kwargs["filesystem"] = filesystem
    return dd.read_parquet(path, **kwargs)


def _norm(values):
    return [None if pd.isna(v) else v for v in values]


def demo_roundtrip():
    path = tempfile.mkdtemp()
    df = pd.DataFrame({"part": ["a/b", "a%2Fb", "a=b", "雪", ""], "id": range(5)})
    dd.from_pandas(df, npartitions=2).to_parquet(
        path, partition_on=["part"], write_index=False
    )
    print("1. Dask-written directories:", sorted(os.listdir(path)))
    print("   depth check (one level):", all("/" not in d for d in os.listdir(path)))

    result = dd.read_parquet(path).compute().sort_values("id")
    print("   Dask read values:", result["part"].tolist())

    arrow = (
        pa_ds.dataset(path, format="parquet", partitioning="hive")
        .to_table()
        .sort_by("id")
    )
    print("   PyArrow Hive read values:", arrow.column("part").to_pylist())

    for value, expected in [("a/b", 0), ("a%2Fb", 1), ("a=b", 2), ("雪", 3), ("", 4)]:
        ids = sorted(
            _read_dir(path, [("part", "==", value)], ["id"]).compute()["id"].tolist()
        )
        assert ids == [expected], (value, ids)
    print("   value filters -> correct ids (compared by id)")

    # Show that a partition-filter-only read prunes files and reads only id
    from dask.dataframe.dask_expr.io import ReadParquet

    opt = _read_dir(path, [("part", "==", "a/b")], ["id"]).optimize()
    for read_op in opt.expr.find_operations(ReadParquet):
        print("   optimized read columns:", read_op.operand("columns"))
    shutil.rmtree(os.path.join(path, "part="))
    shutil.rmtree(os.path.join(path, "part=%E9%9B%AA"))
    print(
        "   after removing non-matching dirs, ids still:",
        sorted(opt.compute()["id"].tolist()),
    )
    return True


def demo_nulls():
    path = tempfile.mkdtemp()
    df = pd.DataFrame({"part": [None, "x", None, "y"], "id": range(4)})
    dd.from_pandas(df, npartitions=2).to_parquet(
        path, partition_on=["part"], write_index=False
    )
    print("2. null-partition directories:", sorted(os.listdir(path)))
    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        all_ids = sorted(dd.read_parquet(path, **kwargs).compute()["id"].tolist())
        is_null = sorted(
            dd.read_parquet(
                path, filters=[("part", "is", None)], columns=["id"], **kwargs
            ).compute()["id"].tolist()
        )
        is_not_null = sorted(
            dd.read_parquet(
                path, filters=[("part", "is not", None)], columns=["id"], **kwargs
            ).compute()["id"].tolist()
        )
        print(f"   {fs}: all={all_ids} is_null={is_null} is_not_null={is_not_null}")
        assert all_ids == [0, 1, 2, 3]
        assert is_null == [0, 2]
        assert is_not_null == [1, 3]
    return True


def demo_all_null_file():
    root = tempfile.mkdtemp()
    part_dir = os.path.join(root, "part=__HIVE_DEFAULT_PARTITION__")
    os.makedirs(part_dir)
    pq.write_table(
        pa.table({"id": [0, 1], "v": ["a", "b"]}), os.path.join(part_dir, "f.parquet")
    )
    for fs in ("fsspec", "arrow"):
        kwargs = {"filesystem": fs} if fs == "arrow" else {}
        result = dd.read_parquet(root, **kwargs).compute()
        print(f"3. all-null single file via {fs}:", sorted(result["id"].tolist()))
        assert sorted(result["id"].tolist()) == [0, 1]
    return True


def demo_two_keys_and_categorical():
    path = tempfile.mkdtemp()
    df = pd.DataFrame(
        {"p": ["a", "a", "b", "b"], "q": ["x", "y", "x", "y"], "id": range(4)}
    )
    dd.from_pandas(df, npartitions=2).to_parquet(
        path, partition_on=["p", "q"], write_index=False
    )
    print("4. two-key directories:", sorted(os.listdir(path)))
    got = dd.read_parquet(path).compute().sort_values("id")
    assert list(zip(got["p"], got["q"])) == [
        ("a", "x"),
        ("a", "y"),
        ("b", "x"),
        ("b", "y"),
    ]
    ids = sorted(
        dd.read_parquet(
            path,
            filters=[[("p", "==", "a"), ("q", "==", "y")], [("p", "==", "b")]],
            columns=["id"],
        )
        .compute()["id"]
        .tolist()
    )
    print("   DNF filter ids:", ids)
    assert ids == [1, 2, 3]

    cat_path = tempfile.mkdtemp()
    cat_df = pd.DataFrame(
        {
            "part": pd.Categorical(
                ["a", None, "a", "b"], categories=["a", "b", "c", "d"]
            ),
            "id": range(4),
        }
    )
    dd.from_pandas(cat_df, npartitions=2).to_parquet(
        cat_path, partition_on=["part"], write_index=False
    )
    print("5. categorical directories (unused cats omitted):", sorted(os.listdir(cat_path)))
    null_ids = sorted(
        dd.read_parquet(
            cat_path, filters=[("part", "is", None)], columns=["id"]
        )
        .compute()["id"]
        .tolist()
    )
    assert null_ids == [1]
    print("   categorical is-null ids:", null_ids)
    return True


def demo_cross_writers():
    values = ["a/b", "a%2Fb", "a=b", "雪", "", None]
    dask_path = tempfile.mkdtemp()
    dd.from_pandas(
        pd.DataFrame({"part": values, "id": range(len(values))}), npartitions=2
    ).to_parquet(dask_path, partition_on=["part"], write_index=False)

    arrow_path = tempfile.mkdtemp()
    pa_ds.write_dataset(
        pa.table({"part": values, "id": range(len(values))}),
        arrow_path,
        format="parquet",
        partitioning=pa_ds.HivePartitioning(pa.schema([("part", pa.string())])),
    )

    for label, path in [("Dask-written", dask_path), ("Arrow-written", arrow_path)]:
        for fs in ("fsspec", "arrow"):
            kwargs = {"filesystem": fs} if fs == "arrow" else {}
            got = dd.read_parquet(path, **kwargs).compute().sort_values("id")
            print(f"6. {label} via {fs}:", _norm(got["part"].tolist()))
            assert _norm(got["part"].tolist()) == values
    return True


if __name__ == "__main__":
    demo_roundtrip()
    demo_nulls()
    demo_all_null_file()
    demo_two_keys_and_categorical()
    demo_cross_writers()
    print("All acceptance checks passed.")
