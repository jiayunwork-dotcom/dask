"""Scale experiments for the two-level composite ``set_index``.

Per input partition we generate 20,000 rows and run 8 or 32 input partitions
with an 8-partition target.  Three distributions are exercised:

* ``uniform``     -- both levels spread uniformly,
* ``skew_first``  -- 90% of rows share the first level (account) but the
  second level is spread out,
* ``skew_full``   -- 90% of rows share the *complete* key.

For each (input partitions, distribution, shuffle method) combination the
script records:

* the number of raw keys each input partition handed to the planner
  (``uploaded_keys``) and the planner's state bound,
* the peak number of keys the planner held at once,
* the per-output-partition row counts,
* the bytes that crossed the exchange (summed over the partition-id tasks),
* wall-clock time.

The test side never concatenates the result into one pandas frame: row counts
and an independent row-identity digest are computed per partition.

Run with::

    python scratch/run_composite_index_experiments.py --output results.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

import dask
from dask.dataframe.dask_expr import from_map
from dask.dataframe.dask_expr._multi_index_expr import (
    _COMPOSITE_INSTRUMENT_ENV,
    LAST_PLAN_STATS,
)

ROWS_PER_PARTITION = 20_000
TARGET_PARTITIONS = 8


def make_partition(kind: str, part: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed * 1000 + part)
    n = ROWS_PER_PARTITION
    payload = np.arange(part * n, (part + 1) * n)
    if kind == "uniform":
        account = rng.integers(0, 1000, n)
        seq = rng.integers(0, 10_000, n)
    elif kind == "skew_first":
        hot = rng.random(n) < 0.9
        account = np.where(hot, 7, rng.integers(0, 1000, n))
        seq = rng.integers(0, 10_000, n)
    elif kind == "skew_full":
        hot = rng.random(n) < 0.9
        account = np.where(hot, 7, rng.integers(0, 1000, n))
        seq = np.where(hot, 4242, rng.integers(0, 10_000, n))
    else:  # pragma: no cover - guarded by CLI
        raise ValueError(kind)
    return pd.DataFrame({"account": account, "seq": seq, "payload": payload})


def build_frame(kind: str, npartitions: int):
    meta = pd.DataFrame(
        {
            "account": pd.Series([], dtype="int64"),
            "seq": pd.Series([], dtype="int64"),
            "payload": pd.Series([], dtype="int64"),
        }
    )

    def read(part):
        return make_partition(kind, part, seed=0)

    return from_map(read, list(range(npartitions)), meta=meta)


def digest_partition(frame: pd.DataFrame) -> dict:
    if len(frame) == 0:
        return {"rows": 0, "payload_digest": 0, "key_min": None, "key_max": None}
    payload = frame["payload"].to_numpy()
    ordered = np.sort(payload)
    return {
        "rows": int(len(frame)),
        "payload_digest": int(hash(ordered.tobytes())),
        "key_min": [int(x) for x in frame.index.min()],
        "key_max": [int(x) for x in frame.index.max()],
    }


def run_case(kind: str, npartitions: int, method: str) -> dict:
    instrument_dir = tempfile.mkdtemp(prefix="composite-bench-")
    os.environ[_COMPOSITE_INSTRUMENT_ENV] = instrument_dir
    try:
        frame = build_frame(kind, npartitions)
        started = time.perf_counter()
        result = frame.set_index(
            ["account", "seq"],
            npartitions=TARGET_PARTITIONS,
            shuffle_method=method,
        )
        divisions = result.divisions
        planned = time.perf_counter() - started
        plan_stats = dict(LAST_PLAN_STATS)

        started = time.perf_counter()
        persisted = result.persist(scheduler="sync")
        compute_seconds = time.perf_counter() - started

        # Per-partition verification against the computed pandas reference.
        expected = (
            pd.concat(
                [make_partition(kind, p, seed=0) for p in range(npartitions)],
                ignore_index=True,
            )
            .set_index(["account", "seq"])
            .sort_index()
        )
        per_partition = []
        total_rows = 0
        digests_match = True
        for i in range(persisted.npartitions):
            part = persisted.get_partition(i).compute(scheduler="sync")
            record = digest_partition(part)
            record["index"] = i
            per_partition.append(record)
            total_rows += record["rows"]
            # Independent slice-based expectation from the sorted reference,
            # never concatenating the dask output.
            if len(part):
                lo = divisions[i]
                hi = divisions[i + 1]
                last = i == persisted.npartitions - 1
                mask = (expected.index >= lo) & (
                    expected.index <= hi if last else expected.index < hi
                )
                want = expected["payload"].to_numpy()[mask]
                digests_match = digests_match and int(
                    hash(np.sort(want).tobytes())
                ) == record["payload_digest"]

        bytes_total = 0
        for path in glob.glob(os.path.join(instrument_dir, "*.json")):
            with open(path) as handle:
                bytes_total += json.load(handle)["bytes"]

        return {
            "distribution": kind,
            "input_partitions": npartitions,
            "target_partitions": TARGET_PARTITIONS,
            "shuffle_method": method,
            "output_partitions": persisted.npartitions,
            "divisions": [[int(a), int(b)] for a, b in divisions],
            "planning_seconds": round(planned, 4),
            "compute_seconds": round(compute_seconds, 4),
            "uploaded_keys": plan_stats.get("uploaded_keys"),
            "planner_state_keys": plan_stats.get("planner_state_keys"),
            "planner_sample_bound": plan_stats.get("planner_sample_bound"),
            "total_rows": total_rows,
            "expected_rows": int(len(expected)),
            "rows_match": total_rows == len(expected),
            "payload_digests_match": digests_match,
            "exchange_bytes": bytes_total,
            "per_partition": per_partition,
        }
    finally:
        os.environ.pop(_COMPOSITE_INSTRUMENT_ENV, None)
        shutil.rmtree(instrument_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="composite_index_results.json")
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["uniform", "skew_first", "skew_full"],
    )
    parser.add_argument("--partitions", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--methods", nargs="+", default=["tasks", "disk"])
    args = parser.parse_args()

    results = []
    for kind in args.distributions:
        for npartitions in args.partitions:
            for method in args.methods:
                case = run_case(kind, npartitions, method)
                results.append(case)
                print(
                    f"{kind:11s} parts={npartitions:2d} {method:5s} "
                    f"out={case['output_partitions']:2d} "
                    f"rows={case['total_rows']:6d} "
                    f"match={case['rows_match'] and case['payload_digests_match']} "
                    f"uploaded={case['uploaded_keys']:6d} "
                    f"bytes={case['exchange_bytes']:9d} "
                    f"t={case['compute_seconds']:.2f}s"
                )
    with open(args.output, "w") as handle:
        json.dump({"dask_version": dask.__version__,
                   "results": results}, handle, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
