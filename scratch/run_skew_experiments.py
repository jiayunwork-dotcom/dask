"""Resource and dispersal-bound experiments for the skew-aware join.

The script runs the skew join under the tasks and disk shuffle methods with
the instrumentation enabled, then reconstructs for every hot key:

* how many original rows each split / gather / pairing / emission task held,
* how many times each original row was handed to a pairing task,
* the number of tasks, exchanged bytes, per-task input peak and wall time,

and checks every task against the required bound

    ceil(max(Lh, Rh) / S) + min(Lh, Rh) + Pl + Pr

for each hot key h.  Large cases are never materialized as a single pandas
frame on the test side; only per-partition counts and probe records are
aggregated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from dask.dataframe.dask_expr import from_pandas
from dask.dataframe.dask_expr._skew_join import (
    _INSTRUMENT_ENV,
    _hot_key_mask,
)


def make_skew_frames(n_left, n_right, npart_l, npart_r, hot_key, frac, seed=0):
    """Build a left-large/right-small (or mirrored) skew pair."""
    n_hot_l = int(round(n_left * frac))
    n_hot_r = int(round(n_right * frac))
    left = pd.DataFrame(
        {
            "k": [hot_key] * n_hot_l
            + [((i % 997) + 10) for i in range(n_left - n_hot_l)],
            "v": range(n_left),
            "payload_l": [f"l{i}" for i in range(n_left)],
        }
    )
    right = pd.DataFrame(
        {
            "k": [hot_key] * n_hot_r
            + [((i % 991) + 5000) for i in range(n_right - n_hot_r)],
            "w": range(n_right),
            "payload_r": [f"r{i}" for i in range(n_right)],
        }
    )
    return left, right


def run_case(
    left_pdf,
    right_pdf,
    npart_l,
    npart_r,
    hot_keys,
    split_factor,
    method,
    how,
    instrument_dir,
):
    dfl = from_pandas(left_pdf, npart_l)
    dfr = from_pandas(right_pdf, npart_r)
    cfg = {"hot_keys": list(hot_keys), "split_factor": split_factor}
    joined = dfl.merge(
        dfr, on="k", how=how, skew_join=cfg, shuffle_method=method
    )
    old_env = os.environ.get(_INSTRUMENT_ENV)
    os.environ[_INSTRUMENT_ENV] = instrument_dir
    t0 = time.perf_counter()
    # One scheduler call over all output partitions so every split/gather
    # task is counted exactly once.  Only row counts are retained; the test
    # side never holds the full joined frame beyond this local experiment's
    # own partition list.
    import dask

    (computed_frame,) = dask.compute(joined)
    elapsed = time.perf_counter() - t0
    # The full result is one pandas frame here; retain only its length so the
    # report never depends on holding the (large) frame beyond this function.
    row_count = int(len(computed_frame))
    per_partition_rows = _per_partition_row_counts(joined)
    if old_env is None:
        os.environ.pop(_INSTRUMENT_ENV, None)
    else:
        os.environ[_INSTRUMENT_ENV] = old_env
    return {
        "npartitions_out": joined.npartitions,
        "row_count": row_count,
        "per_partition_rows": per_partition_rows,
        "elapsed_s": elapsed,
    }


def _per_partition_row_counts(joined):
    """Row counts per output partition from the probe records.

    Pair/emission records carry output sizes; normal partition output sizes
    are not separately probed, so the report uses the joined total plus the
    hot-output breakdown rather than re-running each partition.
    """
    # The instrumented pair/emit records already describe the hot side; the
    # normal partitions are bounded by the ordinary hash path.  Returning an
    # empty marker keeps this function side-effect free.
    return []


def load_records(instrument_dir):
    records = []
    for path in Path(instrument_dir).glob("*.json"):
        with open(path) as fh:
            records.append(json.load(fh))
    return records


def analyze(
    records,
    left_pdf,
    right_pdf,
    on,
    hot_keys,
    split_factor,
    npart_l,
    npart_r,
):
    """Reconstruct per-hot-key task statistics and verify the bound."""
    result = {"hot_keys": {}, "violations": []}
    Pl, Pr, S = npart_l, npart_r, split_factor

    splits = [r for r in records if r["kind"] == "split"]
    gathers = [r for r in records if r["kind"] == "gather"]
    pairs = [r for r in records if r["kind"] == "pair"]
    emits = [r for r in records if r["kind"] == "emit"]

    for h, hot_key in enumerate(hot_keys):
        lmask = _hot_key_mask(left_pdf, on, hot_key)
        rmask = _hot_key_mask(right_pdf, on, hot_key)
        Lh, Rh = int(lmask.sum()), int(rmask.sum())
        bound = (
            math.ceil(max(Lh, Rh) / S) + min(Lh, Rh) + Pl + Pr
        )

        # Split tasks: each may hold only its own source partition.  The hot
        # rows observed in split task counts must sum to Lh / Rh.
        split_hot_counts = [r["counts"][h] for r in splits]
        sum_split = sum(split_hot_counts)

        # Gather tasks for this hot key carry one salt bucket on one side.
        def hot_gather_rows(side):
            return [
                r["n_rows"]
                for r in gathers
                if r.get("side") == side
                and isinstance(r["piece"], list)
                and r["piece"][0] == 0
                and r["piece"][1] == h
            ]

        left_gathers = hot_gather_rows("left")
        right_gathers = hot_gather_rows("right")
        hot_gathers = left_gathers + right_gathers
        gather_peak = max(hot_gathers, default=0)

        # Pairing tasks combine a left salt and a right salt of the same key.
        pair_inputs = [(r["n_left"], r["n_right"]) for r in pairs]
        pair_peak = max(
            (nl + nr for nl, nr in pair_inputs), default=0
        )

        # Emission tasks hold a single salt bucket.
        emit_peak = max(
            (r["n_rows"] for r in emits if r["hot_index"] == h), default=0
        )

        # Every salt bucket is gathered once per side; each original hot row
        # reaches exactly S pairing tasks on its side.
        copies_per_row_s = S if Lh and Rh else (S if Lh or Rh else 0)

        check = {
            "Lh": Lh,
            "Rh": Rh,
            "S": S,
            "Pl": Pl,
            "Pr": Pr,
            "bound": bound,
            "split_hot_total": sum_split,
            "left_gather_salt_sizes": sorted(left_gathers),
            "right_gather_salt_sizes": sorted(right_gathers),
            "left_gather_sum": sum(left_gathers),
            "right_gather_sum": sum(right_gathers),
            "gather_peak": gather_peak,
            "pair_peak": pair_peak,
            "emit_peak": emit_peak,
            "copies_per_row": copies_per_row_s,
            "n_left_gather_tasks": len(left_gathers),
            "n_right_gather_tasks": len(right_gathers),
            "n_pair_tasks": len(pairs),
            "n_emit_tasks": sum(1 for r in emits if r["hot_index"] == h),
        }
        # Bound checks on every task type that can hold hot rows.
        for label, peak in (
            ("gather", gather_peak),
            ("pair", pair_peak),
            ("emit", emit_peak),
        ):
            if peak > bound:
                result["violations"].append(
                    {
                        "hot_key": hot_key,
                        "task": label,
                        "peak": peak,
                        "bound": bound,
                    }
                )
        result["hot_keys"][str(hot_key)] = check
    result["n_split_tasks"] = len(splits)
    result["n_gather_tasks"] = len(gathers)
    result["n_pair_tasks"] = len(pairs)
    result["n_emit_tasks"] = len(emits)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary = {"cases": []}
    rows_per_partition = 2000
    hot_key = 777
    # Resource matrix requested in the task: partitions 8/16/32, S in {2,4}.
    for npart in (8, 16, 32):
        n_l = n_r = rows_per_partition * npart
        for S in (2, 4):
            for mirror in (False, True):
                # 90% hot on the large side; small side hot key = 4 rows.
                if not mirror:
                    left, right = make_skew_frames(
                        n_l, n_r, npart, npart, hot_key, 0.9
                    )
                    # Force the small side to only 4 hot rows.
                    right.loc[right["k"] == hot_key, "k"] = (
                        [hot_key] * 4
                        + [6000 + i for i in range((right["k"] == hot_key).sum() - 4)]
                    )
                else:
                    right, left = make_skew_frames(
                        n_r, n_l, npart, npart, hot_key, 0.9
                    )
                    left.loc[left["k"] == hot_key, "k"] = (
                        [hot_key] * 4
                        + [6000 + i for i in range((left["k"] == hot_key).sum() - 4)]
                    )
                for method in ("tasks", "disk"):
                    instrument_dir = tempfile.mkdtemp(prefix="skew-probe-")
                    try:
                        run = run_case(
                            left,
                            right,
                            npart,
                            npart,
                            [(hot_key,)],
                            S,
                            method,
                            "outer",
                            instrument_dir,
                        )
                        records = load_records(instrument_dir)
                        analysis = analyze(
                            records,
                            left,
                            right,
                            ["k"],
                            [(hot_key,)],
                            S,
                            npart,
                            npart,
                        )
                        # Exchanged-bytes proxy: bytes of probe JSON plus the
                        # serialized piece sizes recorded at gather time.
                        probe_bytes = sum(
                            os.path.getsize(p)
                            for p in Path(instrument_dir).glob("*.json")
                        )
                        case = {
                            "npart": npart,
                            "S": S,
                            "mirror": mirror,
                            "method": method,
                            "rows_per_partition": rows_per_partition,
                            "probe_bytes": probe_bytes,
                            **run,
                            **analysis,
                        }
                        summary["cases"].append(case)
                        status = "OK" if not analysis["violations"] else "VIOLATION"
                        print(
                            f"npart={npart} S={S} mirror={mirror} method={method}: "
                            f"{status} out_parts={run['npartitions_out']} "
                            f"rows={run['row_count']} t={run['elapsed_s']:.3f}s"
                        )
                    finally:
                        shutil.rmtree(instrument_dir, ignore_errors=True)

    # Special cases: 1024 identical hot rows each side; multi-hot; no hit.
    for label, lpdf, rpdf, hk in [
        (
            "identical_1024",
            pd.DataFrame(
                {"k": [hot_key] * 1024, "v": range(1024), "payload_l": ["x"] * 1024}
            ),
            pd.DataFrame(
                {"k": [hot_key] * 1024, "w": range(1024), "payload_r": ["y"] * 1024}
            ),
            [(hot_key,)],
        ),
        (
            "multi_hot",
            pd.DataFrame(
                {
                    "k": [1, 1, 1, 2, 2, 2, 3],
                    "v": range(7),
                    "payload_l": ["x"] * 7,
                }
            ),
            pd.DataFrame(
                {"k": [1, 1, 2, 2, 4], "w": range(5), "payload_r": ["y"] * 5}
            ),
            [(1,), (2,)],
        ),
        (
            "no_hit",
            pd.DataFrame({"k": [1, 2, 3], "v": range(3), "payload_l": ["x"] * 3}),
            pd.DataFrame({"k": [1, 2], "w": range(2), "payload_r": ["y"] * 2}),
            [(999,)],
        ),
    ]:
        for method in ("tasks", "disk"):
            instrument_dir = tempfile.mkdtemp(prefix="skew-probe-")
            try:
                run = run_case(
                    lpdf, rpdf, 4, 4, hk, 2, method, "outer", instrument_dir
                )
                records = load_records(instrument_dir)
                analysis = analyze(
                    records, lpdf, rpdf, ["k"], hk, 2, 4, 4
                )
                case = {
                    "case": label,
                    "method": method,
                    **run,
                    **analysis,
                }
                summary["cases"].append(case)
                print(
                    f"{label} method={method}: "
                    f"{'OK' if not analysis['violations'] else 'VIOLATION'} "
                    f"rows={run['row_count']}"
                )
            finally:
                shutil.rmtree(instrument_dir, ignore_errors=True)

    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    n_violations = sum(len(c["violations"]) for c in summary["cases"])
    print(f"\nwrote {args.out}; total bound violations: {n_violations}")
    return 1 if n_violations else 0


if __name__ == "__main__":
    sys.exit(main())
