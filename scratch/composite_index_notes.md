# Two-level composite `set_index` — implementation notes

This document describes the composite (account, seq) ordered index added to
Dask's `dask.dataframe` query planner. It complements the user-facing text in
`docs/source/dataframe-design.rst` (section *Two-level composite indexes*) and
the `DataFrame.set_index` docstring.

## Files

| File | Role |
| --- | --- |
| `dask/dataframe/dask_expr/_multi_index.py` | Planning-only helpers: argument validation, division normalisation, per-partition sampling, vectorised key→partition assignment, sampling/cut-point selection. No task graph. |
| `dask/dataframe/dask_expr/_multi_index_expr.py` | Expressions: `SetIndexComposite`, `CompositePartitionId`, `CompositeSetIndexPost`, `CompositeDiskShuffle`, `CompositeEmptyIndex`. |
| `dask/dataframe/dask_expr/_collection.py` | `DataFrame.set_index` dispatch to `_set_index_composite`; `from_pandas` pre-indexed validation. |
| `dask/dataframe/dask_expr/_indexing.py` | Composite-aware `loc` (`_composite_getitem`, `LocCompositeElement`, `LocSlice` projection fix). |
| `dask/dataframe/dask_expr/io/io.py` | `FromPandas` keeps unknown divisions for a `sort=False` two-level index. |
| `dask/dataframe/methods.py` | `try_loc` positional masking for unsorted composite partitions (unknown divisions). |
| `dask/dataframe/dask_expr/_expr.py` | `_check_divisions` compares complete composite keys. |
| `dask/dataframe/dask_expr/tests/test_multi_index.py` | New tests. |

## Public API

```python
ddf.set_index(["account", "seq"],
              drop=True,
              npartitions=8,            # None -> reuse input count
              divisions=None,           # [(a0,b0), (a1,b1), ...]
              shuffle_method="tasks")   # or "disk"
```

* Result index: pandas `MultiIndex` with level names `["account", "seq"]`.
* Ordering: lexicographic by component; `int64` exact beyond `2**53`; strings
  by Unicode value; components never concatenated.
* `dd.from_pandas(pre_indexed, npartitions=..., sort=True|False)` accepts a
  pre-built two-level `MultiIndex` in the same range.

### Supported range

* Exactly two distinct column names; each an `int64` or string column with no
  missing values. Types may differ between the two levels.
* `sort=True` (default), `sorted=False` (default), `append=False` (default).
* `shuffle_method` in `{"tasks", "disk"}`.
* `npartitions`: `None` or a positive non-boolean integer.
* Explicit `divisions`: tuple-of-tuples, strictly increasing except the final
  two entries may be equal.

Everything outside this range (other dtypes, mixed/missing values,
`sorted=True`, `sort=False`, `append=True`, `p2p`, `npartitions` of the wrong
type, non-increasing / wrong-width / mismatched-count divisions, out-of-range
data) raises `ValueError` (or `KeyError` for a missing column). Errors decidable
from metadata/configuration are raised **before any source read**; value-level
errors are raised when the offending partition is first inspected.

## Boundary representation

`divisions` is a tuple of `(level0, level1)` pairs; partition *i* covers
`[d[i], d[i+1])`, the last partition includes `d[-1]`. The last two may be equal
(trailing duplicate run). A complete key never straddles a boundary. Empty
input yields unknown divisions `(None, None)` while preserving level names and
column structure.

## Planning cost and when the automatic summary runs

* **Explicit divisions:** building the expression, `_meta`, `divisions`,
  `optimize()` and `npartitions` never touch the source (verified by a
  counting source in the tests).
* **Automatic divisions:** construction only computes metadata. The first
  access to `.divisions` — which also happens on `optimize()` / `persist()` /
  `compute()` — runs a small summary graph **on the client**: each input
  partition uploads at most `COMPOSITE_SAMPLE_LIMIT = 4096` raw key tuples plus
  its low endpoint, high endpoint and row count. The planner's own state is
  `O(input_partitions * (4096 + 2))`, independent of total rows or the number
  of distinct keys. Boundaries chosen are real sampled keys (or the exact data
  endpoints), so published divisions always describe the actual content.
  Planner results are cached in a small LRU keyed by expression name and
  target, so repeated `.divisions` accesses do not re-read.
* The actual exchange/sort is deferred to normal graph evaluation. No task in
  the graph calls `compute`/`persist` or starts a scheduler; verified by
  scanning the materialized graph for such callables.

## Composition of the exchange

`SetIndexComposite._lower`:

1. `CompositePartitionId` — validates the data range (explicit divisions) and
   assigns each row an integer target-partition id. The id is fully vectorised:
   `searchsorted` on the level-0 boundaries plus a per-bucket `searchsorted` on
   the level-1 boundaries (object/int arrays chosen by level kind).
2. `Repartition` down to the target count when needed.
3. `TaskShuffle` or `CompositeDiskShuffle` on the reserved `_partitions`
   column (`shuffle_group` treats that name as an explicit id).
4. `CompositeSetIndexPost` sets the named `MultiIndex` and sorts within each
   partition; `SortIndexBlockwise` is the final ordering pass.
5. Empty input short-circuits to `CompositeEmptyIndex`.

`CompositeDiskShuffle` wraps partd in a reference-counted wrapper (adapted
from the skew-join path) so the temporary directory is removed deterministically
once the last materialising `collect` runs — on success and on source failure.

## `loc`

`indexed.loc[(a, b), :]` → all rows for one complete key; a missing key raises
`KeyError`. `indexed.loc[slice((a, b), (c, d)), columns]` → inclusive complete-key
range; an empty range returns an empty frame with the same structure. The
second tuple component of the row key is never confused with a column name:
a bare two-tuple with no slice is a *row* key; `((a, b), cols)` selects rows then
columns. Range reads prune to the partitions whose `[d[i], d[i+1])` can intersect
the request (proved by a raising, per-partition reader in the tests). Projection
before `loc`, `loc` before projection (fixed `LocSlice` column push-down),
`optimize()` and `persist()` all preserve the semantics.

## Construction trade-off (two organisations)

1. **Single composite string** — convert each row to one sortable string and
   reuse the one-column quantile path. Simple and reuses existing machinery, but
   ambiguous at component boundaries (`"a"+"bc" == "ab"+"c"`), cannot keep
   `int > 2**53` exact, and formats every key.
2. **Separate components compared as tuples (chosen)** — exact integer and
   Unicode ordering, unambiguous boundaries, and the partition-id computation
   stays vectorised. Cost: a per-partition normalisation pass to validate the
   two arrays and cut-point selection over the sampled keys — both linear in
   rows read, no full key set.

## Scale experiment

`scratch/run_composite_index_experiments.py` (results in
`scratch/composite_index_results.json`): 8 and 32 input partitions × 20,000
rows each, 8 target partitions, three distributions, both shuffle methods.
The test side only aggregates per-partition row counts and an independent
payload digest (never a full concatenation).

Headline findings:

* All 12 cases match the expected row count and per-partition payload digest.
* Uploaded keys scale with input partitions (32,752 for 8 partitions;
  131,008 for 32), i.e. bounded by `input_partitions * (4096 + 2)`.
* Planner state (distinct boundary keys) stays constant at 9 regardless of
  row count.
* `uniform` produces even partitions. `skew_first` and `skew_full` are uneven
  because a hot *complete* key (or a hot first level that consumes the same
  second-level cuts) cannot be split without breaking the one-key-per-partition
  guarantee: in `skew_full` the hot partition holds ~90% of the rows. This is
  reported honestly, not hidden.
