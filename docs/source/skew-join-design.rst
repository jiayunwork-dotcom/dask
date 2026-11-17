# Skew-aware hash join design and cost notes

This document explains the data-movement plan behind the ``skew_join`` option
of ``DataFrame.merge`` / ``dask.dataframe.merge`` and the reasoning behind the
design choices.  It complements the user-facing description in
``docs/source/dataframe-joins.rst``.

## 1. Problem

A normal hash shuffle assigns every row a partition via
``hash(join_key) % npartitions``.  All rows with the same composite key value
land in a single partition, no matter how many partitions exist.  When one key
value (a *hot key*) accounts for a large share of the rows, that partition
becomes a memory and compute hotspot and the join cannot scale.

The caller knows the hot key values up front and provides them as
``hot_keys`` together with a ``split_factor`` ``S``.  No data inspection is
allowed while the graph is constructed.

## 2. Static plan (no source consumption)

For each side, every **input** partition runs one local split task
(``skew_split_frame``).  It classifies each row into one of disjoint groups:

* a hot row of hot key ``h`` -> one of ``S`` salt buckets ``(h, s)``,
* a non-hot row -> one of ``npartitions`` ordinary hash buckets ``p``.

Salt assignment is a local, deterministic round-robin over the matching rows
of each input partition (``row_number_within_partition % S``).  This is the key
to the bound: even when every hot-key row is byte-identical, a whole-row hash
would collapse them onto one bucket, whereas the position-based round-robin
spreads them evenly.  No side is ever gathered by key before splitting, so the
"single source partition read" exception in the requirements is the only place
where a task can hold a full input partition.

The split task also emits a tiny count vector ``(count_h0, count_h1, ...)`` --
integer control information only, never original rows.

## 3. Exchange and task assignment

### Non-hot rows

Non-hot rows keep the ordinary hash path: pieces ``(1, p)`` from all input
partitions of a side are gathered for partition ``p`` and joined with the
opposite side's partition ``p`` using the existing full-semantics
``merge_chunk``.  Because hot and non-hot rows are disjoint, this path behaves
exactly like the current hash join for every non-hot key.

### Hot rows: the S x S grid

For hot key ``h``, salt bucket ``s`` of the left side is gathered once and salt
bucket ``t`` of the right side is gathered once.  An inner-pairing task exists
for every ``(s, t)`` cell (``S * S`` cells per hot key).  Each cell calls
``merge_chunk(..., how="inner")`` and only emits matched pairs.

Every qualified left-right record pair is produced exactly once: a left row in
bucket ``s`` and a right row in bucket ``t`` can only meet in cell ``(s, t)``,
and they do meet there.  Salt buckets partition each side's hot rows, so no
pair is duplicated or missed, independent of the output partition boundaries.

### Outer-join unmatched rows

A hot key present on both sides matches every row somewhere in the grid, so no
unmatched emission is needed for it.  A hot key present on only one side has no
matches; the aggregated count vector of the opposite side answers the
"presence" question without moving any rows.  When the opposite count is zero,
each salt bucket of the present side emits its rows as unmatched output rows
(``left_only`` / ``right_only`` with the indicator set accordingly).  Emitting
per salt bucket means each original row appears exactly once.

## 4. Why a symmetric grid instead of split-and-replicate

An asymmetric alternative splits only the big side and replicates the small
side's hot rows to every big-side bucket.  It uses fewer tasks, but choosing
the replication factor safely requires knowing ``Lh`` and ``Rh``, which are
only available after reading the sources.  The symmetric ``S x S`` grid needs
only the statically known ``S`` and guarantees the same bound for any ratio
``Lh / Rh``, including the mirrored case.

## 5. Per-task input bound

For hot key ``h`` let ``Lh``, ``Rh`` be the row counts on the two sides,
``S`` the split factor and ``Pl``, ``Pr`` the input partition counts.

* Each salt bucket holds at most ``ceil(Lh / S)`` left rows or
  ``ceil(Rh / S)`` right rows (round-robin differs by at most one row).
* A pairing cell holds one left and one right bucket, i.e. at most
  ``ceil(Lh / S) + ceil(Rh / S)`` original rows.
* A gather or emission task holds a single bucket.

All of these are bounded by the required expression

  ``ceil(max(Lh, Rh) / S) + min(Lh, Rh) + Pl + Pr``

which is strictly looser than the actual cell peak (the ``Pl + Pr`` slack
covers the per-input-partition pieces gathered by an exchange task).  Each
original hot row is handed to exactly ``S`` pairing tasks on its side (once
per opposite salt), never more.

## 6. Output partition layout

With ``n`` normal partitions, ``H`` hot keys and split factor ``S``, the result
has a finite, deterministic number of partitions:

  ``n + H * S * S + 2 * H * S``

* ``n`` ordinary hash partitions,
* ``H * S * S`` inner-pairing cells,
* ``2 * H * S`` unmatched-row emission tasks (empty for inner joins or when the
  key exists on both sides, but still materialized so the layout is stable).

``npartitions=`` therefore continues to control only the ordinary partitions;
hot-key work adds the bounded extra count above.

## 7. Outer-join correctness details

* Pairing cells are strictly inner.  Keeping unmatched handling out of the
  grid avoids emitting one row once per grid row/column.
* Missing key components match missing values, mirroring pandas
  (``NaN == NaN`` / ``NA == NA`` on join keys).  Hot-key membership tests use
  ``Series.isna()`` for ``None`` components and NA-safe equality otherwise.
* Unmatched output frames are assembled against the join result metadata and
  every output partition is cast to the metadata dtypes, so categorical
  upcasting and integer-to-float NA behavior match pandas 2.2.3 exactly.

## 8. Shuffle methods and cleanup

* ``shuffle_method='tasks'``: split tasks return piece dictionaries; gather
  tasks concatenate the matching pieces across input partitions using Dask's
  empty-filtering ``_concat`` helper.
* ``shuffle_method='disk'``: pieces are appended into two partd stores (one per
  side) behind barriers, then collected per piece.  A reference-counted partd
  wrapper removes each temporary directory when its last collection finishes,
  and ``ensure_cleanup_on_exception`` covers failures during appends.

P2P is not supported (it is built around a single hash column) and is rejected
at graph construction, along with index joins, ``leftsemi`` and explicit
``broadcast=True``.

## 9. Downstream optimization contract

The skew output deliberately advertises no partition mapping for the join key
(``unique_partition_mapping_columns_from_shuffle`` is empty).  Downstream
same-key groupby, ``drop_duplicates`` and merge operations therefore schedule
the shuffles they need instead of assuming co-location.  Ordinary joins and
every other proven partition mapping are untouched, so the optimization is
scoped to this one operator rather than disabled globally.

## 10. Configuration robustness

A listed hot key that does not occur, or an unlisted key that turns out to be
hot, can only change execution cost: the hash/grid partition of every row is a
function of its actual key value and the fixed plan, so the multiset of output
rows is identical to an ordinary pandas merge.  Configurations are normalized
and deduplicated at the public API entry point (NumPy integers -> Python ints),
which makes equivalent configurations produce identical task keys and keeps
distinct configurations isolated.
