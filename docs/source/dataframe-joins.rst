Joins
=====

DataFrame joins are a common and expensive computation that benefit from a
variety of optimizations in different situations.  Understanding how your data
is laid out and what you're trying to accomplish can have a large impact on
performance.  This documentation page goes through the various different
options and their performance impacts.

Large to Large Unsorted Joins
-----------------------------

In the worst case scenario you have two large tables with many partitions each
and you want to join them both along a column that may not be sorted.

This can be slow.  In this case Dask DataFrame will need to move all of your
data around so that rows with matching values in the joining columns are in the
same partition.  This large-scale movement can create communication costs, and
can require a large amount of memory.  If enough memory can not be found then
Dask will have to read and write data to disk, which may cause other
performance costs.

These problems are solvable, but will be significantly slower than many other
operations.  They are best avoided if possible.

Large to Small Joins
--------------------

Many join or merge computations combine a large table with one small one.  If
the small table is either a single partition Dask DataFrame or even just a
normal Pandas DataFrame then the computation can proceed in an embarrassingly
parallel way, where each partition of the large DataFrame is joined against the
single small table.  This incurs almost no overhead relative to Pandas joins.

If your smaller table can easily fit in memory, then you might want to ensure
that it is a single partition with the repartition method.

.. code-block:: python

    import dask
    large = dask.datasets.timeseries(freq="10s", npartitions=10)
    small = dask.datasets.timeseries(freq="1D", dtypes={"z": int})

    small = small.repartition(npartitions=1)
    result = large.merge(small, how="left", on=["timestamp"])

Sorted Joins
------------

The Pandas merge API supports the ``left_index=`` and ``right_index=`` options
to perform joins on the index.  For Dask DataFrames these keyword options hold
special significance if the index has known divisions
(see :ref:`dataframe-design-partitions`).
In this case the DataFrame partitions are aligned along these divisions (which
is generally fast) and then an embarrassingly parallel Pandas join happens
across partition pairs.  This is generally relatively fast.

Sorted or indexed joins are a good solution to the large-large join problem.
If you plan to join against a dataset repeatedly then it may be worthwhile to
set the index ahead of time, and possibly store the data in a format that
maintains that index, like Parquet.

.. code-block:: python

    import dask
    import dask.dataframe as dd

    left = dask.datasets.timeseries(dtypes={"foo": int})

    # timeseries returns a dataframe indexed by
    # timestamp, we don't need to set_index.

    # left.set_index("timestamp")

    left.to_parquet("left", overwrite=True)
    left = dd.read_parquet("left")

    right_one = dask.datasets.timeseries(dtypes={"bar": int})
    right_two = dask.datasets.timeseries(dtypes={"baz": int})

    result = left.merge(
        right_one, how="left", left_index=True, right_index=True)
    result = result.merge(
        right_two, how="left", left_index=True, right_index=True)

Cardinality Validation (``validate``)
-------------------------------------

Like pandas, :meth:`dask.dataframe.DataFrame.merge` and :func:`dask.dataframe.merge`
accept a ``validate`` argument declaring the expected cardinality of the join
keys:

- ``"one_to_one"`` / ``"1:1"``: the join keys must be unique in both datasets
- ``"one_to_many"`` / ``"1:m"``: the join keys must be unique in the left dataset
- ``"many_to_one"`` / ``"m:1"``: the join keys must be unique in the right dataset
- ``"many_to_many"`` / ``"m:m"`` and ``None``: no check

.. code-block:: python

    # Raises pandas.errors.MergeError at compute time if the keys of either
    # side are duplicated, including duplicates that never match a row on the
    # other side.
    orders.merge(customers, on="customer_id", validate="many_to_one")

The semantics match pandas 2.2.3 exactly: key equality (including the fact
that ``NaN``/``NA`` values compare equal) and the
:class:`pandas.errors.MergeError` messages are identical, so the same code can
be validated against pandas on small samples. An invalid ``validate`` string,
or combining a non-default mode with ``how="leftsemi"``, raises ``ValueError``
while the expression is being built — no source data is read at that point.
The actual data check only runs when the result is computed.

Global guarantees and the optimization barrier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The check is defined over the *full inputs* of the merge as they were when
``merge`` was called. Filtering a dataframe before the merge means the
filtered (and therefore potentially still unique) input is what gets checked;
filtering or projecting the merge *result* can never remove the check, even if
every offending row would have been filtered out, and even if you only compute
a single output partition or drop the join-key columns afterwards. This is
implemented as an optimization barrier placed above the merge, so query
optimization cannot fuse, push down or prune the validation away.

Reusing existing uniqueness evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dask tracks which column groups are *globally unique* as expressions are
transformed. A ``drop_duplicates(subset=K, keep="first"/"last")`` establishes
that ``K`` is unique, and that evidence survives:

- projections that keep every column of ``K`` (a larger unique key set does
  **not** prove that any subset is unique),
- pure column renames (including reordering the columns of a composite key),
- row filters and ``dropna``,
- repartitions and shuffles.

It is deliberately invalidated by operations that can change key values or
mix rows, such as overwriting a key column, ``concat`` of two individually
unique tables, or arbitrary ``map_partitions``. Merely knowing the divisions
or that equal keys live in the same partition never counts as a uniqueness
guarantee.

When a checked merge side already carries evidence for its join key, no extra
global uniqueness scan, dedup or data redistribution is planned. In
particular, calling ``drop_duplicates`` again on an already-unique key reuses
the existing plan instead of running a second global reduction, and a
validated one-to-one inner/left/outer merge on a shared, type-stable column
key produces a result whose key is itself known unique, so a later
``drop_duplicates`` on that key is free.

Cost and bounded resource use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The check is distributed and bounded in memory. Each input partition emits a
small local summary of its *distinct* join keys with their multiplicities;
only these summaries are hash-partitioned by the key, so equal keys (including
hot keys) meet on one check task while the raw rows stay on their original
partitions. A per-bucket sum flags any key whose global multiplicity exceeds
one, and a constant-fan-in tree reduction combines the boolean flags into one
scalar that every merge output partition depends on. Consequently no single
new task ever holds all distinct keys of a side, let alone all of its rows,
and the validation state does not grow with the number of join output rows.
It works with the ``tasks``, ``disk`` and ``p2p`` shuffle methods, broadcast
and index-aligned joins, and with the synchronous, threaded and distributed
schedulers. The merge's own shuffle is still required; when a checked side is
already correctly partitioned the validation summary reuses that layout and
adds no further data movement beyond the small summaries.
