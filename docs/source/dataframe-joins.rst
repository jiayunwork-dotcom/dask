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

Skewed Joins (Known Hot Keys)
-----------------------------

A normal hash shuffle sends every row with the same join key to one output
partition.  When a single key value -- a *hot key* -- accounts for most of the
rows, adding more partitions does not help: that one partition still receives
the entire hot-key group and becomes a memory and compute bottleneck.

If the hot key values are known ahead of time, ``DataFrame.merge`` and
``dask.dataframe.merge`` accept a ``skew_join`` mapping that disperses them:

.. code-block:: python

    result = left.merge(
        right,
        on="k",
        how="outer",
        skew_join={"hot_keys": [(777,)], "split_factor": 4},
    )

``hot_keys`` is a non-empty list of tuples, one tuple per hot key with one
component per join column (``None`` denotes a missing component).  Rows
carrying a hot key are split into ``split_factor`` (2-16) *salt buckets* on
both sides and paired on a salt-by-salt grid, while all other rows keep using
the ordinary hash partitions.  Hot-key records are never dropped or
deduplicated, so the result is identical to an ordinary pandas merge,
including full many-to-many pairings and outer-join unmatched rows.

Cost and partition accounting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let ``npartitions`` be the target partition count for non-hot keys, ``H`` the
number of hot keys and ``S`` the split factor.  The result contains

``npartitions + H * S * S + 2 * H * S``

partitions: the ordinary hash partitions, one inner-pairing cell per hot key
and salt pair, and the outer-join unmatched-row outputs.  For a hot key with
``Lh`` left rows and ``Rh`` right rows and ``Pl``/``Pr`` input partitions, no
task that pairs rows or emits unmatched rows for that key holds more than

``ceil(max(Lh, Rh) / S) + min(Lh, Rh) + Pl + Pr``

original rows (and their gathered copies).  Each original hot row is handed to
at most ``S`` pairing tasks.  The bound holds regardless of the ``Lh / Rh``
ratio and even when all hot-key rows are byte-identical, because salt
assignment is a local round-robin over matching rows rather than a whole-row
hash.

The plan is static: it never inspects the data, so a hot key that is absent
from the data, or a key that was not listed, only changes execution cost, never
the result.  Because the output no longer guarantees that equal keys are
co-located, downstream same-key groupby, ``drop_duplicates`` and merge
operations schedule their own shuffles as needed; ordinary joins and other
valid partition mappings keep their existing optimizations.

Only column-based ``inner``, ``left``, ``right`` and ``outer`` joins are
supported, with the ``'tasks'`` and ``'disk'`` shuffle methods.  Index joins,
``leftsemi``, explicit ``broadcast=True`` and ``shuffle_method='p2p'`` raise a
``ValueError`` while the graph is being constructed, before any source task
runs.

