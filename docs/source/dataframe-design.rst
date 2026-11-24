.. _dataframe.design:

Dask DataFrame Design
=====================

Dask DataFrames coordinate many Pandas DataFrames/Series arranged along an
index.  We define a Dask DataFrame object with the following components:

- A Dask graph with a special set of keys designating partitions, such as
  ``('x', 0), ('x', 1), ...``
- A name to identify which keys in the Dask graph refer to this DataFrame, such
  as ``'x'``
- An empty Pandas object containing appropriate metadata (e.g.  column names,
  dtypes, etc.)
- A sequence of partition boundaries along the index called ``divisions``

Metadata
--------

Many DataFrame operations rely on knowing the name and dtype of columns.  To
keep track of this information, all Dask DataFrame objects have a ``_meta``
attribute which contains an empty Pandas object with the same dtypes and names.
For example:

.. code-block:: python

   >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
   >>> ddf = dd.from_pandas(df, npartitions=2)
   >>> ddf._meta
   Empty DataFrame
   Columns: [a, b]
   Index: []
   >>> ddf._meta.dtypes
   a     int64
   b    object
   dtype: object

Internally, Dask DataFrame does its best to propagate this information
through all operations, so most of the time a user shouldn't have to worry
about this.  Usually this is done by evaluating the operation on a small sample
of fake data, which can be found on the ``_meta_nonempty`` attribute:

.. code-block:: python

   >>> ddf._meta_nonempty
      a    b
   0  1  foo
   1  1  foo

Sometimes this operation may fail in user defined functions (e.g. when using
``DataFrame.apply``), or may be prohibitively expensive.  For these cases, many
functions support an optional ``meta`` keyword, which allows specifying the
metadata directly, avoiding the inference step.  For convenience, this supports
several options:

1. A Pandas object with appropriate dtypes and names.  If not empty, an empty
   slice will be taken:

.. code-block:: python

  >>> ddf.map_partitions(foo, meta=pd.DataFrame({'a': [1], 'b': [2]}))

2. A description of the appropriate names and dtypes.  This can take several forms:

* A ``dict`` of ``{name: dtype}`` or an iterable of ``(name, dtype)``
  specifies a DataFrame. Note that order is important: the order of the
  names in ``meta`` should match the order of the columns
* A tuple of ``(name, dtype)`` specifies a series

This keyword is available on all functions/methods that take user provided
callables (e.g. ``DataFrame.map_partitions``, ``DataFrame.apply``, etc...), as
well as many creation functions (e.g. ``dd.from_delayed``).


.. _dataframe-design-partitions:

Partitions
----------

Internally, a Dask DataFrame is split into many partitions, where each partition
is one Pandas DataFrame.  These DataFrames are split vertically along the
index.  When our index is sorted and we know the values of the divisions of our
partitions, then we can be clever and efficient with expensive algorithms (e.g.
groupby's, joins, etc...).

For example, if we have a time-series index, then our partitions might be
divided by month: all of January will live in one partition while all of
February will live in the next.  In these cases, operations like ``loc``,
``groupby``, and ``join/merge`` along the index can be *much* more efficient
than would otherwise be possible in parallel.  You can view the number of
partitions and divisions of your DataFrame with the following fields:

.. code-block:: python

   >>> df.npartitions
   4
   >>> df.divisions
   ['2015-01-01', '2015-02-01', '2015-03-01', '2015-04-01', '2015-04-31']

The number of partitions and the division values might change during optimization.
The optimizer will try to create partitions with a sensible size to avoid straining
the scheduler with many small partitions.

Divisions includes the minimum value of every partition's index and the maximum
value of the last partition's index.  In the example above, if the user searches
for a specific datetime range, then we know which partitions we need to inspect
and which we can drop:

.. code-block:: python

   >>> df.loc['2015-01-20': '2015-02-10']  # Must inspect first two partitions

Often we do not have such information about our partitions.  When reading CSV
files, for example, we do not know, without extra user input, how the data is
divided.  In this case ``.divisions`` will be all ``None``:

.. code-block:: python

   >>> df.divisions
   [None, None, None, None, None]

In these cases, any operation that requires a cleanly partitioned DataFrame with
known divisions will have to perform a sort.  This can generally achieved by
calling ``df.set_index(...)``.


.. _dataframe-design-composite-index:

Two-level composite indexes
---------------------------

``df.set_index(["account", "seq"])`` builds an ordered two-level
``MultiIndex`` from two existing columns.  The complete key is ordered
lexicographically by component (never by a string concatenation), so a row
range such as ``df.loc[slice((a, b), (c, d))]`` maps directly onto a
contiguous run of partitions and only the partitions whose boundary can
intersect the range are read.

**Divisions.**  For a two-level index, ``divisions`` is a sequence of
complete ``(account, seq)`` pairs.  Partition *i* covers
``[d[i], d[i+1])`` and the last partition also includes ``d[-1]``.  The last
two entries may be equal, which represents a trailing run of duplicate
complete keys without splitting it; a complete key never straddles a
boundary.  Out-of-range data under explicit divisions raises rather than
being dropped or clamped.

**Planning cost.**  With explicit ``divisions`` the expression, its
``_meta`` and ``optimize()`` never read the source.  Without ``divisions``
the first access to ``.divisions`` (or ``optimize`` / ``persist`` /
``compute``) runs a small summary graph on the client: each input partition
uploads at most 4096 raw key tuples plus one low/high endpoint and a row
count.  Planner state therefore grows with the number of input partitions
and that fixed per-partition allowance, never with the total row count or
the number of distinct keys.  The chosen boundaries are real data keys, so
the published divisions always describe the actual content.

**Duplicate-heavy keys.**  When a single complete key holds most of the
rows it cannot be split across partitions without breaking the "one key per
partition" guarantee.  The algorithm then produces fewer, unevenly sized
partitions rather than an even split; this is reported honestly by the
per-partition row counts (see ``scratch/run_composite_index_experiments.py``).

**Construction trade-off.**  Two organisations were considered.  (1) Cast
each row to a single sortable *composite string* and reuse the one-column
path.  This is simple and reuses existing quantile machinery, but it is
ambiguous at component boundaries (``"a"+"bc" == "ab"+"c"``), cannot keep
``int > 2**53`` exact, and forces every key through string formatting.  (2)
Keep the two components separate and compare them as tuples (what this
implementation does).  It preserves exact integer and Unicode ordering,
makes boundaries unambiguous, and lets the partition-id computation stay
vectorised in NumPy (``searchsorted`` on the first level plus a per-bucket
``searchsorted`` on the second).  The cost is a modest per-partition
normalisation pass to validate the two arrays and a slightly more involved
cut-point selection over the sampled keys; both are linear in the number of
rows read and do not require materialising a full key set.


.. _dataframe-design-groupby:

Groupby
-------

By default, groupby will choose the number of output partitions based on a few
different factors. It will look at the number of grouping keys to guess the
cardinality of your data. It will use this information to calculate a factor
based on the number of input partitions. You can override this behavior by
specifying the number of output partitions using the `split_out` argument.

.. code-block:: python

   result = df.groupby('id').value.mean()
   result.npartitions  # returns 1

   result = df.groupby(['id', 'id2']).value.mean()
   result.npartitions  # returns 5

   result = df.groupby('id').value.mean(split_out=8)
   result.npartitions  # returns 8

Some groupby aggregation functions have a different `split_out` default value.
`split_out=True` will keep the number of partitions constant, which is useful
for operations that either don't reduce the number of rows very much.

.. code-block:: python

   result = df.groupby('id').value.nunique()
   result.npartitions  # returns same as df.npartitions
