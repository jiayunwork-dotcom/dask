from __future__ import annotations

import datetime

from dask.dataframe import methods

CombinedOutput = type("CombinedOutput", (tuple,), {})


def overlap_chunk(func, before, after, *args, **kwargs):
    dfs = [df for df in args if isinstance(df, CombinedOutput)]
    combined, prev_part_length, next_part_length = dfs[0]

    args = [arg[0] if isinstance(arg, CombinedOutput) else arg for arg in args]

    out = func(*args, **kwargs)

    expansion = None
    if combined.shape[0] != 0:
        expansion = out.shape[0] // combined.shape[0]

    # Trim the context rows that were *actually* added to this partition.
    # This is not necessarily the requested ``before``/``after``: at the
    # global boundaries, and when (empty) partitions provide fewer rows
    # than requested, only the rows that are really present may be
    # dropped so that no genuine rows of this partition are lost.
    head = prev_part_length or 0
    tail = next_part_length or 0
    if expansion:
        head *= expansion
        tail *= expansion
    if tail:
        return out.iloc[head:-tail]
    return out.iloc[head:]


def _tail_with_context(prev, current, before, grouper):
    """Last ``before`` rows of ``prev`` followed by ``current``.

    When ``grouper`` is a dict of ``pandas.groupby`` keyword arguments, the
    last ``before`` rows *per group* are returned instead.  Rows are kept
    in their original (globally ordered) positions.
    """
    if prev is not None:
        current = methods.concat([prev, current])
    if grouper is None:
        return current.tail(before)
    return current.groupby(**grouper).tail(before)


def _head_with_context(current, next_, after, grouper):
    """First ``after`` rows of ``current`` followed by ``next_``.

    When ``grouper`` is a dict of ``pandas.groupby`` keyword arguments, the
    first ``after`` rows *per group* are returned instead.  Rows are kept
    in their original (globally ordered) positions.
    """
    if next_ is not None:
        current = methods.concat([current, next_])
    if grouper is None:
        return current.head(after)
    return current.groupby(**grouper).head(after)


def _head_timedelta(current, next_, after):
    """Return rows of ``next_`` whose index is before the last
    observation in ``current`` + ``after``.

    Parameters
    ----------
    current : DataFrame
    next_ : DataFrame
    after : timedelta

    Returns
    -------
    overlapped : DataFrame
    """
    return next_[next_.index < (current.index.max() + after)]
