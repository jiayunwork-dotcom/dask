"""Helpers for the two-level composite ``(account, seq)`` ordered index.

The module contains the *planning* half of Dask's two-column ``set_index``:
argument validation that only needs metadata, conversions between a
``MultiIndex`` and the tuple-of-tuples ``divisions`` representation, the
per-partition sampling contract, and vectorised key-to-partition assignment.
No task graph is built here.

Ordering
    Keys are ordered lexicographically by component.  ``int64`` levels use
    exact integer ordering (values above ``2**53`` are still exact) and
    string levels use Unicode code-point ordering.  Components are never
    concatenated into a single string, so separators and empty strings stay
    unambiguous.

Divisions
    ``divisions`` is a tuple of ``(level0, level1)`` pairs following the
    normal Dask convention: partition *i* covers ``[d[i], d[i+1])`` and the
    final partition also includes ``d[-1]``.  The last two boundaries may be
    equal, which represents a trailing run of duplicate complete keys
    without splitting it.  A complete key never straddles a boundary.

Automatic partitions
    Every input partition hands the global planner at most
    ``COMPOSITE_SAMPLE_LIMIT`` raw key tuples plus one low endpoint, one high
    endpoint and a row count.  Planner state therefore grows with the input
    partition count and the sample allowance, never with total rows or the
    number of distinct keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Maximum number of raw key tuples one input partition may hand to the
#: global planner during automatic (division-less) planning.
COMPOSITE_SAMPLE_LIMIT = 4096

#: Supported shuffle methods for composite set_index.
SUPPORTED_SHUFFLES = frozenset({"tasks", "disk"})


class CompositeIndexError(ValueError):
    """Invalid composite-index declaration, boundary, or data value."""


def _is_plain_int(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def level_kind(dtype) -> str:
    """Classify a column dtype as ``"int64"`` or ``"string"``.

    Raises :class:`CompositeIndexError` for unsupported dtypes.  This is a
    metadata-only decision and never inspects values.
    """
    if isinstance(dtype, pd.CategoricalDtype):
        raise CompositeIndexError(
            "composite set_index does not support categorical levels; "
            "cast the column to int64 or string first"
        )
    if pd.api.types.is_bool_dtype(dtype):
        raise CompositeIndexError(
            "composite set_index does not support boolean levels; "
            "cast the column to int64 or string first"
        )
    if pd.api.types.is_integer_dtype(dtype):
        if str(dtype) != "int64":
            raise CompositeIndexError(
                "composite set_index requires int64 levels; got integer dtype "
                f"{dtype!r}. Cast the column with astype('int64') first."
            )
        return "int64"
    if (
        pd.api.types.is_float_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
    ):
        raise CompositeIndexError(
            f"composite set_index does not support dtype {dtype!r}; "
            "levels must be int64 or string"
        )
    if pd.api.types.is_string_dtype(dtype) or dtype == object:
        return "string"
    raise CompositeIndexError(
        f"composite set_index does not support dtype {dtype!r}; "
        "levels must be int64 or string"
    )


def index_level_kinds(index) -> list[str] | None:
    """Return the level kinds of a two-level ``MultiIndex`` or ``None``."""
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        return None
    kinds = []
    for level in index.levels:
        try:
            kinds.append(level_kind(level.dtype))
        except CompositeIndexError:
            return None
    return kinds


def validate_key_columns(frame_meta: pd.DataFrame, columns) -> list[str]:
    """Validate the two index column names using metadata only."""
    if len(columns) != 2:
        raise CompositeIndexError(
            "composite set_index requires exactly two column names; "
            f"got {len(columns)}"
        )
    columns = list(columns)
    for col in columns:
        if hasattr(col, "_meta") or isinstance(col, (list, dict, set, tuple)):
            raise CompositeIndexError(
                "composite set_index column declarations must be plain column "
                f"labels, not dask/pandas objects; got {col!r}"
            )
        if not isinstance(col, (str, int)) or isinstance(col, bool):
            raise CompositeIndexError(
                "composite set_index column declarations must be string or "
                f"integer labels; got {col!r}"
            )
    if columns[0] == columns[1]:
        raise CompositeIndexError(
            "composite set_index requires two distinct columns; got "
            f"{columns[0]!r} twice"
        )
    for col in columns:
        if col not in frame_meta.columns:
            raise KeyError(f"None of [{col!r}] are in the [columns]")
        if int((frame_meta.columns == col).sum()) != 1:
            raise CompositeIndexError(
                f"composite set_index column {col!r} is not unique in the frame"
            )
    return [level_kind(frame_meta[col].dtype) for col in columns]


def validate_npartitions(npartitions):
    """``None`` or a positive, non-boolean integer."""
    if npartitions is None:
        return None
    if not _is_plain_int(npartitions):
        raise CompositeIndexError(
            f"npartitions must be None or a positive integer; got {npartitions!r}"
        )
    if int(npartitions) <= 0:
        raise CompositeIndexError(f"npartitions must be positive; got {npartitions!r}")
    return int(npartitions)


def coerce_component(value, kind: str, where: str = "key"):
    """Coerce a Python/NumPy scalar to the declared level kind.

    Never silently stringifies a value: mismatched types raise.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        raise CompositeIndexError(
            f"{where} component of a {kind} level must not be missing"
        )
    if isinstance(value, np.generic):
        value = value.item()
    if kind == "int64":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CompositeIndexError(
                f"{where} component {value!r} is not a valid int64 value "
                f"({type(value).__name__})"
            )
        return int(value)
    if isinstance(value, str):
        return value
    raise CompositeIndexError(
        f"{where} component {value!r} is not a valid string value "
        f"({type(value).__name__})"
    )


def normalize_divisions(divisions, level_kinds) -> tuple[tuple, ...]:
    """Validate explicit ``divisions`` and return them as tuples of tuples."""
    if not isinstance(divisions, (list, tuple)):
        raise CompositeIndexError(
            "divisions must be a list or tuple of (level0, level1) pairs"
        )
    if len(divisions) < 2:
        raise CompositeIndexError("divisions must contain at least two boundaries")
    normalized = []
    for idx, boundary in enumerate(divisions):
        if isinstance(boundary, np.ndarray):
            if boundary.ndim != 1:
                raise CompositeIndexError(
                    f"division {idx} must be one-dimensional; got shape {boundary.shape}"
                )
            boundary = tuple(boundary.tolist())
        if isinstance(boundary, list):
            boundary = tuple(boundary)
        if not isinstance(boundary, tuple):
            raise CompositeIndexError(
                f"division {idx} must be an (level0, level1) pair; got {boundary!r}"
            )
        if len(boundary) != 2:
            raise CompositeIndexError(
                "division boundaries must have exactly two components "
                f"(matching the two index levels); division {idx} has "
                f"{len(boundary)}"
            )
        normalized.append(
            tuple(
                coerce_component(value, kind, where=f"division {idx}")
                for value, kind in zip(boundary, level_kinds)
            )
        )
    check_boundary_order(normalized)
    return tuple(normalized)


def check_boundary_order(boundaries) -> None:
    """Strictly increasing except the last two entries may be equal."""
    for i in range(len(boundaries) - 2):
        if not boundaries[i] < boundaries[i + 1]:
            raise CompositeIndexError(
                "divisions must be strictly increasing (except the last two "
                f"entries may be equal); boundaries {i} and {i + 1} are "
                f"{boundaries[i]!r} and {boundaries[i + 1]!r}"
            )
    if not boundaries[-2] <= boundaries[-1]:
        raise CompositeIndexError(
            "divisions must be strictly increasing (except the last two "
            f"entries may be equal); boundaries {len(boundaries) - 2} and "
            f"{len(boundaries) - 1} are {boundaries[-2]!r} and "
            f"{boundaries[-1]!r}"
        )


def normalize_component_array(values, kind: str, where: str = "key column"):
    """Validate one vector of key components and return a homogeneous array.

    ``int64`` becomes a NumPy ``int64`` array (exact over the full range);
    ``string`` becomes an object array of ``str``.  Booleans, mixed types,
    unsupported dtypes and missing values raise
    :class:`CompositeIndexError` (a ``ValueError``).
    """
    if kind == "int64":
        if pd.api.types.is_integer_dtype(values.dtype):
            if str(values.dtype) != "int64":
                raise CompositeIndexError(
                    "composite set_index requires int64 levels; found dtype "
                    f"{values.dtype!r}"
                )
            if values.isna().any():
                raise CompositeIndexError(
                    f"{where} contains missing values; composite index keys "
                    "must not be missing"
                )
            return np.asarray(values.to_numpy(), dtype=np.int64)
        if values.dtype == object:
            raw = np.asarray(values.to_numpy(dtype=object), dtype=object)
            out = np.empty(len(raw), dtype=np.int64)
            for i, value in enumerate(raw):
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    raise CompositeIndexError(
                        f"{where} contains a non-integer value {value!r}; "
                        "int64 levels must not mix types"
                    )
                out[i] = int(value)
            return out
        raise CompositeIndexError(
            "composite set_index requires int64 levels; found dtype "
            f"{values.dtype!r}"
        )
    # string
    if pd.api.types.is_integer_dtype(values.dtype) or pd.api.types.is_bool_dtype(
        values.dtype
    ):
        raise CompositeIndexError(
            f"{where} is declared as a string level but has dtype "
            f"{values.dtype!r}; mixed values are not allowed"
        )
    raw = np.asarray(values.to_numpy(dtype=object), dtype=object)
    out = np.empty(len(raw), dtype=object)
    for i, value in enumerate(raw):
        if value is None or value is pd.NA or value is pd.NaT:
            raise CompositeIndexError(
                f"{where} contains missing values; composite index keys must "
                "not be missing"
            )
        if isinstance(value, str):
            out[i] = value
        elif isinstance(value, np.str_):
            out[i] = str(value)
        else:
            raise CompositeIndexError(
                f"{where} contains a non-string value {value!r}; string levels "
                "must not mix types"
            )
    return out


def plain_component(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def plain_key(key) -> tuple:
    return tuple(plain_component(value) for value in key)


# ---------------------------------------------------------------------------
# Vectorised key -> partition assignment
# ---------------------------------------------------------------------------


def partition_ids(first, second, boundaries) -> np.ndarray:
    """Partition index for each ``(first, second)`` key.

    ``first`` / ``second`` are homogeneous arrays produced by
    :func:`normalize_component_array`; ``boundaries`` is a sorted sequence of
    ``(a, b)`` tuples.  Fully vectorised, with a per-bucket fallback when
    several boundaries share the same first-level value.
    """
    n = len(first)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    first_boundaries = np.asarray(
        [b[0] for b in boundaries], dtype=np.asarray(first).dtype
    )
    second_boundaries = np.empty(len(boundaries), dtype=object)
    second_boundaries[:] = [b[1] for b in boundaries]
    second_values = np.asarray(second, dtype=object)

    base = np.searchsorted(first_boundaries, first, side="left")
    end = np.searchsorted(first_boundaries, first, side="right")
    has_bucket = end > base
    offsets = np.zeros(n, dtype=np.int64)
    if has_bucket.any():
        offsets[has_bucket] = second_values[has_bucket] >= second_boundaries[
            base[has_bucket]
        ]
    partitions = (base + offsets - 1).astype(np.int64)

    repeated = (end - base) > 1
    if repeated.any():
        for value in np.unique(first[repeated]):
            within = repeated & (first == value)
            start = int(np.searchsorted(first_boundaries, value, side="left"))
            stop = int(np.searchsorted(first_boundaries, value, side="right"))
            second_slice = second_boundaries[start:stop]
            positions = np.searchsorted(
                second_slice, second_values[within], side="right"
            )
            partitions[within] = start + positions - 1

    return np.minimum(partitions, len(boundaries) - 2)


def within_bounds(first, second, boundaries) -> np.ndarray:
    """Mask of keys inside ``[boundaries[0], boundaries[-1]]`` (inclusive)."""
    low_first, low_second = boundaries[0]
    high_first, high_second = boundaries[-1]
    lower_ok = (first > low_first) | ((first == low_first) & (second >= low_second))
    upper_ok = (first < high_first) | ((first == high_first) & (second <= high_second))
    return lower_ok & upper_ok


# ---------------------------------------------------------------------------
# Per-partition sampling
# ---------------------------------------------------------------------------


def summarize_partition(part: pd.DataFrame, columns, limit: int, kinds):
    """Reduce one partition to at most ``limit`` raw keys plus endpoints.

    Returns ``(row_count, keys, low_key, high_key)``.  The endpoints are the
    true minimum / maximum complete key of the partition, so the global
    planner observes the exact data range even when the sample is capped.
    Missing / mixed values raise :class:`CompositeIndexError` here, i.e. on
    first observation.
    """
    first = normalize_component_array(part[columns[0]], kinds[0], where=str(columns[0]))
    second = normalize_component_array(
        part[columns[1]], kinds[1], where=str(columns[1])
    )
    count = len(part)
    if count == 0:
        return 0, [], None, None
    second_obj = np.asarray(second, dtype=object)
    order = np.lexsort((second_obj, first))
    low_position = int(order[0])
    high_position = int(order[-1])
    low_key = (plain_component(first[low_position]), plain_component(second[low_position]))
    high_key = (
        plain_component(first[high_position]),
        plain_component(second[high_position]),
    )
    if count <= limit:
        selected = order
    else:
        positions = np.linspace(0, count - 1, max(limit - 2, 1)).astype(np.int64)
        selected = order[positions]
    keys = [
        (plain_component(first[i]), plain_component(second[i])) for i in selected
    ]
    if low_key not in keys:
        keys.append(low_key)
    if high_key not in keys:
        keys.append(high_key)
    return count, keys, low_key, high_key


def plan_divisions(samples, endpoints, target_npartitions: int):
    """Choose composite divisions from raw samples and global endpoints.

    Sample values are real keys (or the exact data endpoints), so the
    published divisions always describe the actual data.  When the requested
    partition count cannot be reached without splitting a run of identical
    complete keys, fewer partitions are produced.  An empty input returns
    ``()``.
    """
    low, high = endpoints
    if low is None or high is None:
        return ()
    ordered = sorted(set(samples)) if samples else []
    ordered = [key for key in ordered if low <= key <= high]
    if not ordered:
        ordered = [low]
    if ordered[0] != low:
        ordered.insert(0, low)
    if ordered[-1] != high:
        ordered.append(high)
    if len(ordered) < 2 or low == high:
        return (low, low)
    target = max(int(target_npartitions), 1)
    if target <= 1:
        return (low, high)
    n = len(ordered)
    cuts = []
    for k in range(1, target):
        position = int(round(k * (n - 1) / target))
        position = min(max(position, 1), n - 1)
        cut = ordered[position]
        if not cuts or cut > cuts[-1]:
            cuts.append(cut)
    boundaries = [low]
    for cut in cuts:
        if cut > boundaries[-1]:
            boundaries.append(cut)
    if boundaries[-1] != high:
        boundaries.append(high)
    if len(boundaries) < 2:
        boundaries.append(boundaries[0])
    return tuple(boundaries)
