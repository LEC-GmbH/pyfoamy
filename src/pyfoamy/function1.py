"""Builds `Function1` dicts: tables and the wrappers around them."""

from pathlib import Path
from typing import Optional, Sequence, Tuple


def _table(out_of_bounds, interpolation_scheme) -> dict:
    result = {"type": "table"}
    if out_of_bounds is not None:
        result["outOfBounds"] = out_of_bounds
    if interpolation_scheme is not None:
        result["interpolationScheme"] = interpolation_scheme
    return result


def table(
    values, *, out_of_bounds: Optional[str] = None,
    interpolation_scheme: Optional[str] = None,
) -> dict:
    """A `table` of inline `(x, y)` pairs.

    Args:
        out_of_bounds: `"error"`, `"warn"`, `"clamp"`, `"zero"` or
            `"repeat"` (OpenFOAM default: `clamp`).
        interpolation_scheme: E.g. `"step"` (OpenFOAM default: `linear`).
    """
    result = _table(out_of_bounds, interpolation_scheme)
    result["values"] = [tuple(pair) for pair in values]
    return result


def table_file(
    file: str, *, out_of_bounds: Optional[str] = None,
    interpolation_scheme: Optional[str] = None, separator: str = " ",
    merge_separators: bool = False, n_header_line: int = 0,
    columns: Tuple[int, int] = (0, 1),
) -> dict:
    """A `table` read from a csv file, see `table()`.

    Args:
        file: Path as OpenFOAM reads it, e.g.
            `"$FOAM_CASE/constant/p.dat"`.
        columns: Indices of the x and the y column.
    """
    result = _table(out_of_bounds, interpolation_scheme)
    result.update({
        "file": f'"{file}"',
        "format": "csv",
        "separator": f'"{separator}"',
        "mergeSeparators": merge_separators,
        "nHeaderLine": n_header_line,
        "columns": list(columns),
    })
    return result


def repeat(value, period, *, start=None) -> dict:
    """`value` repeated every `period`.

    The wrapped function is read on `[0, period)`.

    Args:
        start: x at which the first period starts (OpenFOAM default: 0).
    """
    result = {"type": "repeat", "period": period}
    if start is not None:
        result["start"] = start
    result["value"] = value
    return result


def shift(value, start) -> dict:
    """`value` read at `x - start`."""
    return {"type": "shift", "start": start, "value": value}


def scale(value, scale, *, x_scale=None) -> dict:
    """`value` multiplied by `scale`, itself a number or `Function1`.

    Args:
        x_scale: Factor on x before `value` is read (OpenFOAM default: 1).
    """
    result = {"type": "scale", "scale": scale}
    if x_scale is not None:
        result["xScale"] = x_scale
    result["value"] = value
    return result


def periodic_samples(x: Sequence[float], y: Sequence[float], period, fill):
    """Moves samples of a periodic function onto `[0, period]`.

    Whole periods are added until the first sample is inside; samples
    past `period` wrap to the front, and both ends are closed with
    `fill` unless a sample is there. Needs numpy.

    Args:
        fill: The function's value at 0.

    Returns:
        The `x` and `y` arrays.

    Raises:
        ValueError: If `period` is not positive.
    """
    import numpy

    period = float(period)
    if period <= 0:
        raise ValueError(f"period must be positive, not {period}")
    x = numpy.array(x, dtype=float)
    y = numpy.array(y, dtype=float)
    while x.min() < 0:
        x += period
    while x.min() > period:
        x -= period
    if len(x) < 2:
        return x, y

    if numpy.isclose(x[-1] - x[0], period):
        x, y = x[:-1], y[:-1]
    wrapped = x > period
    x = numpy.concatenate((x[wrapped] - period, x[~wrapped]))
    y = numpy.concatenate((y[wrapped], y[~wrapped]))
    if not numpy.isclose(x[0], 0):
        x, y = numpy.insert(x, 0, 0), numpy.insert(y, 0, fill)
    if not numpy.isclose(x[-1], period):
        x, y = numpy.append(x, period), numpy.append(y, fill)
    return x, y


def write_samples(path: Path, x: Sequence[float], y: Sequence[float]):
    """Writes the two-column file `table_file()` reads."""
    with open(path, "w") as stream:
        for pair in zip(x, y):
            stream.write("%g %g\n" % pair)
