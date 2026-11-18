"""Reads function-object output under `postProcessing/` as numpy tables."""

import gzip
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pyfoamy.parsing.timedirs import (
    list_time_instances, resolve_time_instance,
)

_CELL = re.compile(r"\([^)]*\)|[^\s,()]+")
_KEY_VALUE = re.compile(r"#\s*(.+?)\s*:\s*(.*)$")
_PROBE = re.compile(r"#\s*(Probe \d+)\s+(.*)$")
_COMPONENTS = {
    1: ("ii",),
    2: ("x", "y"),
    3: ("x", "y", "z"),
    6: ("xx", "xy", "xz", "yy", "yz", "zz"),
    9: ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
}
_SET_SUFFIXES = (".xy", ".csv")


@dataclass
class Table:
    """Columns of a function object's output file(s).

    Vector and tensor columns are split into `<name>_x`, ... as
    OpenFOAM's set writers name them.

    Attributes:
        names: Column names; the first is the time or the coordinate.
        data: `(rows, len(names))` floats; NaN where a merged segment
            lacks a column.
        header: The `# key : value` and `# Probe i (x y z)` lines; of
            the last segment when merged.
        dropped: Rows left out for having another number of cells than
            the header (a field created after the first write).
        segments: `(file, first row in data)` per file read.
    """

    names: List[str]
    data: np.ndarray
    header: Dict[str, str] = field(default_factory=dict)
    dropped: int = 0
    segments: List[Tuple[Path, int]] = field(default_factory=list)

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.data[:, self.names.index(name)]
        except ValueError:
            raise KeyError(f"no column {name!r} in {self.names}") from None

    def to_pandas(self):
        """The table as a `pandas.DataFrame` (needs `pandas`)."""
        import pandas
        return pandas.DataFrame(self.data, columns=self.names)


def _lines(path: Path) -> List[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return stream.read().splitlines()
    return path.read_text().splitlines()


def _names(line: str, csv: bool) -> List[str]:
    line = line.lstrip("#")
    separator = "," if csv else "\t" if "\t" in line else None
    return [name.strip() for name in line.split(separator) if name.strip()]


def _expand(names: List[str], cells: List[str]) -> List[str]:
    expanded = []
    for name, cell in zip(names, cells):
        if not cell.startswith("("):
            expanded.append(name)
            continue
        count = len(cell[1:-1].split())
        components = _COMPONENTS.get(count, [str(i) for i in range(count)])
        expanded.extend(f"{name}_{c}" for c in components)
    return expanded


def read_table(path) -> Table:
    """Reads one `.dat`, probes, `.xy` or `.csv` file (optionally `.gz`).

    Raises:
        ValueError: If the file has no header line or a cell is not a
            number.
    """
    path = Path(path)
    lines = _lines(path)
    csv = path.name.removesuffix(".gz").endswith(".csv")
    header, names, rows, dropped = {}, None, [], 0
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if line.startswith("#") or (csv and names is None):
            if match := _KEY_VALUE.match(line) or _PROBE.match(line):
                header[match.group(1)] = match.group(2)
            else:
                names = _names(line, csv)
            continue
        if names is None:
            raise ValueError(f"{path}: no header before line {number}")
        cells = _CELL.findall(line)
        if len(cells) != len(names):
            dropped += 1
            continue
        try:
            rows.append([
                float(value) for cell in cells
                for value in cell.strip("()").split()
            ])
        except ValueError:
            raise ValueError(
                f"{path}:{number}: not a number in {line!r}"
            ) from None
        if len(rows) == 1:
            first = cells
    if names is None:
        raise ValueError(f"{path}: no header")
    if rows:
        names = _expand(names, first)
        width = len(names)
        kept = [row for row in rows if len(row) == width]
        dropped += len(rows) - len(kept)
        rows = kept
    data = np.array(rows, dtype=float).reshape(len(rows), len(names))
    return Table(names, data, header, dropped, [(path, 0)])


def _output_dir(case, region: Optional[str]) -> Path:
    base = Path(case) / "postProcessing"
    return base / region if region is not None else base


def _time_dirs(directory: Path) -> List[Tuple[float, str]]:
    return [
        (value, name) for value, name in list_time_instances(directory)
        if name != "constant"
    ]


def list_function_objects(case, region=None) -> List[str]:
    """Names of the function objects with output under `postProcessing/`."""
    base = _output_dir(case, region)
    if not base.is_dir():
        return []
    return sorted(
        entry.name for entry in base.iterdir()
        if entry.is_dir() and _time_dirs(entry)
    )


def list_times(case, name, region=None) -> List[float]:
    """Times (start times for time series) of a function object's output."""
    return [value for value, _ in _time_dirs(_function_dir(case, name, region))]


def _function_dir(case, name, region) -> Path:
    directory = _output_dir(case, region) / name
    if not directory.is_dir():
        raise FileNotFoundError(f"no function object output {directory}")
    return directory


def _one_file(directory: Path, file: Optional[str]) -> Path:
    if file is not None:
        return directory / file
    files = sorted(entry.name for entry in directory.iterdir())
    if len(files) != 1:
        raise ValueError(f"{directory} holds {files}; pass file=")
    return directory / files[0]


def read_time_series(cases, name, file=None, region=None) -> Table:
    """Reads a time series over restarts and chained cases.

    Every `<case>/postProcessing/<name>/<start time>/<file>` is one
    segment; segments are ordered by start time and each ends where the
    next starts, so a restart or a following case overrides the rows it
    repeats. Columns are the union over the segments.

    Args:
        cases: One case directory, or the cases of a chain (each started
            from the one before), in any order.
        file: The file in each start directory; needed only when there
            are several.

    Raises:
        FileNotFoundError: If a case has no output of `name`.
        ValueError: If two segments start at the same time, or `file` is
            needed.
    """
    if isinstance(cases, (str, os.PathLike)):
        cases = [cases]
    segments = []
    for case in cases:
        directory = _function_dir(case, name, region)
        segments.extend(
            (value, _one_file(directory / dirname, file))
            for value, dirname in _time_dirs(directory)
        )
    segments.sort(key=lambda segment: segment[0])
    for (start, path), (next_start, next_path) in zip(segments, segments[1:]):
        if start == next_start:
            raise ValueError(f"{path} and {next_path} start at {start}")
    tables = [read_table(path) for _, path in segments]
    names = list(dict.fromkeys(n for table in tables for n in table.names))
    # Relative tolerance as for time directories: "800" ends at 800.
    ends = [s * (1 - np.sign(s) * 1e-8) for s, _ in segments[1:]] + [np.inf]
    parts, header, dropped, sources, row = [], {}, 0, [], 0
    for table, end, (_, path) in zip(tables, ends, segments):
        data = table.data[table.data[:, 0] < end]
        part = np.full((len(data), len(names)), np.nan)
        part[:, [names.index(n) for n in table.names]] = data
        parts.append(part)
        header.update(table.header)
        dropped += table.dropped
        sources.append((path, row))
        row += len(data)
    data = np.concatenate(parts) if parts else np.empty((0, len(names)))
    return Table(names, data, header, dropped, sources)


def read_sets(case, name, time, region=None) -> Dict[str, Table]:
    """Reads the `raw` (`.xy`) or `csv` sets a function object wrote at a time.

    Returns:
        `{set name: Table}`.

    Raises:
        ValueError: If a file is in another set format.
    """
    directory = _function_dir(case, name, region)
    directory = directory / resolve_time_instance(directory, time)
    sets = {}
    for path in sorted(directory.iterdir()):
        stem = Path(path.name.removesuffix(".gz"))
        if stem.suffix not in _SET_SUFFIXES:
            raise ValueError(f"{path}: not a raw or csv set")
        sets[stem.stem] = read_table(path)
    return sets
