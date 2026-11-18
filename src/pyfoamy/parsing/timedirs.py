"""Time directory lookup, as in OpenFOAM's `Time::findInstance`."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

# Relative tolerance like Foam::equal(), so 0.1 finds directory "0.1".
_TIME_RELATIVE_TOLERANCE = 1e-8
_PROCESSOR = re.compile(r"processor(\d+)$")


def _parse_time_name(name: str) -> Optional[float]:
    try:
        return float(name)
    except ValueError:
        return None


def list_time_instances(case_dir) -> List[Tuple[float, str]]:
    """Lists the time directories and `constant` of a case.

    Returns:
        `(value, name)` pairs ascending by value; `"constant"` has
        value 0 and sorts after a `"0"` directory.
    """
    case_dir = Path(case_dir)
    instances = []
    for entry in case_dir.iterdir():
        if not entry.is_dir() or entry.name == "constant":
            continue
        value = _parse_time_name(entry.name)
        if value is not None:
            instances.append((value, entry.name))
    if (case_dir / "constant").is_dir():
        instances.append((0.0, "constant"))
    instances.sort(key=lambda pair: (pair[0], pair[1] == "constant"))
    return instances


def resolve_time_instance(case_dir, time_value) -> str:
    """Returns the name of the time directory at exactly `time_value`.

    Raises:
        FileNotFoundError: If no numeric time directory matches.
    """
    case_dir = Path(case_dir)
    tolerance = abs(time_value) * _TIME_RELATIVE_TOLERANCE
    for value, name in list_time_instances(case_dir):
        if name != "constant" and abs(value - time_value) <= tolerance:
            return name
    raise FileNotFoundError(
        f"no time directory matches {time_value} under {case_dir}"
    )


def _exists_with_optional_gz(path: Path) -> bool:
    return path.exists() or path.with_name(path.name + ".gz").exists()


def find_instance(
    case_dir,
    time_value,
    filename,
    mesh_subdir="polyMesh",
    stop_instance="constant",
    region=None,
) -> Path:
    """Finds a mesh file in the latest instance at or before a time.

    Returns:
        `<instance>[/<region>]/<mesh_subdir>/<filename>`; the file may
        exist only as `.gz`.

    Raises:
        FileNotFoundError: If no instance down to `stop_instance` has
            the file.
    """
    case_dir = Path(case_dir)
    instances = list_time_instances(case_dir)
    tolerance = time_value * _TIME_RELATIVE_TOLERANCE
    candidates = [
        name
        for value, name in instances
        if name == stop_instance or value <= time_value + tolerance
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no time instances at or before {time_value} under "
            f"{case_dir}"
        )
    for name in reversed(candidates):
        instance_dir = case_dir / name
        if region is not None:
            instance_dir = instance_dir / region
        path = instance_dir / mesh_subdir / filename
        if _exists_with_optional_gz(path):
            return path
        if name == stop_instance:
            break
    raise FileNotFoundError(
        f"{mesh_subdir}/{filename} not found in any time instance at or "
        f"before {time_value} (down to {stop_instance!r}) under "
        f"{case_dir}"
        + ("" if region is None else f" for region {region!r}")
    )


def processor_directories(root) -> List[Path]:
    """The `processorN` directories of a case, ascending by `N`."""
    found = [
        (int(match.group(1)), path)
        for path in Path(root).iterdir()
        if (match := _PROCESSOR.match(path.name)) and path.is_dir()
    ]
    return [path for _, path in sorted(found)]
