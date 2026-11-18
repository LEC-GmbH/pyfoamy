from pathlib import Path

import pytest
from conftest import HAVE_TUTORIALS, TUTORIALS_ROOT

from pyfoamy.parsing import PolyMesh

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Few tutorials ship a polyMesh; sweep whatever is there.
_REQUIRED = ("points", "faces", "owner", "neighbour")


def _has_file(directory: Path, name: str) -> bool:
    plain_exists = (directory / name).exists()
    gz_exists = (directory / f"{name}.gz").exists()
    return plain_exists or gz_exists


def _iter_polymesh_dirs():
    for variant in ("ascii", "binary"):
        yield FIXTURES / f"cavity_{variant}" / "constant" / "polyMesh"
    if not HAVE_TUTORIALS:
        return
    for path in TUTORIALS_ROOT.rglob("polyMesh"):
        if path.is_dir() and all(_has_file(path, n) for n in _REQUIRED):
            yield path


@pytest.mark.tutorials
def test_polymesh_reads_every_real_polymesh_directory():
    checked = 0
    failures = []

    for directory in _iter_polymesh_dirs():
        checked += 1
        try:
            mesh = PolyMesh.read(directory)
        except Exception as exc:  # noqa: BLE001 - report any failure
            failures.append((directory, f"read raised: {exc}"))
            continue

        if not len(mesh.points):
            failures.append((directory, "no points"))
        if len(mesh.faces) != len(mesh.owner):
            failures.append((directory, "len(faces) != len(owner)"))
        if len(mesh.neighbour) > len(mesh.owner):
            failures.append(
                (directory, "len(neighbour) > len(owner)")
            )
        if mesh.faces.values.max(initial=-1) >= len(mesh.points):
            msg = "a face references an out-of-range point"
            failures.append((directory, msg))

    assert checked >= 3, f"expected several polyMesh dirs, found {checked}"

    if failures:
        details = "\n".join(f"  {d}: {e}" for d, e in failures)
        pytest.fail(
            f"{len(failures)}/{checked} polyMesh directories failed:\n"
            f"{details}"
        )
