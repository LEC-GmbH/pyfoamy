"""Locating and building pyfoamy's OpenFOAM-linked C++ sources."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_CHECKOUT_SOURCES = Path(__file__).resolve().parents[2] / "openfoam"


def source_dir(*parts: str) -> Path | None:
    override = os.environ.get("PYFOAMY_OPENFOAM_SRC")
    root = Path(override).expanduser() if override else _CHECKOUT_SOURCES
    path = root.joinpath(*parts)
    return path if path.is_dir() else None


def _ignore_build_output(dirname, names):
    if Path(dirname).name == "Make":
        return [n for n in names if (Path(dirname) / n).is_dir()]
    return [n for n in names if n == "lnInclude"]


def wmake_libso(src_dir: Path) -> subprocess.CompletedProcess:
    # wmake writes lnInclude/ and objects next to the sources, so build a
    # copy: the sources may be read-only or shared between installs.
    with tempfile.TemporaryDirectory(prefix="pyfoamy-wmake-") as tmp:
        build_dir = Path(tmp) / src_dir.name
        shutil.copytree(src_dir, build_dir, ignore=_ignore_build_output)
        return subprocess.run(
            ["wmake", "libso", str(build_dir)],
            capture_output=True,
            text=True,
        )
