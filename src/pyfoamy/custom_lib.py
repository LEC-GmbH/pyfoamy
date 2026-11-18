"""Auto-builds pyfoamy's custom OpenFOAM library (`libpyfoamy.so`)."""

import fcntl
import hashlib
import os
import re
from pathlib import Path
from typing import List

from pyfoamy._openfoam_build import source_dir, wmake_libso
from pyfoamy.openfoam_version import SUPPORTED_VERSIONS

_LIB_NAME = "libpyfoamy.so"
_MAKE_FILES = ("Make/files", "Make/options")


class CustomLibError(RuntimeError):
    """Raised for an unsupported or unsourced OpenFOAM, or a build failure."""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise CustomLibError(
            f"${name} is not set - source an OpenFOAM install's "
            "etc/bashrc before using pyfoamy's custom library"
        )
    return value


def _check_version() -> str:
    # The sourced version, not openfoam_version(): the build links to it.
    version = _require_env("WM_PROJECT_VERSION")
    if version not in SUPPORTED_VERSIONS:
        raise CustomLibError(
            f"OpenFOAM version {version!r} is not supported by "
            f"libpyfoamy.so, only {SUPPORTED_VERSIONS}"
        )
    return version


def _version_dir() -> Path:
    path = source_dir("lib", _check_version())
    if path is None:
        raise CustomLibError(
            "pyfoamy's OpenFOAM sources not found; set "
            "$PYFOAMY_OPENFOAM_SRC to a checkout's openfoam/ directory"
        )
    return path


def _installed_lib() -> Path | None:
    # A user build would shadow a site build: $FOAM_USER_LIBBIN comes first
    # on LD_LIBRARY_PATH.
    for name in ("FOAM_LIBBIN", "FOAM_SITE_LIBBIN"):
        directory = os.environ.get(name)
        if directory and (Path(directory) / _LIB_NAME).is_file():
            return Path(directory) / _LIB_NAME
    return None


def _lib_dir() -> Path:
    return Path(_require_env("FOAM_USER_LIBBIN"))


def _lib_path() -> Path:
    return _lib_dir() / _LIB_NAME


def _sig_path() -> Path:
    return _lib_dir() / (_LIB_NAME + ".pyfoamy-sig")


def _tracked_source_files(src_dir: Path) -> List[Path]:
    files = sorted(src_dir.rglob("*.C")) + sorted(src_dir.rglob("*.H"))
    files += [src_dir / name for name in _MAKE_FILES]
    return files


def _linked_lib_names(src_dir: Path) -> List[str]:
    # Parsed from Make/options so the staleness check follows LIB_LIBS.
    options_text = (src_dir / "Make" / "options").read_text()
    return [f"lib{name}.so" for name in re.findall(r"-l(\S+)", options_text)]


def _compute_signature(src_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in _tracked_source_files(src_dir):
        digest.update(path.read_bytes())
    digest.update(_require_env("WM_PROJECT_DIR").encode())
    digest.update(os.environ.get("WM_OPTIONS", "").encode())
    lib_bin = Path(_require_env("FOAM_LIBBIN"))
    for name in _linked_lib_names(src_dir):
        try:
            stat = (lib_bin / name).stat()
        except FileNotFoundError:
            continue
        digest.update(f"{name}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return digest.hexdigest()


def _up_to_date(signature: str) -> bool:
    return (
        _lib_path().exists()
        and _sig_path().exists()
        and _sig_path().read_text() == signature
    )


def _build(src_dir: Path, signature: str) -> None:
    _lib_dir().mkdir(parents=True, exist_ok=True)
    lock_path = _lib_dir() / (_LIB_NAME + ".pyfoamy-lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # Another process may have built it while we waited.
            if _up_to_date(signature):
                return
            result = wmake_libso(src_dir)
            if result.returncode != 0:
                raise CustomLibError(
                    "building pyfoamy's custom OpenFOAM library "
                    f"failed:\n{result.stdout}\n{result.stderr}"
                )
            _sig_path().write_text(signature)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def ensure_built() -> Path:
    """Builds `libpyfoamy.so` for the sourced OpenFOAM if missing or stale.

    A library installed in `$FOAM_LIBBIN` or `$FOAM_SITE_LIBBIN` is used
    as is. Otherwise builds `openfoam/lib/<version>/` for
    `$WM_PROJECT_VERSION` into `$FOAM_USER_LIBBIN`; stale means tracked
    sources, `$WM_PROJECT_DIR`/`$WM_OPTIONS` or the linked OpenFOAM
    libraries changed.

    Returns:
        The path to the library.

    Raises:
        CustomLibError: If no OpenFOAM is sourced, its version is
            unsupported, or the build fails.
    """
    _check_version()
    installed = _installed_lib()
    if installed is not None:
        return installed
    src_dir = _version_dir()
    signature = _compute_signature(src_dir)
    if not _up_to_date(signature):
        _build(src_dir, signature)
    return _lib_path()
