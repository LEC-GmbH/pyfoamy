"""The OpenFOAM version pyfoamy writes cases for."""

import contextlib
import contextvars
import os
from typing import Iterator

SUPPORTED_VERSIONS = ("14", "dev")
DEFAULT_VERSION = "14"

_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "openfoam_version", default=None,
)


def _check(version: str) -> str:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"OpenFOAM version {version!r} is not supported, "
            f"only {SUPPORTED_VERSIONS}"
        )
    return version


def openfoam_version() -> str:
    """The OpenFOAM version (`"14"` or `"dev"`) that output is written for.

    The version set by `use_openfoam_version()`, else the sourced
    `$WM_PROJECT_VERSION`, else `DEFAULT_VERSION`.

    Raises:
        ValueError: If that version is not in `SUPPORTED_VERSIONS`.
    """
    version = _override.get() or os.environ.get("WM_PROJECT_VERSION")
    return _check(version or DEFAULT_VERSION)


@contextlib.contextmanager
def use_openfoam_version(version: str) -> Iterator[str]:
    """Writes for `version` inside the `with` block, whatever is sourced.

    Raises:
        ValueError: If `version` is not in `SUPPORTED_VERSIONS`.
    """
    version = _check(str(version))
    token = _override.set(version)
    try:
        yield version
    finally:
        _override.reset(token)
