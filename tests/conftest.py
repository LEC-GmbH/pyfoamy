import os
from pathlib import Path

import pytest

_tutorials = os.environ.get("PYFOAMY_TUTORIALS_ROOT") or os.environ.get(
    "FOAM_TUTORIALS"
)
_SKIP_REASONS = {
    "openfoam": (
        None if os.environ.get("WM_PROJECT_DIR")
        else "needs a sourced OpenFOAM install"
    ),
    "tutorials": (
        None if _tutorials and Path(_tutorials).expanduser().is_dir()
        else "needs $PYFOAMY_TUTORIALS_ROOT or a sourced OpenFOAM install"
    ),
}


def pytest_collection_modifyitems(items):
    for item in items:
        for mark, reason in _SKIP_REASONS.items():
            if reason and mark in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _no_custom_lib_build(request, monkeypatch):
    # SolverSetup.build() builds libpyfoamy.so whenever OpenFOAM is sourced;
    # only openfoam-marked tests may do that.
    if "openfoam" not in request.keywords:
        from pyfoamy.models import solver_setup
        monkeypatch.setattr(solver_setup, "ensure_built", lambda: None)
