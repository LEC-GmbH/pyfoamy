"""Every script in `examples/` runs."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"

_MARKS = {
    "demo_read_dictionary.py": pytest.mark.openfoam,
    "demo_edit_and_write.py": pytest.mark.openfoam,
    "create_xi_fluid.py": pytest.mark.slow,
}
_IMPORTS = {
    "example_species_data.py": "cantera",
    "convert_mechanism.py": "cantera",
    "create_xi_fluid.py": "cantera",
}


@pytest.mark.parametrize(
    "script",
    [
        pytest.param(path, marks=_MARKS.get(path.name, ()), id=path.stem)
        for path in sorted(EXAMPLES.glob("*.py"))
    ],
)
def test_example_runs(script, tmp_path, monkeypatch):
    if script.name in _IMPORTS:
        pytest.importorskip(_IMPORTS[script.name])
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name="__main__")
