import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pyfoamy.parsing import DecomposedCase, PolyMesh, processor_directories
from pyfoamy.reader import CaseReader

FIXTURES = Path(__file__).parent.parent / "fixtures"
DECOMPOSED = FIXTURES / "cavity_decomposed"


@pytest.fixture
def case(tmp_path):
    return Path(shutil.copytree(DECOMPOSED, tmp_path / "case"))


def _write_points(path: Path, points: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "FoamFile { format ascii; class vectorField; object points; }\n"
        f"{len(points)}\n("
        + " ".join(f"({x!r} {y!r} {z!r})" for x, y, z in points.tolist())
        + ")\n"
    )


def test_moved_points_are_assembled_from_the_processors(case):
    complete = PolyMesh.read(case / "constant" / "polyMesh")
    for directory in processor_directories(case):
        local = PolyMesh.read(directory / "constant" / "polyMesh")
        _write_points(directory / "0.2" / "polyMesh" / "points", local.points + 1)
    mesh = DecomposedCase(case).read_mesh_at(0.2)
    assert np.array_equal(mesh.points, complete.points + 1)
    assert np.array_equal(mesh.owner, complete.owner)


def test_a_topology_without_processor_addressing_raises(case):
    shutil.copytree(case / "constant" / "polyMesh", case / "0.2" / "polyMesh")
    with pytest.raises(ValueError, match="addressing"):
        DecomposedCase(case).read_mesh_at(0.2)


def test_no_processor_directories_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="processor"):
        DecomposedCase(tmp_path).read_mesh_at(0)


@pytest.mark.openfoam
def test_matches_reconstruct_par(case):
    (case / "system").mkdir()
    (case / "system" / "controlDict").write_text(
        "FoamFile { format ascii; class dictionary; object controlDict; }\n"
        "application foamRun; deltaT 0.01; writeControl timeStep;\n"
        "writeInterval 10; writeFormat binary;\n"
    )
    (case / "system" / "decomposeParDict").write_text(
        "FoamFile { format ascii; class dictionary; object decomposeParDict; }\n"
        "numberOfSubdomains 3; method simple; simpleCoeffs { n (3 1 1); }\n"
    )
    subprocess.run(
        ["reconstructPar", "-time", "0.1", "-case", str(case)],
        check=True, capture_output=True,
    )
    reader = CaseReader(case)
    for name in ("U", "p", "phi"):
        expected = reader.read_field_at(0.1, name, decomposed=False)
        field = reader.read_field_at(0.1, name, decomposed=True)
        assert np.array_equal(field.internal, expected.internal)
        for patch, values in expected.patches.items():
            if values.value is not None:
                assert np.array_equal(field.patches[patch].value, values.value)
