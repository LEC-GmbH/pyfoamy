import os
from pathlib import Path

import numpy as np
import pytest

from pyfoamy.native import NativeError, _lib_path, open_case
from pyfoamy.parsing.field import FieldValues
from pyfoamy.parsing.polymesh import PolyMesh

FIXTURES = Path(__file__).parent / "fixtures"
REGION_CASE = FIXTURES / "multiregion_cylinder"


def _assert_same_mesh(a: PolyMesh, b: PolyMesh):
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.faces.offsets, b.faces.offsets)
    assert np.array_equal(a.faces.values, b.faces.values)
    assert np.array_equal(a.owner, b.owner)
    assert np.array_equal(a.neighbour, b.neighbour)
    assert a.n_cells == b.n_cells
    assert a.patches.keys() == b.patches.keys()
    for name, patch in b.patches.items():
        native = a.patches[name]
        assert (native.type, native.start_face, native.n_faces) == (
            patch.type, patch.start_face, patch.n_faces
        )
        # OpenFOAM adds a constraint type to the groups the file lists.
        assert set(patch.groups) <= set(native.groups)


def _assert_same_field(native, python: FieldValues):
    # Native evaluates every patch; python has the ones the file names
    # literally, and None where it gives no value.
    assert native.dimensions == python.dimensions
    assert np.allclose(native.internal, python.internal, rtol=1e-12)
    assert python.patches.keys() <= native.patches.keys()
    for name, values in python.patches.items():
        assert native.patches[name].type == values.type
        if values.value is not None:
            assert np.allclose(
                native.patches[name].value, values.value, rtol=1e-12
            )


# Pre-shipped mesh (21812 points, 4 patches), larger than the cavity
# fixtures; only a "0" directory, so no real mesh change is read.
REAL_CASE = (
    Path(os.environ.get("FOAM_TUTORIALS", "")) / "incompressibleFluid"
    / "airFoil2D"
)


def test_missing_openfoam_env_raises_native_error(monkeypatch):
    # Forces the `_require_env` guard whether or not OpenFOAM is sourced.
    monkeypatch.delenv("WM_PROJECT_DIR", raising=False)
    monkeypatch.delenv("FOAM_USER_LIBBIN", raising=False)
    with pytest.raises(NativeError, match="WM_PROJECT_DIR"):
        open_case("/tmp", "whatever")


@pytest.mark.openfoam
def test_open_case_on_missing_case_raises_native_error(tmp_path):
    with pytest.raises(NativeError):
        open_case(tmp_path, "no_such_case")


@pytest.mark.openfoam
def test_read_mesh_at_matches_pure_python_reader():
    with open_case(REAL_CASE.parent, REAL_CASE.name) as case:
        mesh = case.read_mesh_at(0.0)
    _assert_same_mesh(mesh, PolyMesh.read(REAL_CASE / "constant" / "polyMesh"))


@pytest.mark.openfoam
def test_repeated_reads_on_the_same_handle_agree():
    # Repeated readUpdate() on an open, unchanged case.
    with open_case(REAL_CASE.parent, REAL_CASE.name) as case:
        first = case.read_mesh_at(0.0)
        second = case.read_mesh_at(0.0)
    _assert_same_mesh(first, second)


@pytest.mark.openfoam
def test_context_manager_closes_handle_even_on_exception():
    with pytest.raises(ValueError):
        with open_case(REAL_CASE.parent, REAL_CASE.name) as case:
            assert case._handle
            raise ValueError("boom")
    assert not case._handle


@pytest.mark.openfoam
def test_repeated_calls_do_not_trigger_a_rebuild():
    with open_case(REAL_CASE.parent, REAL_CASE.name) as case:
        case.read_mesh_at(0.0)
        before = _lib_path().stat().st_mtime_ns
        case.read_mesh_at(0.0)
    after = _lib_path().stat().st_mtime_ns
    assert before == after


@pytest.mark.openfoam
def test_read_field_at_matches_pure_python_reader_on_real_case():
    mesh = PolyMesh.read(REAL_CASE / "constant" / "polyMesh")
    with open_case(REAL_CASE.parent, REAL_CASE.name) as case:
        for name in ("p", "U"):
            _assert_same_field(
                case.read_field_at(0.0, name),
                FieldValues.read(REAL_CASE / "0" / name, mesh),
            )


@pytest.mark.openfoam
def test_read_field_at_on_missing_field_raises_native_error():
    with open_case(REAL_CASE.parent, REAL_CASE.name) as case:
        with pytest.raises(NativeError):
            case.read_field_at(0.0, "no_such_field")


@pytest.mark.openfoam
def test_open_case_with_region_matches_pure_python_on_real_fixture():
    # Only "fluid": "film" uses filmWall from a library the backend
    # does not link.
    with open_case(
        REGION_CASE.parent, REGION_CASE.name, region="fluid"
    ) as case:
        native_mesh = case.read_mesh_at(0.0)
        native_u = case.read_field_at(0.0, "U")

    python_mesh = PolyMesh.read_at(REGION_CASE, 0.0, region="fluid")
    _assert_same_mesh(native_mesh, python_mesh)
    _assert_same_field(
        native_u,
        FieldValues.read(REGION_CASE / "0" / "fluid" / "U", python_mesh),
    )


@pytest.mark.openfoam
def test_open_case_with_unknown_region_raises_native_error():
    with pytest.raises(NativeError):
        open_case(
            REGION_CASE.parent,
            REGION_CASE.name,
            region="no_such_region",
        )


@pytest.mark.slow
@pytest.mark.openfoam
def test_editing_native_source_triggers_a_rebuild():
    with open_case(REAL_CASE.parent, REAL_CASE.name) as case:
        case.read_mesh_at(0.0)
    before = _lib_path().stat().st_mtime_ns

    native_src = (
        Path(__file__).parent.parent / "openfoam" / "native" / "pfNative.C"
    )
    original = native_src.read_text()
    try:
        native_src.write_text(
            original + "\n// cache-invalidation test\n"
        )
        with open_case(REAL_CASE.parent, REAL_CASE.name) as case:
            case.read_mesh_at(0.0)
        after = _lib_path().stat().st_mtime_ns
        assert after != before
    finally:
        native_src.write_text(original)
        # Rebuild the .so from the reverted source.
        with open_case(REAL_CASE.parent, REAL_CASE.name) as case:
            case.read_mesh_at(0.0)
