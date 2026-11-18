import struct
from pathlib import Path

import numpy as np
import pytest

from pyfoamy.parsing import FieldValues, PolyMesh, read_field
from pyfoamy.parsing.field import _to_plain
from pyfoamy.parsing.fieldfile import _read_dictionary

FIXTURES = Path(__file__).parent.parent / "fixtures"
FIELD_NAMES = ["alphat", "k", "nut", "omega", "p", "phi", "rho", "T", "U"]


def _cavity(variant):
    case = FIXTURES / f"cavity_{variant}"
    return case, PolyMesh.read(case / "constant" / "polyMesh")


def _plain(value):
    return np.asarray(_to_plain(value), float)


@pytest.mark.parametrize("variant", ["ascii", "binary"])
@pytest.mark.parametrize("name", FIELD_NAMES)
def test_values_equal_the_dictionary_reader(variant, name):
    case, mesh = _cavity(variant)
    dictionary = read_field(case / "0.1" / name)
    values = FieldValues.read(case / "0.1" / name, mesh)
    assert np.array_equal(
        values.internal, _plain(dictionary["internalField"][1])
    )
    assert values.patches.keys() == mesh.patches.keys()
    for patch, patch_values in values.patches.items():
        entry = dictionary["boundaryField"][patch]
        assert patch_values.type == str(entry["type"])
        if "value" not in entry:
            assert patch_values.value is None
        elif str(entry["value"][0]) == "nonuniform":
            assert np.array_equal(
                patch_values.value, _plain(entry["value"][1])
            )


def test_shapes_follow_the_field_class_and_the_mesh():
    case, mesh = _cavity("binary")
    u = FieldValues.read(case / "0.1" / "U", mesh)
    phi = FieldValues.read(case / "0.1" / "phi", mesh)
    assert (u.kind, u.internal.shape, u.n_components) == ("vol", (400, 3), 3)
    assert u.patches["movingWall"].value.shape == (20, 3)
    assert (phi.kind, phi.internal.shape) == ("surface", (760,))


def test_uniform_values_are_repeated_and_missing_values_are_none():
    case, mesh = _cavity("binary")
    u = FieldValues.read(case / "0.1" / "U", mesh)
    assert (u.patches["movingWall"].value == [1, 0, 0]).all()
    p = FieldValues.read(case / "0.1" / "p", mesh)
    assert p.patches["movingWall"].type == "zeroGradient"
    assert p.patches["movingWall"].value is None


def _write(path, body: bytes, fmt="ascii", cls="volVectorField"):
    path.write_bytes(
        b"FoamFile { format " + fmt.encode() + b"; class "
        + cls.encode() + b"; object U; }\n" + body
    )


def test_binary_empty_list_without_parentheses(tmp_path):
    # decomposePar/foamRun write an empty binary list as its count alone.
    case, mesh = _cavity("binary")
    blob = struct.pack("<1200d", *range(1200))
    body = (
        b"dimensions [0 1 -1 0 0 0 0];\n"
        b"internalField nonuniform List<vector> 400(" + blob + b");\n"
        b"boundaryField {\n"
        b" movingWall { type fixedValue; value nonuniform List<vector> \n0\n; }\n"
        b" fixedWalls { type noSlip; }\n"
        b" frontAndBack { type empty; }\n}\n"
    )
    path = tmp_path / "U"
    _write(path, body, fmt="binary")
    assert len(read_field(path)["boundaryField"]["movingWall"]["value"][1]) == 0
    u = FieldValues.read(path, mesh)
    assert u.internal[1].tolist() == [3, 4, 5]
    assert u.patches["movingWall"].value.shape == (0, 3)


def test_macro_and_field_function_values_are_none(tmp_path):
    _, mesh = _cavity("ascii")
    body = (
        b"dimensions [0 1 -1 0 0 0 0];\n"
        b"internalField uniform (0 0 0);\n"
        b"boundaryField {\n"
        b" movingWall { type fixedValue; value $internalField; }\n"
        b" fixedWalls { type fixedValue; value { type coded; name x; } }\n"
        b" frontAndBack { type empty; }\n}\n"
    )
    path = tmp_path / "U"
    _write(path, body)
    u = FieldValues.read(path, mesh)
    assert u.patches["movingWall"].value is None
    assert u.patches["fixedWalls"].value is None
    assert (u.internal == 0).all() and u.internal.shape == (400, 3)


def test_point_field_patch_size_is_its_point_count(tmp_path):
    _, mesh = _cavity("ascii")
    body = (
        b"dimensions [0 1 0 0 0 0 0];\n"
        b"internalField uniform (0 0 0);\n"
        b"boundaryField {\n"
        b" movingWall { type fixedValue; value uniform (1 0 0); }\n"
        b" fixedWalls { type fixedValue; value uniform (0 0 0); }\n"
        b" frontAndBack { type empty; }\n}\n"
    )
    path = tmp_path / "pointDisplacement"
    _write(path, body, cls="pointVectorField")
    d = FieldValues.read(path, mesh)
    assert d.internal.shape == (882, 3)
    assert d.patches["movingWall"].value.shape == (42, 3)


def test_unsupported_class_raises(tmp_path):
    _, mesh = _cavity("ascii")
    path = tmp_path / "U"
    _write(path, b"value uniform 0;\n", cls="volScalarField::Internal")
    with pytest.raises(ValueError, match="unsupported field class"):
        FieldValues.read(path, mesh)


def test_array_mode_keeps_binary_values_exact():
    case, _ = _cavity("binary")
    d = _read_dictionary(case / "0.1" / "U", as_array=True)
    internal = d["internalField"][1]
    assert isinstance(internal, np.ndarray) and internal.shape == (400, 3)
