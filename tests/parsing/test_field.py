import re
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import TUTORIALS_ROOT

from pyfoamy.parsing.field import Field, FieldValue, read_field_value
from pyfoamy.parsing.grammar import (
    Bracket,
    Dictionary,
    DimensionSet,
    dimension_bracket,
    dimensioned,
    parse_dimensions,
)
from pyfoamy.parsing.timedirs import resolve_time_instance

FIXTURES = Path(__file__).parent.parent / "fixtures"
TIME_DIR = "0.1"
FIELD_NAMES = [
    "alphat", "k", "nut", "omega", "p", "phi", "rho", "T", "U",
]

# name -> the dimensions a real solver run wrote for that field.
EXPECTED_DIMENSIONS = {
    "alphat": (1, -1, -1, 0, 0, 0, 0),
    "k": (0, 2, -2, 0, 0, 0, 0),
    "nut": (0, 2, -1, 0, 0, 0, 0),
    "omega": (0, 0, -1, 0, 0, 0, 0),
    "p": (1, -1, -2, 0, 0, 0, 0),
    "phi": (1, 0, -1, 0, 0, 0, 0),
    "rho": (1, -3, 0, 0, 0, 0, 0),
    "T": (0, 0, 0, 1, 0, 0, 0),
    "U": (0, 1, -1, 0, 0, 0, 0),
}

REAL_NAMED_DIMENSION_FIELDS = [
    ("U", (0, 1, -1, 0, 0, 0, 0)),
    ("p", (0, 2, -2, 0, 0, 0, 0)),
]


def field_path(variant: str, name: str) -> Path:
    return FIXTURES / f"cavity_{variant}" / TIME_DIR / name


@pytest.mark.parametrize("variant", ["ascii", "binary"])
@pytest.mark.parametrize("name", FIELD_NAMES)
def test_field_dimensions_match_expected(variant, name):
    f = Field.read(field_path(variant, name))
    assert f.dimensions.as_tuple() == EXPECTED_DIMENSIONS[name]


@pytest.mark.parametrize("variant", ["ascii", "binary"])
@pytest.mark.parametrize("name", FIELD_NAMES)
def test_field_internal_field_is_nonuniform_field_value(variant, name):
    f = Field.read(field_path(variant, name))
    value = f.internal_field
    assert isinstance(value, FieldValue)
    assert value.uniform is False
    # phi is a surface field (760 internal faces), the others have one
    # value per cell (400).
    assert len(value.value) == (760 if name == "phi" else 400)


@pytest.mark.parametrize("variant", ["ascii", "binary"])
@pytest.mark.parametrize("name", FIELD_NAMES)
def test_field_boundary_field_patches_are_typed(variant, name):
    f = Field.read(field_path(variant, name))
    patches = f.boundary_field
    assert set(patches) == {"movingWall", "fixedWalls", "frontAndBack"}
    for patch_name, patch in patches.items():
        assert isinstance(patch.type, str)
        if "value" in patch:
            value = patch["value"]
            assert isinstance(value, FieldValue)


@pytest.mark.openfoam
@pytest.mark.tutorials
@pytest.mark.parametrize("name,expected", REAL_NAMED_DIMENSION_FIELDS)
def test_named_dimension_shorthand_on_real_tutorial_files(name, expected):
    f = Field.read(
        TUTORIALS_ROOT / "legacy/incompressible/icoFoam/cavity/cavity/0"
        / name
    )
    assert f.dimensions.as_tuple() == expected


@pytest.mark.openfoam
@pytest.mark.tutorials
@pytest.mark.slow
def test_named_dimension_vocabulary_covers_real_tutorial_usage():
    # Every named dimension or expression in the tutorials validates
    # against `foamUnits -all`.
    exprs_used = set()
    for path in TUTORIALS_ROOT.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("dimensions"):
                continue
            start = stripped.find("[")
            end = stripped.find("]")
            if start == -1 or end == -1:
                continue
            body = stripped[start + 1:end].strip()
            if body and body[0].isalpha():
                exprs_used.add(body)
    assert exprs_used, "expected to find at least one real usage"
    failures = {}
    for expr in exprs_used:
        try:
            parse_dimensions(Bracket(expr))
        except ValueError as exc:
            failures[expr] = str(exc)
    assert not failures, (
        f"unhandled named dimension expressions: {failures}"
    )


def test_uniform_scalar_shorthand_expands_to_repeated_value():
    text = (
        "FoamFile { version 2.0; format ascii; class volScalarField; "
        "object p; }\n"
        "internalField nonuniform List<scalar> 3{0};\n"
    )
    d = Dictionary(text)
    value = read_field_value(d["internalField"])
    assert value.uniform is False
    assert value.value == [0, 0, 0]


def test_uniform_vector_shorthand_expands_to_repeated_value():
    text = (
        "FoamFile { version 2.0; format ascii; class volVectorField; "
        "object U; }\n"
        "internalField nonuniform List<vector> 3{(1 2 3)};\n"
    )
    d = Dictionary(text)
    value = read_field_value(d["internalField"])
    assert value.uniform is False
    assert value.value == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]


def test_field_function_value_is_exposed_raw():
    text = (
        "FoamFile { version 2.0; format ascii; class volScalarField; "
        "object p; }\n"
        "internalField { type coded; someParam 1; }\n"
    )
    d = Dictionary(text)
    value = read_field_value(d["internalField"])
    assert value.type == "coded"
    assert value.dict["someParam"] == 1


@pytest.mark.openfoam
@pytest.mark.openfoam
def test_shipped_dimension_table_covers_foam_units():
    from pyfoamy.parsing._dimension_table import DIMENSIONS, UNITS

    if shutil.which("foamUnits") is None:
        pytest.skip("foamUnits not on $PATH (OpenFOAM < 14)")
    text = subprocess.run(
        ["foamUnits", "-all"], capture_output=True, text=True, check=True,
    ).stdout
    dims = re.findall(
        r"^Dimension \[(\w+)\]\n(?:\+ .*\n)*?\+ Exponents = \[([^\]]*)\]",
        text, re.MULTILINE,
    )
    assert dims
    for name, exponents in dims:
        assert DIMENSIONS[name] == tuple(map(int, exponents.split()))
    assert set(re.findall(r"^Unit \[([^\]]+)\]", text, re.MULTILINE)) <= UNITS


def test_unknown_named_dimension_raises():
    text = (
        "FoamFile { version 2.0; format ascii; class volScalarField; "
        "object p; }\n"
        "dimensions [totallyMadeUp];\n"
    )
    d = Dictionary(text)
    with pytest.raises(ValueError):
        parse_dimensions(d["dimensions"])


def test_empty_dimensions_bracket_is_dimensionless():
    # "[]" is dimensionless (dimensionSetIO.C).
    text = (
        "FoamFile { version 2.0; format ascii; class volScalarField; "
        "object p; }\n"
        "dimensions [];\n"
    )
    d = Dictionary(text)
    assert parse_dimensions(d["dimensions"]) == DimensionSet(
        0, 0, 0, 0, 0, 0, 0
    )


def test_five_element_dimensions_defaults_current_and_luminous_to_zero():
    # dimensionSetIO.C reads 5 exponents, plus 2 only if a 6th number
    # follows, so 5 values are valid.
    text = (
        "FoamFile { version 2.0; format ascii; class volScalarField; "
        "object p; }\n"
        "dimensions [1 -1 -2 0 0];\n"
    )
    d = Dictionary(text)
    assert parse_dimensions(d["dimensions"]) == DimensionSet(
        1, -1, -2, 0, 0, 0, 0
    )


def test_dimension_bracket_renders_numeric_bracket():
    bracket = dimension_bracket(DimensionSet(0, 1, -1, 0, 0, 0, 0))
    assert str(bracket) == "[0 1 -1 0 0 0 0]"


def test_dimension_bracket_round_trips_through_parse_dimensions():
    dims = DimensionSet(1, -1, -2, 0, 0, 0, 0)
    text = (
        "FoamFile { version 2.0; format ascii; class volScalarField; "
        "object p; }\n"
        f"dimensions {dimension_bracket(dims)};\n"
    )
    d = Dictionary(text)
    assert parse_dimensions(d["dimensions"]) == dims


def test_dimensioned_renders_value_and_bracket():
    value = dimensioned(0.3, DimensionSet(0, 1, -1, 0, 0, 0, 0))
    assert str(value) == "0.3 [0 1 -1 0 0 0 0]"


def test_ascii_nonuniform_declared_count_too_large_raises():
    # Count mismatch: the List backtracks, the entry re-parses as a
    # 4-item Compound, which read_field_value() rejects.
    text = (
        "FoamFile { version 2.0; format ascii; class volScalarField; "
        "object p; }\n"
        "internalField nonuniform List<scalar> 5 (1 2 3);\n"
    )
    d = Dictionary(text)
    with pytest.raises(ValueError, match="unrecognised field value form"):
        read_field_value(d["internalField"])


def test_ascii_nonuniform_declared_count_too_small_raises():
    text = (
        "FoamFile { version 2.0; format ascii; class volScalarField; "
        "object p; }\n"
        "internalField nonuniform List<scalar> 2 (1 2 3);\n"
    )
    d = Dictionary(text)
    with pytest.raises(ValueError, match="unrecognised field value form"):
        read_field_value(d["internalField"])


def test_field_read_at_matches_reading_the_resolved_path_directly():
    case_dir = FIXTURES / "cavity_ascii"
    via_time = Field.read_at(case_dir, 0.1, "U")
    direct = Field.read(case_dir / "0.1" / "U")
    assert via_time.dimensions == direct.dimensions
    assert via_time.internal_field == direct.internal_field


def test_field_read_at_raises_for_a_time_with_no_matching_directory():
    case_dir = FIXTURES / "cavity_ascii"
    with pytest.raises(FileNotFoundError, match="no time directory"):
        Field.read_at(case_dir, 0.5, "U")


def test_field_read_at_does_not_fall_back_to_an_earlier_time(tmp_path):
    # No backward search: "0.2" exists but has no "p", although an
    # earlier time does.
    (tmp_path / "0.1" / "p").parent.mkdir(parents=True)
    (tmp_path / "0.1" / "p").write_text(
        "FoamFile { version 2.0; format ascii; class volScalarField; "
        "object p; }\n"
        "dimensions [0 0 0 0 0 0 0];\n"
        "internalField uniform 0;\n"
        "boundaryField {}\n"
    )
    (tmp_path / "0.2").mkdir()
    assert resolve_time_instance(tmp_path, 0.2) == "0.2"
    with pytest.raises(FileNotFoundError, match="p"):
        Field.read_at(tmp_path, 0.2, "p")


def test_field_read_at_with_region_on_the_real_multiregion_fixture():
    # "fluid" and "film" both have 0/<region>/T; no cross-talk.
    case_dir = FIXTURES / "multiregion_cylinder"
    fluid_t = Field.read_at(case_dir, 0, "T", region="fluid")
    film_t = Field.read_at(case_dir, 0, "T", region="film")
    assert fluid_t.internal_field.uniform is True
    assert film_t.internal_field.uniform is True
    direct_fluid_t = Field.read(case_dir / "0" / "fluid" / "T")
    direct_film_t = Field.read(case_dir / "0" / "film" / "T")
    assert fluid_t.internal_field == direct_fluid_t.internal_field
    assert film_t.internal_field == direct_film_t.internal_field
