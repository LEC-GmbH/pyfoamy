import subprocess

import pytest

from pyfoamy import (
    Case,
    default_field,
    default_wall_boundary_field,
    FixedValue,
    NoSlip,
    PatchRole,
    ZeroGradient,
    patches_by_name,
)
from pyfoamy.fields import FieldFile, FieldTemplate

_ROLES = {
    PatchRole.WALL: ("wall",),
    PatchRole.MOVING_WALL: ("movingWall",),
    PatchRole.SLIP: ("slipNCC",),
    PatchRole.MOVING_SLIP: ("movingSlipNCC",),
    PatchRole.INLET: ("inlet",),
}


def test_default_field_unknown_name_raises():
    with pytest.raises(KeyError, match="U"):
        default_field("not_a_real_field")


@pytest.mark.parametrize(
    "name,expected_class,expected_dims",
    [
        ("U", "volVectorField", "[velocity]"),
        ("p", "volScalarField", "[pressure]"),
        ("p_rgh", "volScalarField", "[pressure]"),
        ("T", "volScalarField", "[temperature]"),
        ("k", "volScalarField", "[turbulentKineticEnergy]"),
        ("epsilon", "volScalarField", "[turbulentEpsilon]"),
        ("omega", "volScalarField", "[turbulentOmega]"),
        ("nut", "volScalarField", "[turbulentViscosity]"),
        ("nuTilda", "volScalarField", "[turbulentViscosity]"),
        ("alphat", "volScalarField", "[dynamicDiffusivity]"),
        ("v2", "volScalarField", "[turbulentKineticEnergy]"),
        ("f", "volScalarField", "[rate]"),
        ("ReThetat", "volScalarField", "[]"),
        ("gammaInt", "volScalarField", "[]"),
        ("b", "volScalarField", "[]"),
        ("Xi", "volScalarField", "[]"),
        ("Su", "volScalarField", "[velocity]"),
        ("egr", "volScalarField", "[]"),
        ("ft", "volScalarField", "[]"),
        ("fu", "volScalarField", "[]"),
    ],
)
def test_default_field_registered_names(
    name, expected_class, expected_dims
):
    template = default_field(name)
    assert template.class_name == expected_class
    assert template.dimensions == expected_dims


def test_to_dict_uses_default_internal_value():
    content = default_field("T").to_dict()
    assert str(content["internalField"]) == "uniform 300"


def test_to_dict_override_internal_value():
    content = default_field("k").to_dict(internal_value=0.05)
    assert str(content["internalField"]) == "uniform 0.05"


def test_to_dict_vector_internal_value():
    content = default_field("U").to_dict()
    assert str(content["internalField"]) == "uniform (0 0 0)"


def test_field_template_accepts_short_form_class_and_dimensions():
    template = FieldTemplate("scalar", "temperature", 300)
    assert template.class_name == "volScalarField"
    assert template.dimensions == "[temperature]"


def test_field_template_accepts_already_full_class_and_dimensions():
    template = FieldTemplate("volScalarField", "[temperature]", 300)
    assert template.class_name == "volScalarField"
    assert template.dimensions == "[temperature]"


def test_field_template_with_no_patches_has_no_default_boundary_field():
    template = default_field("U")
    assert template.patches == {}
    assert "boundaryField" not in template.to_dict()


def test_field_template_boundary_field_zero_gradient_everywhere():
    patches = patches_by_name(default_field("T").boundary_field(), _ROLES)
    assert set(patches) == {
        "wall", "movingWall", "slipNCC", "movingSlipNCC",
    }
    for bc in patches.values():
        assert str(bc["type"]) == "zeroGradient"


def test_field_template_boundary_field_threads_internal_value():
    patches = patches_by_name(default_field("alphat").boundary_field(), _ROLES)
    assert str(patches["wall"]["type"]) == "compressible::alphatWallFunction"
    assert str(patches["wall"]["value"]) == "uniform 0"
    assert str(patches["inlet"]["type"]) == "calculated"


def test_field_template_boundary_field_value_override():
    patches = patches_by_name(
        default_field("alphat").boundary_field(value=0.5), _ROLES,
    )
    assert str(patches["wall"]["value"]) == "uniform 0.5"


def test_field_template_boundary_field_prebuilt_instance_ignores_value():
    # b's "inlet" entry is a pre-built UniformInletOutlet(1) instance,
    # not a class - overriding internal_value must not affect it.
    patches = patches_by_name(
        default_field("b").boundary_field(value=0), _ROLES,
    )
    assert patches["inlet"]["uniformInletValue"] == 1


def test_to_dict_includes_boundary_field_when_patches_declared():
    content = default_field("T").to_dict(roles=_ROLES)
    assert str(content["boundaryField"]["wall"]["type"]) == (
        "zeroGradient"
    )


def test_default_wall_boundary_field_unregistered_name_is_generic_scalar():
    patches = patches_by_name(default_wall_boundary_field("CH4"), _ROLES)
    assert set(patches) == {
        "wall", "movingWall", "slipNCC", "movingSlipNCC",
    }
    assert all(
        str(bc["type"]) == "zeroGradient" for bc in patches.values()
    )


def test_default_wall_boundary_field_alphat_matches_template():
    from_template = patches_by_name(
        default_field("alphat").boundary_field(), _ROLES,
    )
    via_dispatcher = patches_by_name(
        default_wall_boundary_field("alphat"), _ROLES,
    )
    assert from_template == via_dispatcher


def test_field_file_set_bc_adds_a_new_patch():
    field = FieldFile(default_field("T").to_dict(roles=_ROLES))
    field.set_bc("inlet1", FixedValue(310))
    assert str(field["boundaryField"]["inlet1"]["type"]) == (
        "fixedValue"
    )
    # The template's own default wall entry is untouched.
    assert str(field["boundaryField"]["wall"]["type"]) == (
        "zeroGradient"
    )


def test_field_file_set_bc_creates_boundary_field_if_missing():
    field = FieldFile(default_field("U").to_dict())
    assert "boundaryField" not in field
    field.set_bc("wall", NoSlip())
    assert str(field["boundaryField"]["wall"]["type"]) == "noSlip"


def test_field_file_set_bc_rejects_a_plain_dict():
    field = FieldFile(default_field("T").to_dict())
    with pytest.raises(TypeError, match="BoundaryCondition"):
        field.set_bc("inlet1", {"type": "fixedValue"})


def test_write_field_via_case(tmp_path):
    case = Case(tmp_path / "myCase")
    template = default_field("U")
    case.write_dict(
        "0/U",
        {
            **template.to_dict(),
            "boundaryField": {
                "movingWall": FixedValue((1, 0, 0)),
                "fixedWalls": NoSlip(),
            },
        },
        class_name=template.class_name,
    )

    written = case.read_dict("0/U")
    assert str(written["dimensions"]) == "[velocity]"
    assert (
        str(written["boundaryField"]["fixedWalls"]["type"])
        == "noSlip"
    )


@pytest.mark.openfoam
def test_written_field_is_valid_per_foam_dictionary(tmp_path):
    case = Case(tmp_path / "myCase")
    template = default_field("k")
    case.write_dict(
        "0/k",
        {
            **template.to_dict(),
            "boundaryField": {
                "movingWall": ZeroGradient(),
                "fixedWalls": ZeroGradient(),
            },
        },
        class_name=template.class_name,
    )

    path = tmp_path / "myCase" / "0" / "k"
    result = subprocess.run(
        ["foamDictionary", "-entry", "dimensions", "-value", str(path)],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "[ turbulentKineticEnergy]"
