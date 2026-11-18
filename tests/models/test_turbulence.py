import pytest

from pyfoamy import PatchRole, patches_by_name
from pyfoamy.models import TurbulenceSpec


def test_laminar_requires_no_fields():
    spec = TurbulenceSpec("laminar")
    assert spec.required_fields() == []


def test_laminar_rejects_a_model_name():
    with pytest.raises(ValueError, match="laminar"):
        TurbulenceSpec("laminar", "kOmegaSST")


def test_ras_requires_a_model_name():
    with pytest.raises(ValueError):
        TurbulenceSpec("RAS")


def test_unknown_simulation_type_raises():
    with pytest.raises(ValueError, match="simulation_type"):
        TurbulenceSpec("DES")


def test_unknown_ras_model_raises():
    with pytest.raises(ValueError, match="kFooBar"):
        TurbulenceSpec("RAS", "kFooBar")


@pytest.mark.parametrize(
    "model,expected",
    [
        ("kEpsilon", ["nut", "k", "epsilon"]),
        ("realizableKE", ["nut", "k", "epsilon"]),
        ("RNGkEpsilon", ["nut", "k", "epsilon"]),
        ("LaunderSharmaKE", ["nut", "k", "epsilon"]),
        ("kOmega", ["nut", "k", "omega"]),
        ("kOmega2006", ["nut", "k", "omega"]),
        ("kOmegaSST", ["nut", "k", "omega"]),
        (
            "kOmegaSSTLM",
            ["nut", "k", "omega", "ReThetat", "gammaInt"],
        ),
        ("v2f", ["nut", "k", "epsilon", "v2", "f"]),
        ("SpalartAllmaras", ["nut", "nuTilda"]),
    ],
)
def test_ras_model_required_fields(model, expected):
    spec = TurbulenceSpec("RAS", model)
    assert spec.required_fields() == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("Smagorinsky", ["nut"]),
        ("WALE", ["nut"]),
        ("kEqn", ["nut", "k"]),
        ("dynamicKEqn", ["nut", "k"]),
    ],
)
def test_les_model_required_fields(model, expected):
    spec = TurbulenceSpec("LES", model)
    assert spec.required_fields() == expected


def test_laminar_momentum_transport_dict():
    spec = TurbulenceSpec("laminar")
    assert spec.momentum_transport_dict() == {"simulationType": "laminar"}


def test_ras_momentum_transport_dict():
    spec = TurbulenceSpec("RAS", "kOmegaSST")
    assert spec.momentum_transport_dict() == {
        "simulationType": "RAS",
        "RAS": {
            "model": "kOmegaSST",
            "turbulence": True,
            "printCoeffs": True,
        },
    }


def test_ras_momentum_transport_dict_with_coefficients():
    spec = TurbulenceSpec(
        "RAS", "kEpsilon", coefficients={"Cmu": 0.09, "C1": 1.44}
    )
    assert spec.momentum_transport_dict() == {
        "simulationType": "RAS",
        "RAS": {
            "model": "kEpsilon",
            "turbulence": True,
            "printCoeffs": True,
            "kEpsilon": {"Cmu": 0.09, "C1": 1.44},
        },
    }


def test_les_momentum_transport_dict_with_entries():
    delta = {"delta": "cubeRootVol", "cubeRootVolCoeffs": {"deltaCoeff": 1}}
    spec = TurbulenceSpec("LES", "WALE", entries=delta)
    assert spec.momentum_transport_dict()["LES"] == {
        "model": "WALE",
        "turbulence": True,
        "printCoeffs": True,
        **delta,
    }


def test_entries_disallowed_for_laminar():
    with pytest.raises(ValueError, match="entries"):
        TurbulenceSpec("laminar", entries={"delta": "cubeRootVol"})


def test_coefficients_disallowed_for_laminar():
    with pytest.raises(ValueError, match="coefficients"):
        TurbulenceSpec("laminar", coefficients={"Cmu": 0.09})


def test_les_momentum_transport_dict():
    spec = TurbulenceSpec("LES", "WALE")
    assert spec.momentum_transport_dict() == {
        "simulationType": "LES",
        "LES": {
            "model": "WALE",
            "turbulence": True,
            "printCoeffs": True,
        },
    }


_ARCHETYPES = ("wall", "movingWall", "slipNCC", "movingSlipNCC")
_ROLES = {
    PatchRole.WALL: ("wall",),
    PatchRole.MOVING_WALL: ("movingWall",),
    PatchRole.SLIP: ("slipNCC",),
    PatchRole.MOVING_SLIP: ("movingSlipNCC",),
    PatchRole.INLET: ("inlet",),
}


def _wall_patches(spec, values=None):
    return {
        name: patches_by_name(patches, _ROLES)
        for name, patches in spec.wall_patches(values).items()
    }


def test_wall_patches_rejects_laminar():
    spec = TurbulenceSpec("laminar")
    with pytest.raises(ValueError, match="laminar"):
        spec.wall_patches()


def test_wall_patches_rejects_v2f():
    spec = TurbulenceSpec("RAS", "v2f")
    with pytest.raises(ValueError, match="v2f"):
        spec.wall_patches()


@pytest.mark.parametrize(
    "model", ["kEpsilon", "realizableKE", "RNGkEpsilon", "LaunderSharmaKE"]
)
def test_wall_patches_k_epsilon_family(model):
    spec = TurbulenceSpec("RAS", model)
    patches = _wall_patches(spec, {"k": 0.1, "epsilon": 0.01})
    assert set(patches) == {"k", "epsilon", "nut"}
    assert set(patches["k"]) == set(_ARCHETYPES)
    assert set(patches["epsilon"]) == set(_ARCHETYPES)
    # nut is derived, so it has an "inlet" default; k/epsilon/omega
    # need case-specific inlet data.
    assert set(patches["nut"]) == set(_ARCHETYPES) | {"inlet"}
    assert str(patches["k"]["wall"]["type"]) == "kqRWallFunction"
    assert patches["k"]["slipNCC"] == {"type": "zeroGradient"}
    assert str(patches["epsilon"]["wall"]["type"]) == "epsilonWallFunction"
    assert str(patches["nut"]["wall"]["type"]) == "nutkWallFunction"
    assert str(patches["nut"]["slipNCC"]["type"]) == "calculated"
    assert patches["nut"]["inlet"] == patches["nut"]["slipNCC"]
    # Moving and static walls share wall functions (one "wall" group).
    assert patches["k"]["wall"] == patches["k"]["movingWall"]


@pytest.mark.parametrize(
    "model", ["kOmega", "kOmega2006", "kOmegaSST"]
)
def test_wall_patches_k_omega_family(model):
    spec = TurbulenceSpec("RAS", model)
    patches = _wall_patches(spec, {"k": 0.1, "omega": 1})
    assert set(patches) == {"k", "omega", "nut"}
    assert str(patches["omega"]["wall"]["type"]) == "omegaWallFunction"
    assert patches["omega"]["slipNCC"] == {"type": "zeroGradient"}


def test_wall_patches_kOmegaSSTLM_adds_transition_fields():
    spec = TurbulenceSpec("RAS", "kOmegaSSTLM")
    patches = _wall_patches(spec, {"k": 0.1, "omega": 1})
    assert set(patches) == {
        "k", "omega", "nut", "ReThetat", "gammaInt",
    }
    assert patches["ReThetat"]["wall"] == {"type": "zeroGradient"}
    assert patches["gammaInt"]["slipNCC"] == {"type": "zeroGradient"}


def test_wall_patches_spalart_allmaras():
    spec = TurbulenceSpec("RAS", "SpalartAllmaras")
    patches = _wall_patches(spec, {"nuTilda": 0.14})
    assert set(patches) == {"nuTilda", "nut"}
    assert str(patches["nuTilda"]["wall"]["type"]) == "fixedValue"
    assert str(patches["nut"]["wall"]["type"]) == "nutUSpaldingWallFunction"
    assert str(patches["nut"]["slipNCC"]["type"]) == "calculated"


@pytest.mark.parametrize("model", ["Smagorinsky", "WALE"])
def test_wall_patches_les_without_k(model):
    spec = TurbulenceSpec("LES", model)
    patches = _wall_patches(spec)
    assert set(patches) == {"nut"}
    assert patches["nut"]["wall"] == {"type": "zeroGradient"}
    assert patches["nut"]["slipNCC"] == {"type": "zeroGradient"}


@pytest.mark.parametrize("model", ["kEqn", "dynamicKEqn"])
def test_wall_patches_les_with_k(model):
    spec = TurbulenceSpec("LES", model)
    patches = _wall_patches(spec, {"k": 0})
    assert set(patches) == {"nut", "k"}
    assert str(patches["k"]["wall"]["type"]) == "fixedValue"
    assert patches["k"]["slipNCC"] == {"type": "zeroGradient"}


def test_wall_patches_defaults_missing_values_to_zero():
    spec = TurbulenceSpec("RAS", "kOmegaSST")
    patches = _wall_patches(spec)
    assert str(patches["k"]["wall"]["value"]) == "uniform 0"
    assert str(patches["omega"]["wall"]["value"]) == "uniform 0"
