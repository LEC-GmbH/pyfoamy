import pytest

from pyfoamy.models.thermo import ThermoTypeSpec


def _spec(energy):
    return ThermoTypeSpec(
        type="hePsiThermo",
        mixture="pureMixture",
        transport="const",
        thermo="hConst",
        equation_of_state="perfectGas",
        energy=energy,
    )


def test_energy_field_enthalpy():
    assert _spec("sensibleEnthalpy").energy_field == "h"


def test_energy_field_internal_energy():
    assert _spec("sensibleInternalEnergy").energy_field == "e"


def test_energy_field_unknown_raises():
    with pytest.raises(ValueError, match="energy"):
        _spec("someOtherEnergyType").energy_field


def test_to_dict():
    spec = _spec("sensibleInternalEnergy")
    assert spec.to_dict() == {
        "type": "hePsiThermo",
        "mixture": "pureMixture",
        "transport": "const",
        "thermo": "hConst",
        "equationOfState": "perfectGas",
        "specie": "specie",
        "energy": "sensibleInternalEnergy",
    }


def test_specie_defaults_to_specie():
    spec = _spec("sensibleEnthalpy")
    assert spec.specie == "specie"
