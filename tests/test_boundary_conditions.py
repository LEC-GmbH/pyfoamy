import subprocess
from pathlib import Path

import pytest

from pyfoamy import (
    AlphatJayatillekeWallFunction,
    AlphatWallFunction,
    Applicability,
    BoundaryCondition,
    BoundaryField,
    Calculated,
    Case,
    EntrainmentPressure,
    EpsilonWallFunction,
    ExternalTemperature,
    FixedFluxExtrapolatedPressure,
    FixedFluxPressure,
    FixedValue,
    FlowRateInletVelocity,
    Freestream,
    FreestreamPressure,
    FreestreamVelocity,
    GenericBoundaryCondition,
    InletOutlet,
    KqRWallFunction,
    MovingWallSlipVelocity,
    MovingWallVelocity,
    NoSlip,
    NutkWallFunction,
    NutUSpaldingWallFunction,
    OmegaWallFunction,
    PatchRole,
    PressureInletOutletVelocity,
    RotatingWallVelocity,
    Slip,
    TurbulentEpsilon,
    TurbulentInlet,
    TurbulentKineticEnergy,
    TurbulentOmega,
    UniformFixedValue,
    UniformInletOutlet,
    UniformTotalPressure,
    WaveTransmissive,
    ZeroGradient,
    patches_by_name,
    pressure_wall_patches,
    u_wall_patches,
)
from pyfoamy.parsing.grammar import (
    Bracket,
    Compound,
    Dictionary,
    Directive,
    KeywordEntry,
    Uniform,
    Variable,
    to_grammar_value,
)
from pyfoamy.parsing.polymesh import PolyMesh

_FIXTURE_MESH = (
    Path(__file__).parent / "fixtures" / "cavity_ascii" / "constant"
    / "polyMesh"
)


def test_zero_gradient():
    assert ZeroGradient() == {"type": "zeroGradient"}
    assert ZeroGradient.applicability is Applicability.ANY


def test_no_slip():
    assert NoSlip() == {"type": "noSlip"}
    assert NoSlip.applicability is Applicability.WALL


def test_fixed_value_scalar():
    bc = FixedValue(0)
    assert str(bc["type"]) == "fixedValue"
    assert str(bc["value"]) == "uniform 0"


def test_fixed_value_vector():
    bc = FixedValue((1, 0, 0))
    assert str(bc["type"]) == "fixedValue"
    assert str(bc["value"]) == "uniform (1 0 0)"


def test_calculated():
    bc = Calculated(0)
    assert str(bc["type"]) == "calculated"
    assert str(bc["value"]) == "uniform 0"


def test_fixed_flux_pressure():
    bc = FixedFluxPressure(0)
    assert str(bc["type"]) == "fixedFluxPressure"
    assert str(bc["value"]) == "uniform 0"


def test_fixed_flux_pressure_value_is_optional():
    assert FixedFluxPressure() == {"type": "fixedFluxPressure"}


def test_fixed_flux_extrapolated_pressure():
    bc = FixedFluxExtrapolatedPressure(0)
    assert str(bc["type"]) == "fixedFluxExtrapolatedPressure"
    assert str(bc["value"]) == "uniform 0"


def test_slip():
    assert Slip() == {"type": "slip"}


def test_moving_wall_velocity():
    bc = MovingWallVelocity((1, 0, 0))
    assert str(bc["type"]) == "movingWallVelocity"
    assert str(bc["value"]) == "uniform (1 0 0)"
    assert bc.applicability is Applicability.WALL


def test_moving_wall_slip_velocity():
    bc = MovingWallSlipVelocity((1, 0, 0))
    assert str(bc["type"]) == "movingWallSlipVelocity"
    assert str(bc["value"]) == "uniform (1 0 0)"


def test_k_q_r_wall_function():
    bc = KqRWallFunction(0.1)
    assert str(bc["type"]) == "kqRWallFunction"
    assert str(bc["value"]) == "uniform 0.1"


def test_epsilon_wall_function():
    bc = EpsilonWallFunction(0.01)
    assert str(bc["type"]) == "epsilonWallFunction"
    assert str(bc["value"]) == "uniform 0.01"


def test_omega_wall_function():
    bc = OmegaWallFunction(1)
    assert str(bc["type"]) == "omegaWallFunction"
    assert str(bc["value"]) == "uniform 1"


def test_nutk_wall_function():
    bc = NutkWallFunction(0)
    assert str(bc["type"]) == "nutkWallFunction"
    assert str(bc["value"]) == "uniform 0"


def test_nut_u_spalding_wall_function():
    bc = NutUSpaldingWallFunction(0)
    assert str(bc["type"]) == "nutUSpaldingWallFunction"
    assert str(bc["value"]) == "uniform 0"


_ARCHETYPES = ("wall", "movingWall", "slipNCC", "movingSlipNCC")
_ROLES = {
    PatchRole.WALL: ("wall",),
    PatchRole.MOVING_WALL: ("movingWall",),
    PatchRole.SLIP: ("slipNCC",),
    PatchRole.MOVING_SLIP: ("movingSlipNCC",),
    PatchRole.INLET: ("inlet",),
}


def test_u_wall_patches_has_all_four_archetypes():
    patches = patches_by_name(u_wall_patches((1, 0, 0)), _ROLES)
    assert set(patches) == set(_ARCHETYPES)


def test_u_wall_patches_static_types():
    patches = patches_by_name(u_wall_patches((1, 0, 0)), _ROLES)
    assert str(patches["wall"]["type"]) == "noSlip"
    assert str(patches["slipNCC"]["type"]) == "slip"


def test_u_wall_patches_moving_types_use_the_moving_value():
    patches = patches_by_name(u_wall_patches((1, 0, 0)), _ROLES)
    for key in ("movingWall", "movingSlipNCC"):
        assert str(patches[key]["value"]) == "uniform (1 0 0)"
    assert str(patches["movingWall"]["type"]) == "movingWallVelocity"
    assert str(patches["movingSlipNCC"]["type"]) == "movingWallSlipVelocity"


def test_pressure_wall_patches_has_all_four_archetypes():
    patches = patches_by_name(pressure_wall_patches(0), _ROLES)
    assert set(patches) == set(_ARCHETYPES)


def test_pressure_wall_patches_defaults_to_fixed_flux_pressure():
    patches = patches_by_name(pressure_wall_patches(0), _ROLES)
    for key in _ARCHETYPES:
        assert str(patches[key]["type"]) == "fixedFluxPressure"


def test_pressure_wall_patches_moving_wall_can_use_extrapolated():
    patches = patches_by_name(
        pressure_wall_patches(
        0, moving_wall_pressure="fixedFluxExtrapolatedPressure"
    ), _ROLES,
    )
    assert str(patches["movingWall"]["type"]) == (
        "fixedFluxExtrapolatedPressure"
    )
    # Everything else stays fixedFluxPressure - only the moving-wall
    # entry is affected by the choice.
    assert str(patches["wall"]["type"]) == "fixedFluxPressure"
    assert str(patches["movingSlipNCC"]["type"]) == "fixedFluxPressure"


def test_pressure_wall_patches_rejects_an_unknown_choice():
    with pytest.raises(ValueError, match="moving_wall_pressure"):
        pressure_wall_patches(0, moving_wall_pressure="bogus")


def test_generic_boundary_condition_sets_type_and_fields():
    bc = GenericBoundaryCondition("someExoticType", foo=1, bar="baz")
    assert bc == {"type": "someExoticType", "foo": 1, "bar": "baz"}
    assert bc.applicability is Applicability.ANY
    assert isinstance(bc, BoundaryCondition)


def test_generic_boundary_condition_accepts_explicit_applicability():
    bc = GenericBoundaryCondition(
        "someWallType", applicability=Applicability.WALL,
    )
    assert bc.applicability is Applicability.WALL


def test_inlet_outlet():
    bc = InletOutlet(0.5)
    assert str(bc["type"]) == "inletOutlet"
    assert str(bc["inletValue"]) == "uniform 0.5"
    assert bc.applicability is Applicability.INLET_OUTLET


def test_pressure_inlet_outlet_velocity_no_fields_by_default():
    bc = PressureInletOutletVelocity()
    assert bc == {"type": "pressureInletOutletVelocity"}


def test_pressure_inlet_outlet_velocity_optional_value():
    bc = PressureInletOutletVelocity((0, 0, 0))
    assert str(bc["value"]) == "uniform (0 0 0)"


def test_alphat_wall_function():
    bc = AlphatWallFunction(0)
    assert str(bc["type"]) == "compressible::alphatWallFunction"
    assert bc["Prt"] == to_grammar_value(0.85)
    assert str(bc["value"]) == "uniform 0"


def test_alphat_jayatilleke_wall_function_custom_prt():
    bc = AlphatJayatillekeWallFunction(0, prt=0.9)
    assert str(bc["type"]) == "compressible::alphatJayatillekeWallFunction"
    assert bc["Prt"] == to_grammar_value(0.9)


def test_turbulent_kinetic_energy_requires_intensity():
    bc = TurbulentKineticEnergy(0.05)
    assert str(bc["type"]) == "turbulentKineticEnergy"
    assert bc["intensity"] == to_grammar_value(0.05)
    assert "value" not in bc


def test_turbulent_kinetic_energy_accepts_variable_value():
    bc = TurbulentKineticEnergy(0.05, value=Variable("$internalField"))
    assert str(bc["value"]) == "$internalField"


def test_turbulent_epsilon():
    bc = TurbulentEpsilon(0.01)
    assert str(bc["type"]) == "turbulentEpsilon"
    assert bc["mixingLength"] == to_grammar_value(0.01)


def test_turbulent_omega():
    bc = TurbulentOmega(0.01)
    assert str(bc["type"]) == "turbulentOmega"
    assert bc["mixingLength"] == to_grammar_value(0.01)


def test_turbulent_inlet():
    bc = TurbulentInlet("internalField", 0.1, (1, 0, 0))
    assert str(bc["type"]) == "turbulentInlet"
    assert bc["referenceField"] == to_grammar_value("internalField")
    assert bc["fluctuationScale"] == to_grammar_value(0.1)
    assert str(bc["value"]) == "uniform (1 0 0)"


def test_freestream():
    bc = Freestream(0)
    assert str(bc["type"]) == "freestream"
    assert str(bc["freestreamValue"]) == "uniform 0"


def test_freestream_velocity():
    bc = FreestreamVelocity((1, 0, 0), (0, 0, 0))
    assert str(bc["type"]) == "freestreamVelocity"
    assert str(bc["freestreamValue"]) == "uniform (1 0 0)"
    assert str(bc["value"]) == "uniform (0 0 0)"


def test_freestream_pressure():
    bc = FreestreamPressure(101325)
    assert str(bc["type"]) == "freestreamPressure"
    assert str(bc["freestreamValue"]) == "uniform 101325"


def test_rotating_wall_velocity():
    bc = RotatingWallVelocity(
        origin=(0, 0, 0), axis=(0, 0, 1), omega=13.19, value=(0, 0, 0),
    )
    assert str(bc["type"]) == "rotatingWallVelocity"
    assert bc["origin"] == to_grammar_value((0, 0, 0))
    assert bc["axis"] == to_grammar_value((0, 0, 1))
    assert bc["omega"] == to_grammar_value(13.19)
    assert bc.applicability is Applicability.WALL


def test_rotating_wall_velocity_rpm():
    bc = RotatingWallVelocity(origin=(0, 0, 0), axis=(0, 0, 1), rpm=126)
    assert bc["rpm"] == to_grammar_value(126)
    assert "omega" not in bc


@pytest.mark.parametrize("speed", [{}, {"omega": 1, "rpm": 10}])
def test_rotating_wall_velocity_needs_one_of_omega_rpm(speed):
    with pytest.raises(ValueError, match="exactly one"):
        RotatingWallVelocity(origin=(0, 0, 0), axis=(0, 0, 1), **speed)


def test_rotating_wall_velocity_value_is_optional():
    bc = RotatingWallVelocity(origin=(0, 0, 0), axis=(0, 0, 1), omega=13.19)
    assert "value" not in bc


def test_uniform_inlet_outlet():
    bc = UniformInletOutlet(0.5)
    assert str(bc["type"]) == "uniformInletOutlet"
    assert bc["uniformInletValue"] == to_grammar_value(0.5)


def test_flow_rate_inlet_velocity_mass_flow_rate():
    bc = FlowRateInletVelocity(mass_flow_rate=Compound("constant", 0.1))
    assert str(bc["type"]) == "flowRateInletVelocity"
    assert str(bc["massFlowRate"]) == "constant 0.1"


def test_flow_rate_inlet_velocity_with_profile_and_rho_inlet():
    bc = FlowRateInletVelocity(
        mass_flow_rate=Compound("constant", 0.1),
        profile="turbulentBL", rho_inlet=1,
    )
    assert str(bc["profile"]) == "turbulentBL"
    assert bc["rhoInlet"] == to_grammar_value(1)


def test_flow_rate_inlet_velocity_requires_exactly_one():
    with pytest.raises(ValueError, match="exactly one"):
        FlowRateInletVelocity()
    with pytest.raises(ValueError, match="exactly one"):
        FlowRateInletVelocity(
            mass_flow_rate=1, volumetric_flow_rate=1,
        )


def test_external_temperature_ta_h():
    bc = ExternalTemperature(300, ta=Compound("constant", 290), h=8)
    assert str(bc["type"]) == "externalTemperature"
    assert str(bc["Ta"]) == "constant 290"
    assert str(bc["h"]) == "uniform 8"
    assert str(bc["value"]) == "uniform 300"


def test_external_temperature_q_alone():
    bc = ExternalTemperature(300, q=66919.2)
    assert str(bc["q"]) == "uniform 66919.2"
    assert "Ta" not in bc
    assert "h" not in bc


def test_external_temperature_q_capital_alone():
    bc = ExternalTemperature(300, Q=1000)
    assert str(bc["Q"]) == "uniform 1000"
    assert "Ta" not in bc
    assert "h" not in bc
    assert "q" not in bc


def test_external_temperature_function1_and_macro():
    bc = ExternalTemperature(
        Variable("$internalField"), q={"type": "table", "file": "q.dat"},
    )
    assert str(bc["q"]["type"]) == "table"
    assert str(bc["value"]) == "$internalField"


def test_external_temperature_requires_ta_and_h_together():
    with pytest.raises(ValueError, match="together"):
        ExternalTemperature(300, ta=Compound("constant", 290))


def test_external_temperature_requires_ta_h_or_q():
    with pytest.raises(ValueError, match="needs ta"):
        ExternalTemperature(300)


def test_uniform_total_pressure_compressible():
    bc = UniformTotalPressure(
        Compound("constant", 101325), psi="psi", gamma=1.4,
    )
    assert str(bc["psi"]) == "psi"
    assert bc["gamma"] == to_grammar_value(1.4)


def test_uniform_total_pressure_requires_psi_and_gamma_together():
    with pytest.raises(ValueError, match="together"):
        UniformTotalPressure(Compound("constant", 101325), gamma=1.4)


def test_uniform_total_pressure():
    bc = UniformTotalPressure(Compound("constant", 101325))
    assert str(bc["type"]) == "uniformTotalPressure"
    assert str(bc["p0"]) == "constant 101325"


def test_entrainment_pressure():
    bc = EntrainmentPressure(Variable("$internalField"))
    assert str(bc["type"]) == "entrainmentPressure"
    assert str(bc["p0"]) == "$internalField"
    assert bc.applicability is Applicability.INLET_OUTLET


def test_entrainment_pressure_optional_value():
    bc = EntrainmentPressure(0, value=0)
    assert str(bc["value"]) == "uniform 0"


def test_wave_transmissive():
    bc = WaveTransmissive("p", 1.3, 1e5, 0.3, value=1e5)
    assert str(bc["type"]) == "waveTransmissive"
    assert bc["field"] == to_grammar_value("p")
    assert bc["gamma"] == to_grammar_value(1.3)
    assert bc["fieldInf"] == to_grammar_value(1e5)
    assert bc["lInf"] == to_grammar_value(0.3)
    assert str(bc["value"]) == "uniform 100000"


def test_uniform_fixed_value_plain_constant():
    bc = UniformFixedValue(320)
    assert str(bc["type"]) == "uniformFixedValue"
    assert bc["uniformValue"] == to_grammar_value(320)


def test_uniform_fixed_value_function1():
    bc = UniformFixedValue({"type": "table", "values": [(0, 352.9)]})
    assert str(bc["uniformValue"]["type"]) == "table"


def test_boundary_field_rejects_a_plain_dict():
    with pytest.raises(TypeError, match="BoundaryCondition"):
        BoundaryField({"movingWall": {"type": "fixedValue"}})


def test_boundary_field_prepends_the_constraint_types_directive():
    d = BoundaryField({"movingWall": FixedValue(0)})
    assert isinstance(d[0], Directive)
    assert d[0].name == "#includeEtc"
    assert str(d[0].args) == '"caseDicts/setConstraintTypes"'
    assert isinstance(d[1], KeywordEntry)
    assert d[1].key.data == "movingWall"


def test_boundary_field_is_a_real_dictionary_subclass():
    # As a Dictionary subclass it is accepted by .set() unchanged.
    d = BoundaryField({"movingWall": FixedValue(0)})
    assert isinstance(d, Dictionary)
    outer = Dictionary()
    outer.set("boundaryField", d)
    assert outer["boundaryField"] is d


def test_boundary_field_round_trips_through_write_and_parse():
    d = BoundaryField({
        "movingWall": FixedValue((1, 0, 0)),
        "fixedWalls": NoSlip(),
    })
    wrapper = Dictionary()
    wrapper["boundaryField"] = d
    text = wrapper.as_text
    assert '#includeEtc "caseDicts/setConstraintTypes"' in text

    reparsed = Dictionary(text)
    bf = reparsed["boundaryField"]
    assert str(bf["movingWall"]["type"]) == "fixedValue"
    assert str(bf["fixedWalls"]["type"]) == "noSlip"


@pytest.mark.openfoam
def test_boundary_field_is_valid_per_foam_dictionary(tmp_path):
    case = Case(tmp_path / "myCase")
    case.write_dict(
        "0/U",
        {
            "dimensions": Bracket("velocity"),
            "internalField": Uniform((0, 0, 0)),
            "boundaryField": BoundaryField({
                "movingWall": FixedValue((1, 0, 0)),
                "fixedWalls": NoSlip(),
            }),
        },
        class_name="volVectorField",
    )

    path = tmp_path / "myCase" / "0" / "U"
    result = subprocess.run(
        [
            "foamDictionary", "-entry", "boundaryField/fixedWalls/type",
            "-value", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "noSlip"


def _fixture_mesh() -> PolyMesh:
    return PolyMesh.read(_FIXTURE_MESH)


def test_boundary_field_mesh_check_passes_for_matching_applicability():
    mesh = _fixture_mesh()
    # movingWall/fixedWalls are real "wall"-type patches in this
    # fixture - a WALL-only BC on either is not a mismatch.
    BoundaryField(
        {"movingWall": NoSlip(), "fixedWalls": NoSlip()}, mesh=mesh,
    )


def test_boundary_field_mesh_check_flags_wall_only_on_non_wall_patch():
    mesh = _fixture_mesh()
    with pytest.raises(ValueError, match="WALL-only"):
        BoundaryField({"frontAndBack": NoSlip()}, mesh=mesh)


def test_boundary_field_mesh_check_flags_inlet_outlet_on_wall_patch():
    mesh = _fixture_mesh()
    with pytest.raises(ValueError, match="non-wall-only"):
        BoundaryField({"movingWall": InletOutlet(0)}, mesh=mesh)


def test_boundary_field_mesh_check_is_opt_in():
    # Without mesh=, no validation happens at all - the same
    # mismatched BC/patch combination above is not an error here.
    BoundaryField({"movingWall": InletOutlet(0)})


def test_boundary_field_keeps_the_condition_class():
    bf = BoundaryField({"lid": MovingWallVelocity((1, 0, 0))})
    assert isinstance(bf["lid"], MovingWallVelocity)
    assert bf["lid"].applicability is Applicability.WALL


def test_boundary_field_copies_a_condition_shared_by_several_patches():
    bf = BoundaryField({PatchRole.WALL: FixedValue(1)}, roles={
        PatchRole.WALL: ("left", "right"),
    })
    bf["left"]["value"] = Uniform(2)
    assert bf["left"] is not bf["right"]
    assert str(bf["right"]["value"]) == "uniform 1"


def test_boundary_field_role_keys_need_roles():
    with pytest.raises(TypeError, match="roles"):
        BoundaryField(u_wall_patches((1, 0, 0)))


def test_patches_by_name_drops_roles_without_patches():
    patches = patches_by_name(
        {PatchRole.WALL: NoSlip(), PatchRole.SLIP: Slip(), "top": NoSlip()},
        {PatchRole.WALL: ("a", "b")},
    )
    assert set(patches) == {"a", "b", "top"}
