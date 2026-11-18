"""Builds a case per solver setup on the cavity mesh and runs it briefly."""

import shutil
from pathlib import Path

import pytest

from pyfoamy import (
    BoundaryField,
    Calculated,
    PatchRole,
    default_wall_boundary_field,
    patches_by_name,
    pressure_wall_patches,
    u_wall_patches,
)
from pyfoamy.models import (
    FluidSetup,
    IncompressibleFluidSetup,
    LinearSolver,
    MulticomponentFluidSetup,
    SuModelSpec,
    ThermoTypeSpec,
    TurbulenceSpec,
    XiFluidSetup,
    XiModelSpec,
    xi_fluid_physical_properties_extra,
)
from pyfoamy.parsing.grammar import (
    AnonymousListEntry, Dictionary, DimensionSet, Word, dimensioned,
)

_FIXTURE_MESH = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "cavity_ascii" / "constant" / "polyMesh"
)

P_SOLVER = LinearSolver("multigrid", tolerance=1e-7, rel_tol=0.01)
MOMENTUM_SOLVER = LinearSolver(
    "conjugate_gradient", tolerance=1e-6, rel_tol=0.1
)

_ROLES = {
    PatchRole.WALL: ("wall",),
    PatchRole.MOVING_WALL: ("movingWall",),
    PatchRole.SLIP: ("slipNCC",),
    PatchRole.MOVING_SLIP: ("movingSlipNCC",),
    PatchRole.INLET: ("inlet",),
}


def _add_wall_scalar_field(case, name, **kwargs):
    # kwargs go to Case.add_field(). "wall" covers movingWall and
    # fixedWalls (group wall); unmatched archetype keys are unused.
    d = case.add_field(name, time=0, **kwargs)
    d["boundaryField"] = BoundaryField(
        default_wall_boundary_field(name), roles=_ROLES,
    )


def _add_velocity_field(case, moving_wall_value):
    d = case.add_field("U", time=0)
    d["boundaryField"] = BoundaryField(
        u_wall_patches(moving_wall_value), roles=_ROLES,
    )


def _add_pressure_field(case, value, **kwargs):
    # kwargs go to Case.add_field(), e.g. for kinematic pressure.
    d = case.add_field("p", time=0, internal_value=value, **kwargs)
    d["boundaryField"] = BoundaryField(
        pressure_wall_patches(value), roles=_ROLES,
    )


def _add_buoyant_pressure_fields(case):
    # p is derived from p_rgh, never solved, so "calculated".
    p = case.add_field("p", time=0)
    p["boundaryField"] = BoundaryField({"wall": Calculated(0)})
    p_rgh = case.add_field("p_rgh", time=0)
    p_rgh["boundaryField"] = BoundaryField(
        pressure_wall_patches(0), roles=_ROLES,
    )


def _attach_mesh_and_run(case, root):
    case.write()
    shutil.copytree(_FIXTURE_MESH, root / "constant" / "polyMesh")
    exit_code = case.run("foamRun")
    assert exit_code == 0
    assert (root / "2").is_dir()


def _attach_mesh_with_slip_moving_wall_and_run(case, root):
    # Rotors often get a slip moving wall: put "movingWall" into group
    # "movingSlipNCC" to exercise group matching.
    case.write()
    mesh_dir = root / "constant" / "polyMesh"
    shutil.copytree(_FIXTURE_MESH, mesh_dir)

    boundary_path = mesh_dir / "boundary"
    boundary = Dictionary(boundary_path.read_text())
    patch_list = next(
        n for n in boundary if isinstance(n, AnonymousListEntry)
    )
    in_groups = patch_list.body["movingWall"]["inGroups"]
    for i, item in enumerate(in_groups):
        if str(item) == "wall":
            in_groups[i] = Word("movingSlipNCC")
    boundary_path.write_text(boundary.as_text)

    exit_code = case.run("foamRun")
    assert exit_code == 0
    assert (root / "2").is_dir()


@pytest.mark.openfoam
def test_incompressible_fluid_end_to_end(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    root = tmp_path / "myCase"
    case = setup.build(
        root,
        ddt_order="steady",
        div_order="firstOrder",
        p_solver=P_SOLVER,
        momentum_solver=MOMENTUM_SOLVER,
        pressure_reference=(0, 0),
        start_time=0, end_time=2, delta_t=1,
        write_control="timeStep", write_interval=1,
    )

    _add_velocity_field(case, (1, 0, 0))
    # Kinematic "p", not the registry's [pressure].
    _add_pressure_field(
        case, 0, type="scalar", dimensions="[kinematicPressure]",
    )

    _attach_mesh_and_run(case, root)


@pytest.mark.openfoam
def test_incompressible_fluid_slip_wall_end_to_end(tmp_path):
    turbulence = TurbulenceSpec("RAS", "kEpsilon")
    setup = IncompressibleFluidSetup(nu=1e-05, turbulence=turbulence)
    root = tmp_path / "myCase"
    case = setup.build(
        root,
        ddt_order="steady",
        div_order="firstOrder",
        p_solver=P_SOLVER,
        momentum_solver=MOMENTUM_SOLVER,
        pressure_reference=(0, 0),
        start_time=0, end_time=2, delta_t=1,
        write_control="timeStep", write_interval=1,
    )

    # No literal "movingWall" key: it would win over the group entry.
    u = case.add_field("U", time=0)
    u_patches = patches_by_name(u_wall_patches((0, 0, 1)), _ROLES)
    u["boundaryField"] = BoundaryField({
        "wall": u_patches["wall"],
        "movingSlipNCC": u_patches["movingSlipNCC"],
    })
    _add_pressure_field(
        case, 0, type="scalar", dimensions="[kinematicPressure]",
    )
    turbulence_values = {"k": 0.1, "epsilon": 0.01}
    for name, patches in turbulence.wall_patches(turbulence_values).items():
        patches = patches_by_name(patches, _ROLES)
        d = case.add_field(
            name, time=0, internal_value=turbulence_values.get(name, 0)
        )
        d["boundaryField"] = BoundaryField({
            "wall": patches["wall"],
            "movingSlipNCC": patches["movingSlipNCC"],
        })

    _attach_mesh_with_slip_moving_wall_and_run(case, root)


@pytest.mark.openfoam
def test_fluid_end_to_end(tmp_path):
    thermo_type = ThermoTypeSpec(
        type="hePsiThermo",
        mixture="pureMixture",
        transport="const",
        thermo="hConst",
        equation_of_state="perfectGas",
        energy="sensibleInternalEnergy",
    )
    mixture_data = {
        "specie": {"molWeight": 28.9},
        "thermodynamics": {"Cp": 1005, "hf": 0},
        "transport": {"mu": 1.82e-05, "Pr": 0.71},
    }
    setup = FluidSetup(thermo_type, mixture_data)
    root = tmp_path / "myCase"
    case = setup.build(
        root,
        ddt_order="steady",
        div_order="firstOrder",
        p_solver=P_SOLVER,
        momentum_solver=MOMENTUM_SOLVER,
        start_time=0, end_time=2, delta_t=1,
        write_control="timeStep", write_interval=1,
    )

    _add_velocity_field(case, (1, 0, 0))
    _add_pressure_field(case, 1e5)
    _add_wall_scalar_field(case, "T")

    _attach_mesh_and_run(case, root)


@pytest.mark.openfoam
def test_fluid_buoyant_end_to_end(tmp_path):
    thermo_type = ThermoTypeSpec(
        type="heRhoThermo",
        mixture="pureMixture",
        transport="const",
        thermo="hConst",
        equation_of_state="Boussinesq",
        energy="sensibleEnthalpy",
    )
    mixture_data = {
        "specie": {"molWeight": 28.9},
        "equationOfState": {"rho0": 1, "T0": 300, "beta": 3e-03},
        "thermodynamics": {"Cp": 1000, "hf": 0},
        "transport": {"mu": 1e-05, "Pr": 0.7},
    }
    setup = FluidSetup(thermo_type, mixture_data, buoyant=True)
    root = tmp_path / "myCase"
    case = setup.build(
        root,
        ddt_order="steady",
        div_order="firstOrder",
        p_solver=P_SOLVER,
        momentum_solver=MOMENTUM_SOLVER,
        gravity=(0, -9.81, 0),
        pressure_reference=(0, 0),
        # Without under-relaxation it diverges within a few iterations.
        relaxation_factors={
            "fields": {"p_rgh": 0.7},
            "equations": {"U": 0.2, "h": 0.6},
        },
        start_time=0, end_time=2, delta_t=1,
        write_control="timeStep", write_interval=1,
    )

    _add_velocity_field(case, (1, 0, 0))
    _add_buoyant_pressure_fields(case)
    _add_wall_scalar_field(case, "T")

    _attach_mesh_and_run(case, root)


@pytest.mark.openfoam
def test_multicomponent_fluid_end_to_end(tmp_path):

    def specie(mol_weight, t_low, t_high, t_common, high_cp, low_cp):
        return {
            "specie": {"molWeight": mol_weight},
            "thermodynamics": {
                "Tlow": t_low, "Thigh": t_high, "Tcommon": t_common,
                "highCpCoeffs": high_cp, "lowCpCoeffs": low_cp,
            },
            "transport": {"As": 1.67212e-06, "Ts": 170.672},
        }

    # JANAF data from tutorials/multicomponentFluid/simplifiedSiwek.
    species = {
        "CH4": specie(
            16.043, 0, 5000, 1000,
            [0.0748515, 0.0133909, -5.73286e-06, 1.22293e-09,
             -1.01815e-13, -9468.34, 18.4373],
            [5.14988, -0.013671, 4.91801e-05, -4.84743e-08,
             1.66694e-11, -10246.6, -4.6413],
        ),
        "O2": specie(
            31.9988, 0, 5000, 1000,
            [3.28254, 0.00148309, -7.57967e-07, 2.09471e-10,
             -2.16718e-14, -1088.46, 5.45323],
            [3.78246, -0.00299673, 9.8473e-06, -9.6813e-09,
             3.24373e-12, -1063.94, 3.65768],
        ),
        "N2": specie(
            28.0134, 0, 5000, 1000,
            [2.92664, 0.00148798, -5.68476e-07, 1.0097e-10,
             -6.75335e-15, -922.798, 5.98053],
            [3.29868, 0.00140824, -3.96322e-06, 5.64152e-09,
             -2.44485e-12, -1020.9, 3.95037],
        ),
    }
    thermo_type = ThermoTypeSpec(
        type="hePsiThermo",
        mixture="multicomponentMixture",
        transport="sutherland",
        thermo="janaf",
        equation_of_state="perfectGas",
        energy="sensibleEnthalpy",
    )
    setup = MulticomponentFluidSetup(thermo_type, "N2", species)
    root = tmp_path / "myCase"
    case = setup.build(
        root,
        ddt_order="steady",
        div_order="firstOrder",
        p_solver=P_SOLVER,
        momentum_solver=MOMENTUM_SOLVER,
        start_time=0, end_time=2, delta_t=1,
        write_control="timeStep", write_interval=1,
    )

    _add_velocity_field(case, (1, 0, 0))
    _add_pressure_field(case, 1e5)
    _add_wall_scalar_field(case, "T")
    for name, value in (("CH4", 0.0), ("O2", 0.23), ("N2", 0.77)):
        _add_wall_scalar_field(
            case, name, type="scalar", dimensions="[]",
            internal_value=value,
        )

    _attach_mesh_and_run(case, root)


@pytest.mark.openfoam
def test_xi_fluid_homogeneous_end_to_end(tmp_path):
    thermo_type = ThermoTypeSpec(
        type="ignoredForXiFluid",
        mixture="ignoredForXiFluid",
        transport="const",
        thermo="hConst",
        equation_of_state="perfectGas",
        energy="sensibleEnthalpy",
    )
    # From tutorials/XiFluid/1Dlaminar physicalProperties.{u,b}.
    u_extra = {
        "stoichiometricAirFuelMassRatio": 15.675,
        "Phi": 1,
        "reactants": {
            "specie": {"molWeight": 29.4649},
            "thermodynamics": {"Cp": 1000, "hf": 1700e3},
            "transport": {"mu": 1e-5, "Pr": 1},
        },
    }
    b_extra = {
        "products": {
            "specie": {"molWeight": 28.3233},
            "thermodynamics": {"Cp": 1000, "hf": 0},
            "transport": {"mu": 1e-5, "Pr": 1},
        },
    }
    # A bare number would be dimensionless, which the solver rejects.
    su_model = SuModelSpec(
        "unstrained",
        {
            "unstrainedLaminarFlameSpeed": {
                "model": "constant",
                "Su": dimensioned(
                    0.3, DimensionSet(0, 1, -1, 0, 0, 0, 0)
                ),
            },
        },
    )
    xi_model = XiModelSpec("uniformConstant", xi_value=1)

    setup = XiFluidSetup(
        thermo_type, "homogeneous", u_extra, b_extra, su_model, xi_model,
    )
    root = tmp_path / "myCase"
    case = setup.build(
        root,
        ddt_order="steady",
        div_order="firstOrder",
        p_solver=P_SOLVER,
        momentum_solver=MOMENTUM_SOLVER,
        start_time=0, end_time=2, delta_t=1,
        write_control="timeStep", write_interval=1,
    )

    _add_velocity_field(case, (1, 0, 0))
    _add_pressure_field(case, 1e5)
    _add_wall_scalar_field(case, "T", path="T.u")
    _add_wall_scalar_field(case, "T", path="T.b")
    # b=0 (unburnt), matching T.u == T.b; b=1 without a temperature
    # rise diverges.
    _add_wall_scalar_field(case, "b", internal_value=0)
    _add_wall_scalar_field(case, "Xi")

    _attach_mesh_and_run(case, root)


@pytest.mark.openfoam
def test_xi_fluid_inhomogeneous_egr_end_to_end(tmp_path):
    thermo_type = ThermoTypeSpec(
        type="ignoredForXiFluid",
        mixture="ignoredForXiFluid",
        transport="const",
        thermo="hConst",
        equation_of_state="perfectGas",
        energy="sensibleEnthalpy",
    )
    # Same body for uInhomogeneous(EGR) and bInhomogeneous; only the
    # mixture type and the "egr" field differ.
    physical_properties_extra = {
        "stoichiometricAirFuelMassRatio": 15.0336,
        "fuel": {
            "specie": {"molWeight": 114.23},
            "thermodynamics": {"Cp": 1000, "hf": 1700e3},
            "transport": {"mu": 1e-5, "Pr": 1},
        },
        "oxidant": {
            "specie": {"molWeight": 28.8504},
            "thermodynamics": {"Cp": 1000, "hf": 0},
            "transport": {"mu": 1e-5, "Pr": 1},
        },
        "products": {
            "specie": {"molWeight": 28.6068},
            "thermodynamics": {"Cp": 1000, "hf": 0},
            "transport": {"mu": 1e-5, "Pr": 1},
        },
    }
    su_model = SuModelSpec(
        "unstrained",
        {
            "unstrainedLaminarFlameSpeed": {
                "model": "constant",
                "Su": dimensioned(
                    0.3, DimensionSet(0, 1, -1, 0, 0, 0, 0)
                ),
            },
        },
    )
    xi_model = XiModelSpec("uniformConstant", xi_value=1)

    setup = XiFluidSetup(
        thermo_type,
        "inhomogeneousEGR",
        physical_properties_extra,
        physical_properties_extra,
        su_model,
        xi_model,
    )
    root = tmp_path / "myCase"
    case = setup.build(
        root,
        ddt_order="steady",
        div_order="firstOrder",
        p_solver=P_SOLVER,
        momentum_solver=MOMENTUM_SOLVER,
        start_time=0, end_time=2, delta_t=1,
        write_control="timeStep", write_interval=1,
    )

    _add_velocity_field(case, (1, 0, 0))
    _add_pressure_field(case, 1e5)
    _add_wall_scalar_field(case, "T", path="T.u")
    _add_wall_scalar_field(case, "T", path="T.b")
    _add_wall_scalar_field(case, "b", internal_value=0)
    _add_wall_scalar_field(case, "Xi")
    # Mass fractions, zero everywhere (cold start, b=0).
    _add_wall_scalar_field(case, "fu", path="fu.u")
    _add_wall_scalar_field(case, "egr", path="egr.u")
    _add_wall_scalar_field(case, "ft", path="ft.b")

    _attach_mesh_and_run(case, root)


@pytest.mark.openfoam
def test_xi_fluid_inhomogeneous_egr_natural_gas_end_to_end(tmp_path):
    # As above, but physicalProperties.{u,b} derived from a mechanism
    # via ThermoDatabase; inputs are fuel, oxidant and phi only.
    pytest.importorskip("cantera")
    pytest.importorskip("numpy")
    from pyfoamy.cantera.species_data import ThermoDatabase

    mechanism = (
        Path(__file__).resolve().parent.parent.parent / "converted.yml"
    )
    if not mechanism.is_file():
        pytest.skip("converted.yml not present in the repo root")
    # One thermo/transport model per phase (OpenFOAM's thermoType), so
    # it is set on the database, not per species.
    db = ThermoDatabase(
        str(mechanism), thermo_model="janaf",
        transport_model="logPolynomial",
    )

    fuel = {
        "CH4": 0.9, "C2H6": 0.05, "C3H8": 0.02, "CO2": 0.02, "N2": 0.01,
    }
    oxidant = db.mechanism.air(basis="mole")
    equivalence_ratio = 0.9

    physical_properties_extra = xi_fluid_physical_properties_extra(
        db.mechanism.stoichiometric_air_fuel_ratio(
            fuel, oxidant, basis="mole",
        ),
        db.lump(fuel, basis="mole"),
        db.lump(oxidant, basis="mole"),
        db.lump(
            db.mechanism.complete_combustion_products(
                fuel, oxidant, basis="mole",
                equivalence_ratio=equivalence_ratio,
            ),
            basis="mass",
        ),
    )

    thermo_type = ThermoTypeSpec(
        type="ignoredForXiFluid",
        mixture="ignoredForXiFluid",
        equation_of_state="perfectGas",
        energy="sensibleEnthalpy",
        **db.thermo_type_entries,
    )
    su_model = SuModelSpec(
        "unstrained",
        {
            "unstrainedLaminarFlameSpeed": {
                "model": "constant",
                "Su": dimensioned(
                    0.3, DimensionSet(0, 1, -1, 0, 0, 0, 0)
                ),
            },
        },
    )
    xi_model = XiModelSpec("uniformConstant", xi_value=1)

    setup = XiFluidSetup(
        thermo_type,
        "inhomogeneousEGR",
        physical_properties_extra,
        physical_properties_extra,
        su_model,
        xi_model,
    )
    root = tmp_path / "myCase"
    case = setup.build(
        root,
        ddt_order="steady",
        div_order="firstOrder",
        p_solver=P_SOLVER,
        momentum_solver=MOMENTUM_SOLVER,
        start_time=0, end_time=2, delta_t=1,
        write_control="timeStep", write_interval=1,
    )

    _add_velocity_field(case, (1, 0, 0))
    _add_pressure_field(case, 1e5)
    _add_wall_scalar_field(case, "T", path="T.u")
    _add_wall_scalar_field(case, "T", path="T.b")
    _add_wall_scalar_field(case, "b", internal_value=0)
    _add_wall_scalar_field(case, "Xi")
    _add_wall_scalar_field(case, "fu", path="fu.u")
    _add_wall_scalar_field(case, "egr", path="egr.u")
    _add_wall_scalar_field(case, "ft", path="ft.b")

    _attach_mesh_and_run(case, root)
