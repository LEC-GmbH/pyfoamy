"""Writes a demo XiFluid case: premixed natural gas/air in a wedge duct.

All `physicalProperties.{u,b}` data and the laminar flame speed come
from a mechanism via `pyfoamy.cantera`. Simplifications: one flame
speed point (isobaric, isothermal T.u), no ignition (nothing burns as
shipped), placeholder T.b, laminar flow. The wedge follows
`tutorials/multicomponentFluid/SandiaD_LTS`.

    python examples/create_xi_fluid.py
    cd xi_fluid_inhomogeneous && blockMesh && foamRun
"""

import math
from pathlib import Path

from pyfoamy import (
    Case,
    FixedFluxPressure,
    FixedValue,
    InletOutlet,
    NoSlip,
    PressureInletOutletVelocity,
    UniformTotalPressure,
    ZeroGradient,
)
from pyfoamy.cantera.laminar_flame_speed import LaminarFlameSpeedDatabase
from pyfoamy.cantera.species_data import ThermoDatabase
from pyfoamy.models import (
    LinearSolver,
    SuModelSpec,
    ThermoTypeSpec,
    TurbulenceSpec,
    XiFluidSetup,
    XiModelSpec,
    xi_fluid_physical_properties_extra,
)
from pyfoamy.parsing.grammar import Compound, DimensionSet, String, dimensioned

ROOT = Path("xi_fluid_inhomogeneous")
# Shipped with Cantera.
MECHANISM = "gri30.yaml"

# --- operating conditions (the only real "user inputs") ---
FUEL = {  # a representative natural-gas blend, mole fractions
    "CH4": 0.9, "C2H6": 0.05, "C3H8": 0.02, "CO2": 0.02, "N2": 0.01,
}
EQUIVALENCE_RATIO = 0.8  # fuel-lean
P_OPERATING = 1e5  # Pa - isobaric assumption
T_UNBURNT = 300.0  # K - roughly isothermal T.u assumption
T_BURNT = 2000.0  # K - rough placeholder only, see module docstring

# Inlet: a time-varying (sinusoidal), spatially uniform total pressure.
P_INLET_MEAN = 1.01e5  # Pa
P_INLET_AMPLITUDE = 2e3  # Pa
P_INLET_FREQUENCY = 10.0  # Hz

# Outlet: a fixed static pressure.
P_OUTLET = 1e5  # Pa

# --- mesh geometry (axisymmetric wedge cylinder) ---
RADIUS = 0.02  # m
LENGTH = 0.1  # m
N_RADIAL = 8
N_AXIAL = 40
WEDGE_HALF_ANGLE_DEG = 2.5  # 5deg total wedge angle


def build_mixture_data():
    """Derives `physicalProperties.{u,b}` data and the inlet `fu`.

    Returns:
        `(db, extra, fu_inlet)`; `extra` serves both phases.
    """
    db = ThermoDatabase(
        MECHANISM, thermo_model="janaf", transport_model="const",
    )
    mech = db.mechanism

    oxidant = mech.air(basis="mole")
    afr = mech.stoichiometric_air_fuel_ratio(FUEL, oxidant, basis="mole")
    products = mech.complete_combustion_products(
        FUEL, oxidant, basis="mole", equivalence_ratio=EQUIVALENCE_RATIO,
    )
    extra = xi_fluid_physical_properties_extra(
        afr,
        db.lump(FUEL, basis="mole"),
        db.lump(oxidant, basis="mole"),
        db.lump(products, basis="mass"),
    )

    # mixture = fu*fuel + ox*oxidant + egr*products; with egr=0, fu is
    # the fuel mass fraction at this equivalence ratio.
    actual_air_fuel_ratio = afr / EQUIVALENCE_RATIO
    fu_inlet = 1.0 / (1.0 + actual_air_fuel_ratio)

    return db, extra, fu_inlet


def compute_laminar_flame_speed(db: ThermoDatabase) -> float:
    """Solves one laminar flame speed [m/s] at the operating point."""
    print(
        "Solving a single laminar flame speed point - this is a real "
        "Cantera FreeFlame solve and can take 30-60s ..."
    )
    lfs_db = LaminarFlameSpeedDatabase(
        fuel=FUEL, oxidant=db.mechanism.air(basis="mole"),
        pressures=[P_OPERATING], temperatures=[T_UNBURNT],
        equivalence_ratios=[EQUIVALENCE_RATIO], egr_values=[0.0],
        basis="mole", mechanism=db.mechanism,
    )
    point = lfs_db.solve_point(
        P_OPERATING, T_UNBURNT, EQUIVALENCE_RATIO, 0.0,
    )
    print(f"laminar flame speed Su = {point['sL']:.4g} m/s")
    return point["sL"]


def build_block_mesh_dict() -> dict:
    """Returns the blockMeshDict of one axisymmetric wedge block.

    Vertex and face order follow SandiaD_LTS to get the winding right.
    `boundary` is a flat `[name, dict, name, dict, ...]` list.
    """
    half_angle = math.radians(WEDGE_HALF_ANGLE_DEG)
    y = RADIUS * math.tan(half_angle)

    vertices = [
        (0, 0, 0),                 # 0 axis @ inlet
        (RADIUS, -y, 0),           # 1 neg  @ inlet
        (RADIUS, y, 0),            # 2 pos  @ inlet
        (0, 0, LENGTH),            # 3 axis @ outlet
        (RADIUS, -y, LENGTH),      # 4 neg  @ outlet
        (RADIUS, y, LENGTH),       # 5 pos  @ outlet
    ]

    ret = dict()
    ret["vertices"] = vertices
    ret["blocks"] = [
        Compound(
            "hex", [0, 1, 2, 0, 3, 4, 5, 3], [N_RADIAL, 1, N_AXIAL],
            "simpleGrading", [1, 1, 1],
        ),
    ]
    ret["edges"] = []
    ret["boundary"] = [
        "inlet", {"type": "patch", "faces": [(1, 0, 0, 2)]},
        "outlet", {"type": "patch", "faces": [(3, 4, 5, 3)]},
        "wall", {"type": "wall", "faces": [(1, 2, 5, 4)]},
        "axis", {"type": "empty", "faces": [(0, 3, 3, 0)]},
        "wedgePos", {"type": "wedge", "faces": [(2, 0, 3, 5)]},
        "wedgeNeg", {"type": "wedge", "faces": [(0, 1, 4, 3)]},
    ]
    return ret


def build_case(
    extra: dict, su_value: float, fu_inlet: float,
) -> Case:
    """Builds the case without writing it."""
    thermo_type = ThermoTypeSpec(
        type="ignoredForXiFluid",
        mixture="ignoredForXiFluid",
        equation_of_state="perfectGas",
        energy="sensibleEnthalpy",
        thermo="janaf",
        transport="const",
    )
    su_model = SuModelSpec(
        "unstrained",
        {
            "unstrainedLaminarFlameSpeed": {
                "model": "constant",
                "Su": dimensioned(
                    su_value, DimensionSet(0, 1, -1, 0, 0, 0, 0),
                ),
            },
        },
    )
    xi_model = XiModelSpec("uniformConstant", xi_value=1)

    setup = XiFluidSetup(
        thermo_type,
        "inhomogeneousEGR",
        extra,
        extra,
        su_model,
        xi_model,
        turbulence=TurbulenceSpec("laminar"),
    )

    case = setup.build(
        ROOT,
        ddt_order="steady",
        div_order="firstOrder",
        p_solver=LinearSolver("multigrid", tolerance=1e-7, rel_tol=0.01),
        momentum_solver=LinearSolver(
            "conjugate_gradient", tolerance=1e-6, rel_tol=0.1,
        ),
        n_outer_correctors=1,
        relaxation_factors={
            "fields": {"p": 0.3},
            "equations": {String(".*"): 0.7},
        },
        start_time=0, end_time=1000, delta_t=1,
        write_control="timeStep", write_interval=100,
    )

    case["system/blockMeshDict"] = build_block_mesh_dict()

    # pressureInletOutletVelocity derives U from the pressure.
    u = case.add_field("U", time=0)
    u.set_bc("wall", NoSlip())
    u.set_bc("inlet", PressureInletOutletVelocity())
    u.set_bc("outlet", PressureInletOutletVelocity())

    # Uniform, time-varying total pressure in; static pressure out.
    p = case.add_field("p", time=0, internal_value=P_OUTLET)
    p.set_bc("wall", FixedFluxPressure(P_OUTLET))
    p.set_bc(
        "inlet",
        UniformTotalPressure(
            p0={
                "type": "sine",
                "frequency": P_INLET_FREQUENCY,
                "amplitude": P_INLET_AMPLITUDE,
                "level": P_INLET_MEAN,
            },
            value=P_OUTLET,
        ),
    )
    p.set_bc("outlet", FixedValue(P_OUTLET))

    # Unburnt, unwrinkled inflow.
    b = case.add_field("b", time=0, internal_value=0)
    b.set_bc("inlet", FixedValue(0))
    b.set_bc("outlet", ZeroGradient())

    xi = case.add_field("Xi", time=0)
    xi.set_bc("inlet", FixedValue(1))
    xi.set_bc("outlet", ZeroGradient())

    # T.u/T.b
    t_u = case.add_field("T", time=0, path="T.u")
    t_u.set_bc("inlet", FixedValue(T_UNBURNT))
    t_u.set_bc("outlet", InletOutlet(T_UNBURNT))

    t_b = case.add_field("T", time=0, path="T.b", internal_value=T_BURNT)
    t_b.set_bc("inlet", FixedValue(T_BURNT))
    t_b.set_bc("outlet", InletOutlet(T_BURNT))

    # Premixed, EGR-free inflow.
    fu = case.add_field(
        "fu", time=0, path="fu.u", type="scalar", dimensions="[]",
        internal_value=0,
    )
    fu.set_bc("inlet", FixedValue(fu_inlet))
    fu.set_bc("outlet", InletOutlet(fu_inlet))

    egr = case.add_field(
        "egr", time=0, path="egr.u", type="scalar", dimensions="[]",
        internal_value=0,
    )
    egr.set_bc("inlet", FixedValue(0))
    egr.set_bc("outlet", InletOutlet(0))

    # No burnt mixture enters; the default suffices.
    case.add_field(
        "ft", time=0, path="ft.b", type="scalar", dimensions="[]",
        internal_value=0,
    )

    return case


def main():
    print(f"Building XiFluid case at {ROOT} ...")
    db, extra, fu_inlet = build_mixture_data()
    su_value = compute_laminar_flame_speed(db)
    case = build_case(extra, su_value, fu_inlet)
    case.write()
    print(f"Case written to {ROOT} (mesh not yet generated).")
    print(
        "\nTo mesh and run it:\n"
        f"    cd {ROOT}\n"
        "    blockMesh\n"
        "    checkMesh\n"
        "    foamRun\n"
    )


if __name__ == "__main__":
    main()
