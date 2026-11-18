import subprocess

import pytest

from pyfoamy import native
from pyfoamy.models import FluidSetup, LinearSolver, ThermoTypeSpec, \
    TurbulenceSpec
from pyfoamy.parsing.grammar import Directive

P_SOLVER = LinearSolver("multigrid", tolerance=1e-7, rel_tol=0.01)
MOMENTUM_SOLVER = LinearSolver(
    "conjugate_gradient", tolerance=1e-6, rel_tol=0.1
)

PERFECT_GAS_THERMO = ThermoTypeSpec(
    type="hePsiThermo",
    mixture="pureMixture",
    transport="const",
    thermo="hConst",
    equation_of_state="perfectGas",
    energy="sensibleInternalEnergy",
)

AIR_MIXTURE = {
    "specie": {"molWeight": 28.9},
    "thermodynamics": {"Cp": 1005, "hf": 0},
    "transport": {"mu": 1.82e-05, "Pr": 0.71},
}

BOUSSINESQ_THERMO = ThermoTypeSpec(
    type="heRhoThermo",
    mixture="pureMixture",
    transport="const",
    thermo="hConst",
    equation_of_state="Boussinesq",
    energy="sensibleEnthalpy",
)

BOUSSINESQ_AIR_MIXTURE = {
    "specie": {"molWeight": 28.9},
    "equationOfState": {"rho0": 1, "T0": 300, "beta": 3e-03},
    "thermodynamics": {"Cp": 1000, "hf": 0},
    "transport": {"mu": 1e-05, "Pr": 0.7},
}


def _words(entry):
    return str(entry).split()


def _build(setup, root, **kwargs):
    kwargs.setdefault("ddt_order", "steady")
    kwargs.setdefault("div_order", "firstOrder")
    kwargs.setdefault("p_solver", P_SOLVER)
    kwargs.setdefault("momentum_solver", MOMENTUM_SOLVER)
    return setup.build(root, **kwargs)


def test_required_fields_laminar_non_buoyant():
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    assert setup.required_fields() == ["p", "U", "T"]


def test_required_fields_excludes_energy_field_and_rho():
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    fields = setup.required_fields()
    assert "e" not in fields
    assert "h" not in fields
    assert "rho" not in fields


def test_required_fields_with_turbulence_adds_alphat():
    setup = FluidSetup(
        PERFECT_GAS_THERMO,
        AIR_MIXTURE,
        turbulence=TurbulenceSpec("RAS", "kEpsilon"),
    )
    assert setup.required_fields() == [
        "p", "U", "T", "nut", "k", "epsilon", "alphat",
    ]


def test_required_fields_buoyant_needs_both_p_rgh_and_p():
    # p_rgh is solved, but p (= p_rgh + rho*gh) is still required.
    setup = FluidSetup(
        BOUSSINESQ_THERMO, BOUSSINESQ_AIR_MIXTURE, buoyant=True
    )
    fields = setup.required_fields()
    assert "p_rgh" in fields
    assert "p" in fields


def test_build_requires_gravity_when_buoyant(tmp_path):
    setup = FluidSetup(
        BOUSSINESQ_THERMO, BOUSSINESQ_AIR_MIXTURE, buoyant=True
    )
    with pytest.raises(ValueError, match="gravity"):
        _build(setup, tmp_path / "myCase")


def test_build_writes_a_bare_solver_entry_no_application(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(setup, tmp_path / "myCase")
    d = case["system/controlDict"]
    assert str(d["solver"]) == "fluid"
    assert "application" not in d


def test_build_writes_thermo_type_and_mixture(tmp_path):
    # mixture_file=None opts back into the old inline-embedded shape.
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE, mixture_file=None)
    case = _build(setup, tmp_path / "myCase")
    d = case["constant/physicalProperties"]
    thermo_type = d["thermoType"]
    assert str(thermo_type["type"]) == "hePsiThermo"
    assert str(thermo_type["equationOfState"]) == "perfectGas"
    mixture = d["mixture"]
    assert float(mixture["transport"]["mu"]) == 1.82e-05


def test_mixture_file_defaults_to_mixture_dat(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    root = tmp_path / "myCase"
    case = _build(setup, root)
    physical_properties = case["constant/physicalProperties"]
    assert "mixture" not in physical_properties
    directive = physical_properties[-1]
    assert isinstance(directive, Directive)
    assert directive.name == "#include"
    assert str(directive.args) == '"$FOAM_CASE/constant/mixture.dat"'

    case.write()
    text = (root / "constant" / "mixture.dat").read_text()
    assert "FoamFile" not in text
    mixture_dict = case.read_dict("constant/mixture.dat")
    assert float(mixture_dict["mixture"]["transport"]["mu"]) == 1.82e-05


def test_build_writes_g_when_buoyant(tmp_path):
    setup = FluidSetup(
        BOUSSINESQ_THERMO, BOUSSINESQ_AIR_MIXTURE, buoyant=True
    )
    case = _build(setup, tmp_path / "myCase", gravity=(0, -9.81, 0))
    g = case["constant/g"]
    assert str(g["FoamFile"]["class"]) == (
        "uniformDimensionedVectorField"
    )
    assert str(g["dimensions"]) == "[acceleration]"


def test_build_does_not_write_g_when_not_buoyant(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(setup, tmp_path / "myCase")
    case.write()
    assert not (tmp_path / "myCase" / "constant" / "g").exists()


def test_div_schemes_include_fixed_k_and_viscous_terms(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(setup, tmp_path / "myCase", div_order="secondOrder")
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,K)"]) == ["Gauss", "linear"]
    assert _words(
        div["div(((rho*nuEff)*dev2(T(grad(U)))))"]
    ) == ["Gauss", "linear"]


def test_second_order_div_scheme_names_the_field_gradient(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(setup, tmp_path / "myCase", div_order="secondOrder")
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,U)"]) == ["Gauss", "LUST", "grad(U)"]


def test_div_schemes_include_fluid_specific_p_rho_term(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(setup, tmp_path / "myCase")
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,(p|rho))"]) == ["Gauss", "linear"]


def test_div_schemes_include_energy_field_resolved_via_div_order(
    tmp_path,
):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(
        setup, tmp_path / "myCase", div_order="secondOrderBounded"
    )
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,e)"]) == ["Gauss", "limitedLinear", "1"]


def test_div_schemes_pin_turbulence_fields_regardless_of_order(
    tmp_path,
):
    setup = FluidSetup(
        PERFECT_GAS_THERMO, AIR_MIXTURE,
        turbulence=TurbulenceSpec("RAS", "kEpsilon"),
    )
    case = _build(setup, tmp_path / "myCase", div_order="secondOrder")
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,k)"]) == ["Gauss", "upwind"]
    assert _words(div["div(phi,epsilon)"]) == ["Gauss", "upwind"]


def test_div_schemes_add_phid_p_when_transonic(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE, transonic=True)
    case = _build(setup, tmp_path / "myCase", div_order="firstOrder")
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(phid,p)" in div
    assert _words(div["div(phid,p)"]) == ["Gauss", "upwind"]


def test_div_schemes_no_phid_p_when_not_transonic(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(setup, tmp_path / "myCase")
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(phid,p)" not in div


def test_div_schemes_use_p_rgh_for_phid_when_transonic_and_buoyant(
    tmp_path,
):
    setup = FluidSetup(
        BOUSSINESQ_THERMO,
        BOUSSINESQ_AIR_MIXTURE,
        buoyant=True,
        transonic=True,
    )
    case = _build(
        setup,
        tmp_path / "myCase",
        div_order="firstOrder",
        gravity=(0, -9.81, 0),
    )
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(phid,p_rgh)" in div
    assert "div(phid,p)" not in div


def test_rho_solver_is_hardcoded_diagonal(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(setup, tmp_path / "myCase")
    rho = case["system/fvSolution"]["solvers"]["rho"]
    assert str(rho["solver"]) == "diagonal"
    assert "tolerance" not in rho


def test_pressure_field_is_p_when_not_buoyant(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(setup, tmp_path / "myCase")
    solvers = case["system/fvSolution"]["solvers"]
    assert "p" in solvers
    assert "p_rgh" not in solvers


def test_pressure_field_is_p_rgh_when_buoyant(tmp_path):
    setup = FluidSetup(
        BOUSSINESQ_THERMO, BOUSSINESQ_AIR_MIXTURE, buoyant=True
    )
    case = _build(
        setup, tmp_path / "myCase", gravity=(0, -9.81, 0)
    )
    solvers = case["system/fvSolution"]["solvers"]
    assert "p_rgh" in solvers
    assert "p" not in solvers


def test_pressure_solver_is_symmetric_when_not_transonic(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(
        setup,
        tmp_path / "myCase",
        p_solver=LinearSolver(
            "conjugate_gradient", tolerance=1e-6, rel_tol=0.01
        ),
    )
    p = case["system/fvSolution"]["solvers"]["p"]
    assert str(p["solver"]) == "PCG"


def test_pressure_solver_is_asymmetric_when_transonic(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE, transonic=True)
    case = _build(
        setup,
        tmp_path / "myCase",
        p_solver=LinearSolver(
            "conjugate_gradient", tolerance=1e-6, rel_tol=0.01
        ),
    )
    p = case["system/fvSolution"]["solvers"]["p"]
    assert str(p["solver"]) == "PBiCGStab"


def test_momentum_group_includes_energy_field_not_temperature(
    tmp_path,
):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(setup, tmp_path / "myCase")
    solvers = case["system/fvSolution"]["solvers"]
    assert "e" in solvers
    assert "T" not in solvers


def test_full_write_read_roundtrip(tmp_path):
    setup = FluidSetup(
        PERFECT_GAS_THERMO, AIR_MIXTURE,
        turbulence=TurbulenceSpec("RAS", "kEpsilon"),
    )
    root = tmp_path / "myCase"
    case = _build(setup, root, div_order="secondOrderBounded")
    case.write()

    reread = case.read_dict("constant/physicalProperties")
    assert str(reread["thermoType"]["type"]) == (
        "hePsiThermo"
    )
    assert (root / "system" / "controlDict").is_file()
    assert (root / "constant" / "physicalProperties").is_file()
    assert (root / "system" / "fvSchemes").is_file()
    assert (root / "system" / "fvSolution").is_file()


@pytest.mark.openfoam
def test_written_case_is_valid_per_foam_dictionary(tmp_path):
    setup = FluidSetup(
        PERFECT_GAS_THERMO, AIR_MIXTURE,
        turbulence=TurbulenceSpec("RAS", "kEpsilon"),
    )
    root = tmp_path / "myCase"
    case = _build(setup, root, div_order="secondOrderBounded")
    case.write()

    def foam_value(relative_path, entry):
        result = subprocess.run(
            [
                "foamDictionary",
                "-entry",
                entry,
                "-value",
                str(root / relative_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    assert foam_value("system/controlDict", "solver") == "fluid"
    assert (
        foam_value("constant/physicalProperties", "thermoType/thermo")
        == "hConst"
    )
    assert (
        foam_value("system/fvSolution", "solvers/rho/solver")
        == "diagonal"
    )
    assert (
        foam_value("system/fvSchemes", "divSchemes/div(phi,K)")
        == "Gauss linear"
    )


@pytest.mark.openfoam
def test_written_buoyant_case_is_valid_per_foam_dictionary(tmp_path):
    setup = FluidSetup(
        BOUSSINESQ_THERMO, BOUSSINESQ_AIR_MIXTURE, buoyant=True
    )
    root = tmp_path / "myCase"
    case = _build(
        setup, root, gravity=(0, -9.81, 0), pressure_reference=(0, 0)
    )
    case.write()

    result = subprocess.run(
        ["foamDictionary", "-entry", "value", "-value",
         str(root / "constant" / "g")],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "( 0 -9.81 0 )"


@pytest.mark.slow
@pytest.mark.openfoam
def test_control_dict_satisfies_real_time_construction(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    root = tmp_path / "myCase"
    case = _build(setup, root)
    case.write()

    with pytest.raises(native.NativeError, match="points|polyMesh"):
        with native.open_case(tmp_path, "myCase"):
            pass


def test_build_adds_field_gradients_wall_distance_and_extras(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE, transonic=True)
    case = _build(
        setup, tmp_path, grad_schemes={"U": "cellLimited Gauss linear 1"},
        pimple={"consistent": False},
        solvers={"MeshPhi": {"solver": "smoothSolver"}},
    )
    schemes = case["system/fvSchemes"]
    assert _words(schemes["gradSchemes"]["grad(U)"]) == [
        "cellLimited", "Gauss", "linear", "1",
    ]
    assert str(schemes["wallDist"]["method"]) == "meshWave"
    solution = case["system/fvSolution"]
    assert str(solution["PIMPLE"]["transonic"]) == "true"
    assert str(solution["PIMPLE"]["consistent"]) == "false"
    assert str(solution["solvers"]["MeshPhi"]["solver"]) == "smoothSolver"

    case = _build(
        FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE), tmp_path,
        wall_dist=None,
    )
    assert "wallDist" not in case["system/fvSchemes"]
    assert "transonic" not in case["system/fvSolution"]["PIMPLE"]


def test_build_loads_the_pyfoamy_lib_unless_switched_off(tmp_path):
    setup = FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE)
    case = _build(setup, tmp_path / "a")
    libs = [v.data for v in case["system/controlDict"]["libs"]]
    assert libs == ["libpyfoamy.so"]
    case = _build(setup, tmp_path / "b", pyfoamy_lib=False)
    assert "libs" not in case["system/controlDict"]


def test_build_raises_for_an_unsupported_openfoam(tmp_path, monkeypatch):
    monkeypatch.setenv("WM_PROJECT_VERSION", "13")
    with pytest.raises(ValueError, match="'13'"):
        _build(FluidSetup(PERFECT_GAS_THERMO, AIR_MIXTURE), tmp_path / "a")
