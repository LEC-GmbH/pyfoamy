import subprocess

import pytest

from pyfoamy import native
from pyfoamy.models import (
    DLB,
    DLB_PYJAC,
    LinearSolver,
    ReactionSpec,
    SuModelSpec,
    ThermoTypeSpec,
    TurbulenceSpec,
    XiEqModelSpec,
    XiFluidSetup,
    XiGModelSpec,
    XiModelSpec,
    XiProfileSpec,
)
from pyfoamy.parsing.grammar import Directive

P_SOLVER = LinearSolver("multigrid", tolerance=1e-7, rel_tol=0.01)
MOMENTUM_SOLVER = LinearSolver(
    "conjugate_gradient", tolerance=1e-6, rel_tol=0.1
)

THERMO = ThermoTypeSpec(
    type="ignoredForXiFluid",
    mixture="ignoredForXiFluid",
    transport="const",
    thermo="janaf",
    equation_of_state="perfectGas",
    energy="sensibleEnthalpy",
)

# angledDuct-style specie data, real shape from tutorials/XiFluid/stratified
FUEL = {
    "specie": {"molWeight": 114.23},
    "thermodynamics": {
        "Tlow": 298.15, "Thigh": 5000, "Tcommon": 1000,
        "highCpCoeffs": [
            19.3963, 0.0488137, -1.55798e-05, 2.17831e-09,
            -1.08391e-13, -37133.9, -78.8531,
        ],
        "lowCpCoeffs": [
            0.429357, 0.0765769, 8.41306e-06, -5.86603e-08,
            2.79414e-11, -30446.4, 25.6735,
        ],
    },
    "transport": {"mu": 1e-5, "Pr": 1},
}
OXIDANT = {
    "specie": {"molWeight": 28.8504},
    "thermodynamics": {
        "Tlow": 200, "Thigh": 6000, "Tcommon": 1000,
        "highCpCoeffs": [
            3.10205, 0.00123963, -4.17512e-07, 6.60292e-11,
            -3.87448e-15, -985.517, 5.35187,
        ],
        "lowCpCoeffs": [
            3.58378, -0.0007269, 1.66985e-06, -1.08452e-10,
            -4.31951e-13, -1050.53, 3.11223,
        ],
    },
    "transport": {"mu": 1e-5, "Pr": 1},
}
PRODUCTS = {
    "specie": {"molWeight": 28.6068},
    "thermodynamics": {
        "Tlow": 200, "Thigh": 6000, "Tcommon": 1000,
        "highCpCoeffs": [
            3.12468, 0.00178578, -5.94695e-07, 9.09801e-11,
            -5.1246e-15, -11005.8, 5.03593,
        ],
        "lowCpCoeffs": [
            3.47612, 0.000758205, -3.68739e-07, 1.3449e-09,
            -8.08868e-13, -11071.5, 3.3058,
        ],
    },
    "transport": {"mu": 1e-5, "Pr": 1},
}

INHOMOGENEOUS_EXTRA = {
    "stoichiometricAirFuelMassRatio": 15.675,
    "fuel": FUEL,
    "oxidant": OXIDANT,
    "products": PRODUCTS,
}

HOMOGENEOUS_U_EXTRA = {
    "stoichiometricAirFuelMassRatio": 15.675,
    "Phi": 1,
    "reactants": {
        "specie": {"molWeight": 29.4649},
        "thermodynamics": {"Cp": 1000, "hf": 1700e3},
        "transport": {"mu": 1e-5, "Pr": 1},
    },
}
HOMOGENEOUS_B_EXTRA = {
    "products": {
        "specie": {"molWeight": 28.3233},
        "thermodynamics": {"Cp": 1000, "hf": 0},
        "transport": {"mu": 1e-5, "Pr": 1},
    },
}

UNSTRAINED_SU_MODEL = SuModelSpec(
    "unstrained",
    {
        "unstrainedLaminarFlameSpeed": {
            "model": "Gulder",
            "fuel": "Propane",
            "Propane": {
                "W": 0.446, "eta": 0.12, "xi": 4.95,
                "alpha": 1.77, "beta": -0.2, "f": 2.3,
            },
        },
    },
)

EQUILIBRIUM_XI_MODEL = XiModelSpec(
    "equilibrium",
    eq=XiEqModelSpec("Gulder"),
    profile=XiProfileSpec("linear"),
)
TRANSPORT_XI_MODEL = XiModelSpec(
    "transport",
    eq=XiEqModelSpec("Gulder"),
    profile=XiProfileSpec("linear"),
    generation_rate=XiGModelSpec("KTS"),
)
UNIFORM_CONSTANT_XI_MODEL = XiModelSpec("uniformConstant", xi_value=1)


def _words(entry):
    return str(entry).split()


def _build(setup, root, **kwargs):
    kwargs.setdefault("ddt_order", "firstOrder")
    kwargs.setdefault("div_order", "secondOrderBounded")
    kwargs.setdefault("p_solver", P_SOLVER)
    kwargs.setdefault("momentum_solver", MOMENTUM_SOLVER)
    return setup.build(root, **kwargs)


def _setup(**kwargs):
    kwargs.setdefault("thermo_type", THERMO)
    kwargs.setdefault("mixture_type", "inhomogeneous")
    kwargs.setdefault(
        "u_physical_properties_extra", dict(INHOMOGENEOUS_EXTRA)
    )
    kwargs.setdefault(
        "b_physical_properties_extra", dict(INHOMOGENEOUS_EXTRA)
    )
    kwargs.setdefault("su_model", UNSTRAINED_SU_MODEL)
    kwargs.setdefault("xi_model", EQUILIBRIUM_XI_MODEL)
    return XiFluidSetup(**kwargs)


# --- SuModelSpec / XiModelSpec / sub-models ---


def test_su_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="RaviPetersen"):
        SuModelSpec("RaviPetersen")


def test_su_model_to_dict_includes_coefficients():
    spec = SuModelSpec("uniformConstant", {"Su": 0.4})
    assert spec.to_dict() == {
        "model": "uniformConstant",
        "uniformConstant": {"Su": 0.4},
    }


def test_xi_eq_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="RaviPetersen"):
        XiEqModelSpec("RaviPetersen")


def test_xi_g_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown"):
        XiGModelSpec("bogus")


def test_xi_profile_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown"):
        XiProfileSpec("bogus")


def test_xi_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown"):
        XiModelSpec("bogus")


def test_xi_model_equilibrium_requires_eq_and_profile():
    with pytest.raises(ValueError, match="requires both eq and profile"):
        XiModelSpec("equilibrium")


def test_xi_model_equilibrium_rejects_generation_rate():
    with pytest.raises(ValueError, match="generation_rate"):
        XiModelSpec(
            "equilibrium",
            eq=XiEqModelSpec("Gulder"),
            profile=XiProfileSpec("linear"),
            generation_rate=XiGModelSpec("KTS"),
        )


def test_xi_model_transport_requires_generation_rate():
    with pytest.raises(ValueError, match="requires generation_rate"):
        XiModelSpec(
            "transport",
            eq=XiEqModelSpec("Gulder"),
            profile=XiProfileSpec("linear"),
        )


def test_xi_model_uniform_constant_requires_xi_value():
    with pytest.raises(ValueError, match="requires xi_value"):
        XiModelSpec("uniformConstant")


def test_xi_model_uniform_constant_rejects_sub_models():
    with pytest.raises(ValueError, match="doesn't use"):
        XiModelSpec(
            "uniformConstant", xi_value=1, eq=XiEqModelSpec("Gulder")
        )


def test_xi_model_rejects_xi_value_when_not_uniform_constant():
    with pytest.raises(ValueError, match="doesn't use"):
        XiModelSpec(
            "equilibrium",
            eq=XiEqModelSpec("Gulder"),
            profile=XiProfileSpec("linear"),
            xi_value=1,
        )


def test_xi_model_to_dict_nests_sub_models():
    assert TRANSPORT_XI_MODEL.to_dict() == {
        "model": "transport",
        "eq": {"model": "Gulder"},
        "profile": {"model": "linear"},
        "generationRate": {"model": "KTS"},
    }


def test_xi_model_uniform_constant_to_dict():
    assert UNIFORM_CONSTANT_XI_MODEL.to_dict() == {
        "model": "uniformConstant", "Xi": 1,
    }


# --- XiFluidSetup: construction / mixture types ---


def test_rejects_unknown_mixture_type():
    with pytest.raises(ValueError, match="mixture_type"):
        _setup(mixture_type="bogus")


def test_homogeneous_required_fields_has_no_species():
    setup = _setup(
        mixture_type="homogeneous",
        u_physical_properties_extra=dict(HOMOGENEOUS_U_EXTRA),
        b_physical_properties_extra=dict(HOMOGENEOUS_B_EXTRA),
    )
    fields = setup.required_fields()
    assert "fu.u" not in fields
    assert "ft.b" not in fields
    assert "egr.u" not in fields
    assert fields[:6] == ["p", "U", "b", "Xi", "T.u", "T.b"]


def test_inhomogeneous_required_fields_include_fu_and_ft():
    setup = _setup(mixture_type="inhomogeneous")
    fields = setup.required_fields()
    assert "fu.u" in fields
    assert "ft.b" in fields
    assert "egr.u" not in fields


def test_inhomogeneous_egr_required_fields_include_egr():
    setup = _setup(mixture_type="inhomogeneousEGR")
    fields = setup.required_fields()
    assert "fu.u" in fields
    assert "egr.u" in fields
    assert "ft.b" in fields


def test_alphat_only_when_non_laminar():
    laminar = _setup(turbulence=TurbulenceSpec("laminar"))
    assert "alphat.u" not in laminar.required_fields()

    turbulent = _setup(turbulence=TurbulenceSpec("RAS", "kEpsilon"))
    fields = turbulent.required_fields()
    assert "alphat.u" in fields
    assert "alphat.b" in fields
    assert "k" in fields
    assert "epsilon" in fields


# --- buoyancy / transonic ---


def test_required_fields_buoyant_needs_both_p_rgh_and_p():
    # p_rgh is solved, but p (= p_rgh + rho*gh) is still required.
    setup = _setup(buoyant=True)
    fields = setup.required_fields()
    assert "p_rgh" in fields
    assert "p" in fields


def test_required_fields_uses_p_when_not_buoyant():
    setup = _setup()
    fields = setup.required_fields()
    assert "p" in fields
    assert "p_rgh" not in fields


def test_build_requires_gravity_when_buoyant(tmp_path):
    setup = _setup(buoyant=True)
    with pytest.raises(ValueError, match="gravity"):
        _build(setup, tmp_path / "myCase")


def test_build_writes_g_when_buoyant(tmp_path):
    setup = _setup(buoyant=True)
    case = _build(
        setup, tmp_path / "myCase", gravity=(0, -9.81, 0)
    )
    g = case["constant/g"]
    assert str(g["FoamFile"]["class"]) == (
        "uniformDimensionedVectorField"
    )
    assert str(g["value"]) == "(0 -9.81 0)"


def test_build_does_not_write_g_when_not_buoyant(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    case.write()
    assert not (tmp_path / "myCase" / "constant" / "g").exists()


def test_div_schemes_add_phid_p_when_transonic(tmp_path):
    setup = _setup(transonic=True)
    case = _build(setup, tmp_path / "myCase", div_order="firstOrder")
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(phid,p)" in div
    assert _words(div["div(phid,p)"]) == ["Gauss", "upwind"]


def test_div_schemes_no_phid_when_not_transonic(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(phid,p)" not in div
    assert "div(phid,p_rgh)" not in div


def test_div_schemes_use_p_rgh_for_phid_when_transonic_and_buoyant(
    tmp_path,
):
    setup = _setup(buoyant=True, transonic=True)
    case = _build(
        setup, tmp_path / "myCase", gravity=(0, -9.81, 0)
    )
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(phid,p_rgh)" in div
    assert "div(phid,p)" not in div


def test_pressure_solver_is_symmetric_when_not_transonic(tmp_path):
    setup = _setup()
    case = _build(
        setup,
        tmp_path / "myCase",
        p_solver=LinearSolver(
            "conjugate_gradient", tolerance=1e-6, rel_tol=0.1
        ),
    )
    p = case["system/fvSolution"]["solvers"]["p"]
    assert str(p["solver"]) == "PCG"


def test_pressure_solver_is_asymmetric_when_transonic(tmp_path):
    setup = _setup(transonic=True)
    case = _build(
        setup,
        tmp_path / "myCase",
        p_solver=LinearSolver(
            "conjugate_gradient", tolerance=1e-6, rel_tol=0.1
        ),
    )
    p = case["system/fvSolution"]["solvers"]["p"]
    assert str(p["solver"]) == "PBiCGStab"


# --- build(): controlDict / physicalProperties.{u,b} ---


def test_build_writes_a_bare_solver_entry(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    assert str(case["system/controlDict"]["solver"]) == "XiFluid"


def test_physical_properties_u_and_b_have_distinct_mixtures(tmp_path):
    setup = _setup(mixture_type="inhomogeneousEGR")
    case = _build(setup, tmp_path / "myCase")
    u_thermo = case["constant/physicalProperties.u"]["thermoType"]
    b_thermo = case["constant/physicalProperties.b"]["thermoType"]
    assert str(u_thermo["type"]) == "uRhoThermo"
    assert str(u_thermo["mixture"]) == "uInhomogeneousEGR"
    assert str(b_thermo["type"]) == "bRhoThermo"
    assert str(b_thermo["mixture"]) == "bInhomogeneous"


def test_physical_properties_extra_is_passed_through_per_phase(
    tmp_path,
):
    # species_file=None: inline per-phase specie data.
    setup = _setup(
        mixture_type="homogeneous",
        u_physical_properties_extra=dict(HOMOGENEOUS_U_EXTRA),
        b_physical_properties_extra=dict(HOMOGENEOUS_B_EXTRA),
        species_file=None,
    )
    case = _build(setup, tmp_path / "myCase")
    u = case["constant/physicalProperties.u"]
    b = case["constant/physicalProperties.b"]
    assert "reactants" in u
    assert "products" not in u
    assert "products" in b
    assert "reactants" not in b
    assert float(u["Phi"]) == 1


def test_momentum_transport_shared_file_has_full_model(tmp_path):
    setup = _setup(turbulence=TurbulenceSpec("RAS", "kOmegaSST"))
    case = _build(setup, tmp_path / "myCase")
    shared = case["constant/momentumTransport"]
    assert str(shared["simulationType"]) == "RAS"
    assert str(shared["RAS"]["model"]) == "kOmegaSST"


def test_momentum_transport_phase_files_are_bare(tmp_path):
    setup = _setup(turbulence=TurbulenceSpec("RAS", "kOmegaSST"))
    case = _build(setup, tmp_path / "myCase")
    u = case["constant/momentumTransport.u"]
    b = case["constant/momentumTransport.b"]
    assert str(u["simulationType"]) == "RAS"
    assert "RAS" not in u
    assert str(b["simulationType"]) == "RAS"
    assert "RAS" not in b


def test_combustion_properties_shape(tmp_path):
    setup = _setup(
        su_model=UNSTRAINED_SU_MODEL,
        xi_model=TRANSPORT_XI_MODEL,
    )
    case = _build(setup, tmp_path / "myCase")
    combustion = case["constant/combustionProperties"]
    assert (
        str(combustion["laminarFlameSpeed"]["model"])
        == "unstrained"
    )
    flame_wrinkling = combustion["flameWrinkling"]
    assert str(flame_wrinkling["model"]) == "transport"
    assert str(flame_wrinkling["eq"]["model"]) == "Gulder"


# --- fvSchemes ---


def test_div_schemes_use_alphaphi_for_species_and_energy(tmp_path):
    setup = _setup(mixture_type="inhomogeneous")
    case = _build(setup, tmp_path / "myCase")
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(alphaPhi,fu)" in div
    assert "div(alphaPhi,ft)" in div
    assert "div(alphaPhi,h)" in div
    assert "div(alphaPhi,K)" in div
    assert _words(div["div(alphaPhi,fu)"]) == [
        "Gauss", "limitedLinear01", "1",
    ]
    assert _words(div["div(alphaPhi,h)"]) == [
        "Gauss", "limitedLinear", "1",
    ]


def test_div_schemes_pin_b_to_bounded_category(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase", div_order="firstOrder")
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,b)"]) == ["Gauss", "limitedLinear01", "1"]


def test_div_schemes_include_phixi_when_xi_model_not_uniform(tmp_path):
    setup = _setup(xi_model=EQUILIBRIUM_XI_MODEL)
    case = _build(setup, tmp_path / "myCase")
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(phiXi,Xi)" in div


def test_div_schemes_exclude_phixi_when_xi_model_uniform_constant(
    tmp_path,
):
    setup = _setup(xi_model=UNIFORM_CONSTANT_XI_MODEL)
    case = _build(setup, tmp_path / "myCase")
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(phiXi,Xi)" not in div


def test_div_schemes_include_shared_viscous_stress_term(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(
        div["div(((rho*nuEff)*dev2(T(grad(U)))))"]
    ) == ["Gauss", "linear"]


def test_div_schemes_follow_div_order_for_momentum(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase", div_order="firstOrder")
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,U)"]) == ["Gauss", "upwind"]


def test_div_schemes_pin_turbulence_fields_to_first_order(tmp_path):
    setup = _setup(turbulence=TurbulenceSpec("RAS", "kEpsilon"))
    case = _build(setup, tmp_path / "myCase", div_order="secondOrder")
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,k)"]) == ["Gauss", "upwind"]


# --- fvSolution ---


def test_fv_solution_includes_rho_diagonal_entry(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    rho = case["system/fvSolution"]["solvers"]["rho"]
    assert str(rho["solver"]) == "diagonal"


def test_fv_solution_p_is_symmetric(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    p = case["system/fvSolution"]["solvers"]["p"]
    assert str(p["solver"]) == "GAMG"


def test_fv_solution_momentum_group_is_asymmetric(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    u = case["system/fvSolution"]["solvers"]["U"]
    assert str(u["solver"]) == "PBiCGStab"


def test_fv_solution_includes_b_asymmetric_solver_entry(tmp_path):
    # b is solved with U/h.u/h.b ("(U|b|h.u|h.b).*" as in the 1Dlaminar
    # tutorial); without it an ignited foamRun fails in burn().
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    solvers = case["system/fvSolution"]["solvers"]
    assert "b" in solvers
    assert "bFinal" in solvers
    b = solvers["b"]
    assert str(b["solver"]) == "PBiCGStab"


def test_fv_solution_includes_species_and_energy_per_phase(tmp_path):
    setup = _setup(mixture_type="inhomogeneousEGR")
    case = _build(setup, tmp_path / "myCase")
    solvers = case["system/fvSolution"]["solvers"]
    for name in [
        "fu.u", "egr.u", "ft.b", "h.u", "h.b", "Xi", "Su",
        "fu.uFinal", "egr.uFinal", "ft.bFinal",
    ]:
        assert name in solvers, f"missing solver block for {name!r}"


def test_fv_solution_excludes_xi_su_when_xi_model_uniform_constant(
    tmp_path,
):
    setup = _setup(xi_model=UNIFORM_CONSTANT_XI_MODEL)
    case = _build(setup, tmp_path / "myCase")
    solvers = case["system/fvSolution"]["solvers"]
    assert "Xi" not in solvers
    assert "Su" not in solvers


def test_final_block_forces_rel_tol_zero(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    p_final = (
        case["system/fvSolution"]["solvers"]["pFinal"]
    )
    assert float(p_final["relTol"]) == 0


# --- reaction models (per phase) ---


def test_no_reaction_writes_no_reaction_properties_files(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    case.write()
    root = tmp_path / "myCase"
    assert not (root / "constant" / "reactionProperties.u").exists()
    assert not (root / "constant" / "reactionProperties.b").exists()


def test_u_reaction_writes_suffixed_reaction_properties(tmp_path):
    setup = _setup(u_reaction=ReactionSpec("PaSR", {"Cmix": 1.0}))
    case = _build(setup, tmp_path / "myCase")
    case.write()
    root = tmp_path / "myCase"
    assert (root / "constant" / "reactionProperties.u").is_file()
    assert not (root / "constant" / "reactionProperties.b").exists()
    reaction = case.read_dict("constant/reactionProperties.u")
    assert str(reaction["reactionModel"]) == "PaSR"


def test_species_file_rejects_a_genuine_key_conflict():
    with pytest.raises(ValueError, match="species_file"):
        _setup(
            u_physical_properties_extra={"stoichiometricAirFuelMassRatio": 1},
            b_physical_properties_extra={"stoichiometricAirFuelMassRatio": 2},
            species_file="species.dat",
        )


def test_species_file_defaults_on(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    directive = case["constant/physicalProperties.u"][-1]
    assert isinstance(directive, Directive)
    assert directive.name == "#include"
    assert str(directive.args) == '"$FOAM_CASE/constant/species.dat"'


def test_species_file_writes_the_union_for_both_phases(tmp_path):
    # The U and B extras share no keys; both phases include the
    # combined file.
    setup = _setup(
        mixture_type="homogeneous",
        u_physical_properties_extra=dict(HOMOGENEOUS_U_EXTRA),
        b_physical_properties_extra=dict(HOMOGENEOUS_B_EXTRA),
    )
    root = tmp_path / "myCase"
    case = _build(setup, root)
    for phase in ("u", "b"):
        physical_properties = case[f"constant/physicalProperties.{phase}"]
        assert "reactants" not in physical_properties
        assert "products" not in physical_properties
        directive = physical_properties[-1]
        assert isinstance(directive, Directive)
        assert directive.name == "#include"
        assert str(directive.args) == (
            '"$FOAM_CASE/constant/species.dat"'
        )

    case.write()
    text = (root / "constant" / "species.dat").read_text()
    assert "FoamFile" not in text
    species_dict = case.read_dict("constant/species.dat")
    assert "reactants" in species_dict
    assert "products" in species_dict
    assert float(species_dict["Phi"]) == 1


def test_u_and_b_reaction_files_do_not_collide(tmp_path):
    setup = _setup(
        u_reaction=ReactionSpec(
            "PaSR", reactions={"r0": {"type": "u-reaction"}},
        ),
        b_reaction=ReactionSpec(
            "PaSR", reactions={"r0": {"type": "b-reaction"}},
        ),
    )
    root = tmp_path / "myCase"
    case = _build(setup, root)
    case.write()
    u_reactions = case.read_dict("constant/u.reactions.dat")
    b_reactions = case.read_dict("constant/b.reactions.dat")
    assert str(u_reactions["reactions"]["r0"]["type"]) == "u-reaction"
    assert str(b_reactions["reactions"]["r0"]["type"]) == "b-reaction"


def test_both_phases_load_balanced_pyjac_raises():
    with pytest.raises(ValueError, match="loadBalanced_pyJac"):
        _setup(
            u_reaction=ReactionSpec(
                "laminar", chemistry_type=DLB_PYJAC,
            ),
            b_reaction=ReactionSpec(
                "laminar", chemistry_type=DLB_PYJAC,
            ),
        )


def test_one_phase_load_balanced_pyjac_is_fine(tmp_path):
    setup = _setup(
        u_reaction=ReactionSpec(
            "laminar", chemistry_type=DLB_PYJAC,
        ),
        b_reaction=ReactionSpec("laminar", chemistry_type=DLB),
    )
    case = _build(setup, tmp_path / "myCase")
    assert str(
        case["constant/chemistryProperties.u"]["type"]
    ) == "loadBalanced_pyJac"
    assert str(
        case["constant/chemistryProperties.b"]["type"]
    ) == "loadBalanced"
    libs = [v.data for v in case["system/controlDict"]["libs"]]
    assert libs == [
        "libpyfoamy.so", "libchemistryModel_DLB.so", "libODE_DLB.so",
    ]


# --- full roundtrip / oracle ---


def test_full_write_read_roundtrip(tmp_path):
    setup = _setup(
        mixture_type="inhomogeneousEGR",
        turbulence=TurbulenceSpec("RAS", "LaunderSharmaKE"),
        xi_model=TRANSPORT_XI_MODEL,
    )
    root = tmp_path / "myCase"
    case = _build(setup, root)
    case.write()

    assert (root / "system" / "controlDict").is_file()
    assert (root / "constant" / "physicalProperties.u").is_file()
    assert (root / "constant" / "physicalProperties.b").is_file()
    assert (root / "constant" / "momentumTransport").is_file()
    assert (root / "constant" / "momentumTransport.u").is_file()
    assert (root / "constant" / "momentumTransport.b").is_file()
    assert (root / "constant" / "combustionProperties").is_file()


@pytest.mark.openfoam
def test_written_case_is_valid_per_foam_dictionary(tmp_path):
    setup = _setup(
        mixture_type="inhomogeneousEGR",
        turbulence=TurbulenceSpec("RAS", "LaunderSharmaKE"),
        xi_model=TRANSPORT_XI_MODEL,
        u_reaction=ReactionSpec("PaSR", {"Cmix": 1.0}),
    )
    root = tmp_path / "myCase"
    case = _build(setup, root)
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

    assert foam_value("system/controlDict", "solver") == "XiFluid"
    assert (
        foam_value("constant/physicalProperties.u", "thermoType/mixture")
        == "uInhomogeneousEGR"
    )
    assert (
        foam_value("constant/physicalProperties.b", "thermoType/mixture")
        == "bInhomogeneous"
    )
    assert (
        foam_value("constant/momentumTransport.u", "simulationType")
        == "RAS"
    )
    assert (
        foam_value("constant/combustionProperties", "flameWrinkling/model")
        == "transport"
    )
    assert (
        foam_value(
            "constant/combustionProperties",
            "flameWrinkling/generationRate/model",
        )
        == "KTS"
    )
    assert (
        foam_value("constant/reactionProperties.u", "reactionModel")
        == "PaSR"
    )
    assert (
        foam_value("system/fvSolution", "solvers/fu.u/solver")
        == "PBiCGStab"
    )
    assert foam_value("system/fvSolution", "solvers/p/solver") == "GAMG"
    assert (
        foam_value("system/fvSolution", "solvers/rho/solver")
        == "diagonal"
    )


@pytest.mark.openfoam
def test_written_buoyant_case_is_valid_per_foam_dictionary(tmp_path):
    setup = _setup(buoyant=True)
    root = tmp_path / "myCase"
    case = _build(setup, root, gravity=(0, -9.81, 0))
    case.write()

    result = subprocess.run(
        ["foamDictionary", "-entry", "value", "-value",
         str(root / "constant" / "g")],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "( 0 -9.81 0 )"


@pytest.mark.openfoam
def test_control_dict_satisfies_real_time_construction(tmp_path):
    setup = _setup()
    root = tmp_path / "myCase"
    case = _build(setup, root)
    case.write()

    with pytest.raises(native.NativeError, match="points|polyMesh"):
        with native.open_case(tmp_path, "myCase"):
            pass


def test_xi_model_spec_coefficients_are_written_beside_the_sub_models():
    spec = XiModelSpec(
        "transport", eq=XiEqModelSpec("Gulder"),
        profile=XiProfileSpec("cubic"), generation_rate=XiGModelSpec("KTS"),
        coefficients={"strainReduction": False},
    )
    assert spec.to_dict() == {
        "model": "transport", "eq": {"model": "Gulder"},
        "profile": {"model": "cubic"}, "generationRate": {"model": "KTS"},
        "strainReduction": False,
    }
