import subprocess

import pytest

from pyfoamy import native
from pyfoamy.enums import CustomName
from pyfoamy.models import (
    DLB,
    DLB_PYJAC,
    LinearSolver,
    MulticomponentFluidSetup,
    ReactionSpec,
    ThermoTypeSpec,
    TurbulenceSpec,
)
from pyfoamy.parsing.grammar import Directive

P_SOLVER = LinearSolver("multigrid", tolerance=1e-7, rel_tol=0.01)
MOMENTUM_SOLVER = LinearSolver(
    "conjugate_gradient", tolerance=1e-6, rel_tol=0.1
)

GAS_THERMO = ThermoTypeSpec(
    type="hePsiThermo",
    mixture="multicomponentMixture",
    transport="sutherland",
    thermo="janaf",
    equation_of_state="perfectGas",
    energy="sensibleEnthalpy",
)

SPECIES = {
    "CH4": {"specie": {"molWeight": 16.04}},
    "O2": {"specie": {"molWeight": 31.9988}},
    "N2": {"specie": {"molWeight": 28.0134}},
}


def _words(entry):
    return str(entry).split()


def _build(setup, root, **kwargs):
    kwargs.setdefault("ddt_order", "steady")
    kwargs.setdefault("div_order", "firstOrder")
    kwargs.setdefault("p_solver", P_SOLVER)
    kwargs.setdefault("momentum_solver", MOMENTUM_SOLVER)
    return setup.build(root, **kwargs)


def _setup(**kwargs):
    kwargs.setdefault("thermo_type", GAS_THERMO)
    kwargs.setdefault("default_specie", "N2")
    kwargs.setdefault("species", SPECIES)
    return MulticomponentFluidSetup(**kwargs)


# --- ReactionSpec ---


def test_reaction_spec_rejects_unknown_model():
    with pytest.raises(ValueError, match="FSD"):
        ReactionSpec("FSD")


def test_reaction_spec_no_reaction_writes_nothing():
    spec = ReactionSpec("noReaction")
    assert spec.reaction_properties_dict() is None
    assert spec.chemistry_properties_dict() is None


def test_reaction_spec_pasr_reaction_properties():
    spec = ReactionSpec("PaSR", coefficients={"Cmix": 1.0})
    assert spec.reaction_properties_dict() == {
        "reactionModel": "PaSR",
        "PaSR": {"Cmix": 1.0},
    }


def test_reaction_spec_edc_with_version():
    spec = ReactionSpec("EDC", coefficients={"version": "v2016"})
    assert spec.reaction_properties_dict() == {
        "reactionModel": "EDC",
        "EDC": {"version": "v2016"},
    }


def test_reaction_spec_chemistry_properties_default_standard():
    spec = ReactionSpec("laminar")
    assert spec.chemistry_properties_dict() == {"type": "standard"}


def test_reaction_spec_chemistry_properties_load_balanced():
    spec = ReactionSpec(
        "PaSR",
        chemistry_type=DLB,
        chemistry_coefficients={"skipSpecies": True},
    )
    result = spec.chemistry_properties_dict()
    assert result["type"] == "loadBalanced"
    assert result["skipSpecies"] is True


def test_reaction_spec_chemistry_properties_load_balanced_pyjac():
    spec = ReactionSpec("laminar", chemistry_type=DLB_PYJAC)
    result = spec.chemistry_properties_dict()
    assert result["type"] == "loadBalanced_pyJac"


def test_reaction_spec_required_libs():
    assert ReactionSpec("laminar").required_libs() == []
    assert ReactionSpec(
        "laminar", chemistry_type=DLB,
    ).required_libs() == ["libchemistryModel_DLB.so", "libODE_DLB.so"]
    assert ReactionSpec(
        "laminar", chemistry_type=DLB_PYJAC,
    ).required_libs() == ["libchemistryModel_DLB.so", "libODE_DLB.so"]


def test_reaction_spec_loadbalancing_subdict():
    spec = ReactionSpec(
        "laminar", chemistry_type=DLB,
        loadbalancing={"active": True, "log": True},
    )
    result = spec.chemistry_properties_dict()
    assert result["loadbalancing"] == {"active": True, "log": True}


def test_reaction_spec_rejects_type_in_chemistry_coefficients():
    with pytest.raises(ValueError, match="chemistry_type"):
        ReactionSpec(
            "laminar",
            chemistry_coefficients={"type": "loadBalanced_pyJac"},
        )


def test_reaction_spec_chemistry_properties_includes_ode_and_time_step():
    spec = ReactionSpec(
        "laminar",
        initial_chemical_time_step=1e-07,
        ode={"solver": "seulex", "absTol": 1e-8, "relTol": 1e-1},
    )
    result = spec.chemistry_properties_dict()
    assert result["initialChemicalTimeStep"] == 1e-07
    assert result["ode"] == {
        "solver": "seulex", "absTol": 1e-8, "relTol": 1e-1,
    }


def test_reaction_spec_chemistry_properties_includes_reactions_inline():
    spec = ReactionSpec(
        "laminar",
        reactions={"r0": {"type": "irreversibleArrhenius"}},
        reactions_file=None,
    )
    result = spec.chemistry_properties_dict()
    assert result["reactions"] == {
        "r0": {"type": "irreversibleArrhenius"},
    }


def test_reaction_spec_chemistry_properties_omits_reactions_by_default():
    # reactions_file defaults to "reactions.dat": not inline.
    spec = ReactionSpec(
        "laminar", reactions={"r0": {"type": "irreversibleArrhenius"}},
    )
    result = spec.chemistry_properties_dict()
    assert "reactions" not in result


def test_reaction_spec_reactions_file_without_reactions_is_a_no_op():
    spec = ReactionSpec("laminar", reactions_file="reactions.dat")
    assert "reactions" not in spec.chemistry_properties_dict()


def test_reaction_spec_no_reaction_rejects_reactions():
    with pytest.raises(ValueError, match="noReaction"):
        ReactionSpec("noReaction", reactions={"r0": {}})


# --- MulticomponentFluidSetup ---


def test_default_specie_must_be_in_species():
    with pytest.raises(ValueError, match="default_specie"):
        _setup(default_specie="Ar")


def test_required_fields_includes_all_species_including_default(
):
    setup = _setup()
    fields = setup.required_fields()
    assert "CH4" in fields
    assert "O2" in fields
    assert "N2" in fields  # the default specie, still a real field


def test_required_fields_base_fields_present():
    setup = _setup()
    fields = setup.required_fields()
    assert fields[:3] == ["p", "U", "T"]


def test_build_writes_a_bare_solver_entry(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    assert str(case["system/controlDict"]["solver"]) == (
        "multicomponentFluid"
    )


def test_build_writes_default_specie_and_species(tmp_path):
    # species_file=None opts back into the old inline-embedded shape.
    setup = _setup(species_file=None)
    case = _build(setup, tmp_path / "myCase")
    d = case["constant/physicalProperties"]
    assert str(d["defaultSpecie"]) == "N2"
    ch4 = d["CH4"]
    assert float(ch4["specie"]["molWeight"]) == 16.04


def test_physical_properties_has_no_mixture_subdict(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    assert "mixture" not in case["constant/physicalProperties"]


def test_physical_properties_includes_species_list(tmp_path):
    # `species` is required (multicomponentMixture.C lookup);
    # species_file=None embeds the specie data inline.
    setup = _setup(species_file=None)
    case = _build(setup, tmp_path / "myCase")
    species = [
        str(v) for v in case["constant/physicalProperties"]["species"]
    ]
    assert set(species) == {"CH4", "O2", "N2"}


def test_species_file_defaults_to_species_dat(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    directive = case["constant/physicalProperties"][-1]
    assert str(directive.args) == '"$FOAM_CASE/constant/species.dat"'


def test_div_schemes_bundle_species_and_energy_into_yi_h(tmp_path):
    setup = _setup()
    case = _build(
        setup, tmp_path / "myCase", div_order="secondOrderBounded"
    )
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(phi,Yi_h)" in div
    assert "div(phi,h)" not in div
    assert _words(div["div(phi,Yi_h)"]) == [
        "Gauss", "limitedLinear", "1",
    ]


def test_div_schemes_include_multicomponent_specific_plain_u_term(
    tmp_path,
):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(U)"]) == ["Gauss", "linear"]


def test_div_schemes_no_fluid_specific_p_rho_term(tmp_path):
    # div(phi,(p|rho)) is FluidSetup-only; not in the tutorials here.
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(phi,(p|rho))" not in div


def test_div_schemes_still_include_shared_fixed_terms(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,K)"]) == ["Gauss", "linear"]
    assert _words(
        div["div(((rho*nuEff)*dev2(T(grad(U)))))"]
    ) == ["Gauss", "linear"]


def test_div_schemes_add_phid_p_when_transonic(tmp_path):
    setup = _setup(transonic=True)
    case = _build(setup, tmp_path / "myCase")
    div = case["system/fvSchemes"]["divSchemes"]
    assert "div(phid,p)" in div


def test_fv_solution_includes_species_in_momentum_group(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    solvers = case["system/fvSolution"]["solvers"]
    for name in ["CH4", "CH4Final", "O2", "O2Final", "N2", "N2Final"]:
        assert name in solvers, f"missing solver block for {name!r}"


def test_fv_solution_includes_yi_entry(tmp_path):
    # All species are solved under "Yi" (thermophysicalPredictor.C:
    # YiEqn.solve("Yi")).
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    solvers = case["system/fvSolution"]["solvers"]
    assert "Yi" in solvers
    assert "YiFinal" in solvers


def test_fv_solution_species_use_asymmetric_family(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    ch4 = case["system/fvSolution"]["solvers"]["CH4"]
    assert str(ch4["solver"]) == "PBiCGStab"


def test_no_reaction_writes_no_reaction_properties_files(tmp_path):
    setup = _setup()
    case = _build(setup, tmp_path / "myCase")
    case.write()
    root = tmp_path / "myCase"
    assert not (root / "constant" / "reactionProperties").exists()
    assert not (root / "constant" / "chemistryProperties").exists()


def test_pasr_writes_reaction_and_chemistry_properties(tmp_path):
    setup = _setup(reaction=ReactionSpec("PaSR", {"Cmix": 1.0}))
    case = _build(setup, tmp_path / "myCase")
    case.write()
    root = tmp_path / "myCase"
    assert (root / "constant" / "reactionProperties").is_file()
    assert (root / "constant" / "chemistryProperties").is_file()
    reaction = case.read_dict("constant/reactionProperties")
    assert str(reaction["reactionModel"]) == "PaSR"
    assert float(reaction["PaSR"]["Cmix"]) == 1.0


def test_load_balanced_chemistry_type_written(tmp_path):
    setup = _setup(
        reaction=ReactionSpec("laminar", chemistry_type=DLB)
    )
    case = _build(setup, tmp_path / "myCase")
    chemistry = case["constant/chemistryProperties"]
    assert str(chemistry["type"]) == "loadBalanced"
    libs = [v.data for v in case["system/controlDict"]["libs"]]
    assert libs == [
        "libpyfoamy.so", "libchemistryModel_DLB.so", "libODE_DLB.so",
    ]


def test_load_balanced_pyjac_chemistry_type_written(tmp_path):
    setup = _setup(
        reaction=ReactionSpec(
            "laminar", chemistry_type=DLB_PYJAC,
        )
    )
    case = _build(setup, tmp_path / "myCase")
    chemistry = case["constant/chemistryProperties"]
    assert str(chemistry["type"]) == "loadBalanced_pyJac"
    libs = [v.data for v in case["system/controlDict"]["libs"]]
    assert libs == [
        "libpyfoamy.so", "libchemistryModel_DLB.so", "libODE_DLB.so",
    ]


def test_standard_chemistry_type_adds_no_libs(tmp_path):
    setup = _setup(reaction=ReactionSpec("laminar"))
    case = _build(setup, tmp_path / "myCase", pyfoamy_lib=False)
    assert "libs" not in case["system/controlDict"]


def test_pyjac_species_order_mismatch_raises():
    with pytest.raises(ValueError, match="species order"):
        _setup(pyjac_species_order=["not", "the", "right", "order"])


def test_species_file_omits_inline_species(tmp_path):
    setup = _setup(species_file="species.dat")
    case = _build(setup, tmp_path / "myCase")
    physical_properties = case["constant/physicalProperties"]
    assert "species" not in physical_properties
    assert "CH4" not in physical_properties
    assert str(physical_properties["defaultSpecie"]) == "N2"


def test_species_file_appends_an_include_directive(tmp_path):
    setup = _setup(species_file="species.dat")
    case = _build(setup, tmp_path / "myCase")
    physical_properties = case["constant/physicalProperties"]
    directive = physical_properties[-1]
    assert isinstance(directive, Directive)
    assert directive.name == "#include"
    assert str(directive.args) == '"$FOAM_CASE/constant/species.dat"'


def test_species_file_is_written_headerless_with_full_content(tmp_path):
    setup = _setup(species_file="species.dat")
    root = tmp_path / "myCase"
    case = _build(setup, root)
    case.write()
    text = (root / "constant" / "species.dat").read_text()
    assert "FoamFile" not in text
    species_dict = case.read_dict("constant/species.dat")
    assert set(str(v) for v in species_dict["species"]) == {
        "CH4", "O2", "N2",
    }
    assert float(species_dict["CH4"]["specie"]["molWeight"]) == 16.04


def test_reactions_file_omits_inline_reactions_and_appends_include(
    tmp_path,
):
    setup = _setup(
        reaction=ReactionSpec(
            "laminar",
            reactions={"r0": {"type": "irreversibleArrhenius"}},
            reactions_file="reactions.dat",
        ),
    )
    root = tmp_path / "myCase"
    case = _build(setup, root)
    chemistry_properties = case["constant/chemistryProperties"]
    assert "reactions" not in chemistry_properties
    directive = chemistry_properties[-1]
    assert isinstance(directive, Directive)
    assert directive.name == "#include"
    assert str(directive.args) == '"$FOAM_CASE/constant/reactions.dat"'

    case.write()
    text = (root / "constant" / "reactions.dat").read_text()
    assert "FoamFile" not in text
    reactions_dict = case.read_dict("constant/reactions.dat")
    assert str(
        reactions_dict["reactions"]["r0"]["type"]
    ) == "irreversibleArrhenius"


def test_full_write_read_roundtrip(tmp_path):
    setup = _setup(
        reaction=ReactionSpec("EDC", {"version": "v2016"}),
        turbulence=TurbulenceSpec("RAS", "kEpsilon"),
    )
    root = tmp_path / "myCase"
    case = _build(setup, root, div_order="secondOrderBounded")
    case.write()

    reread = case.read_dict("constant/reactionProperties")
    assert str(reread["reactionModel"]) == "EDC"
    assert (root / "system" / "controlDict").is_file()
    assert (root / "constant" / "physicalProperties").is_file()
    assert (root / "constant" / "chemistryProperties").is_file()


@pytest.mark.openfoam
def test_written_case_is_valid_per_foam_dictionary(tmp_path):
    setup = _setup(
        reaction=ReactionSpec("PaSR", {"Cmix": 1.0}),
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

    assert foam_value("system/controlDict", "solver") == (
        "multicomponentFluid"
    )
    assert (
        foam_value("constant/physicalProperties", "defaultSpecie")
        == "N2"
    )
    assert (
        foam_value("constant/reactionProperties", "reactionModel")
        == "PaSR"
    )
    assert (
        foam_value("constant/chemistryProperties", "type")
        == "standard"
    )
    assert (
        foam_value("system/fvSchemes", "divSchemes/div(U)")
        == "Gauss linear"
    )
    assert (
        foam_value("system/fvSolution", "solvers/CH4/solver")
        == "PBiCGStab"
    )


@pytest.mark.openfoam
def test_control_dict_satisfies_real_time_construction(tmp_path):
    setup = _setup()
    root = tmp_path / "myCase"
    case = _build(setup, root)
    case.write()

    with pytest.raises(native.NativeError, match="points|polyMesh"):
        with native.open_case(tmp_path, "myCase"):
            pass


def test_reaction_spec_typed_chemistry_entries():
    spec = ReactionSpec(
        "laminar", chemistry_type=DLB, chemistry=False,
        reduction={"method": "none"},
        tabulation={"method": "ISAT", "tolerance": 1e-4},
        skip_species=True, skip_threshold=1e-5,
        ref_mapping={"active": True, "field": "Z"},
        cell_zone="cylinder",
    )
    result = spec.chemistry_properties_dict()
    assert result == {
        "type": "loadBalanced", "chemistry": False,
        "skipSpecies": True, "skipThreshold": 1e-5,
        "reduction": {"method": "none"},
        "tabulation": {"method": "ISAT", "tolerance": 1e-4},
        "refmapping": {"active": True, "field": "Z"},
        "cellZone": "cylinder",
    }


def test_reaction_spec_omits_unset_chemistry_entries():
    result = ReactionSpec("laminar").chemistry_properties_dict()
    assert result == {"type": "standard"}


def test_reaction_spec_accepts_third_party_names():
    spec = ReactionSpec(
        CustomName("myPaSR"), coefficients={"Cmix": 1.0},
        chemistry_type=CustomName("myChemistry"),
    )
    assert spec.reaction_properties_dict() == {
        "reactionModel": "myPaSR", "myPaSR": {"Cmix": 1.0},
    }
    assert spec.chemistry_properties_dict() == {"type": "myChemistry"}
    assert spec.required_libs() == []


def test_reaction_spec_rejects_unknown_plain_names():
    with pytest.raises(ValueError, match="unknown reaction model"):
        ReactionSpec("myPaSR")
