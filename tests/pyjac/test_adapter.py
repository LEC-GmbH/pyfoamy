import math

import pytest

ct = pytest.importorskip("cantera")
pytest.importorskip("numpy")

from pyfoamy.cantera.mechanism import Mechanism  # noqa: E402
from pyfoamy.pyjac import adapter  # noqa: E402
from pyfoamy.pyjac.records import GAS_CONSTANT_SI  # noqa: E402


@pytest.fixture(scope="module")
def gri30():
    return Mechanism("gri30.yaml")


@pytest.fixture(scope="module")
def gri30_gas():
    return ct.Solution("gri30.yaml")


def test_extract_species_and_reaction_counts(gri30, gri30_gas):
    species, reactions = adapter.extract(gri30, "N2")
    assert len(species) == gri30_gas.n_species
    assert len(reactions) == gri30_gas.n_reactions


def test_extract_moves_last_spec_to_the_end(gri30):
    species, _reactions = adapter.extract(gri30, "N2")
    assert species[-1].name == "N2"
    assert all(sp.name != "N2" for sp in species[:-1])


def test_extract_preserves_relative_order_of_other_species(
    gri30, gri30_gas,
):
    species, _reactions = adapter.extract(gri30, "N2")
    expected = [n for n in gri30_gas.species_names if n != "N2"]
    assert [sp.name for sp in species[:-1]] == expected


def test_extract_last_spec_is_case_insensitive(gri30):
    species, _reactions = adapter.extract(gri30, "n2")
    assert species[-1].name == "N2"


def test_extract_unknown_last_spec_raises(gri30):
    with pytest.raises(ValueError, match="no species named"):
        adapter.extract(gri30, "not_a_real_species")


def test_extract_species_thermo_matches_cantera(gri30, gri30_gas):
    species, _reactions = adapter.extract(gri30, "N2")
    by_name = {sp.name: sp for sp in species}
    h2 = by_name["H2"]
    ct_h2 = gri30_gas.species("H2")
    assert h2.mw == pytest.approx(ct_h2.molecular_weight)
    coeffs = ct_h2.thermo.coeffs
    assert h2.t_mid == pytest.approx(coeffs[0])
    assert list(h2.high) == pytest.approx(list(coeffs[1:8]))
    assert list(h2.low) == pytest.approx(list(coeffs[8:15]))


def test_extract_elementary_reaction(gri30, gri30_gas):
    species, reactions = adapter.extract(gri30, "N2")
    by_name = {sp.name: i for i, sp in enumerate(species)}
    idx = next(
        i for i, r in enumerate(gri30_gas.reactions())
        if r.reaction_type == "Arrhenius"
    )
    ct_r = gri30_gas.reactions()[idx]
    r = reactions[idx]
    assert r.third_body is False
    assert r.pressure_dependent is False
    assert r.reactants == [by_name[n] for n in ct_r.reactants]
    assert r.A == pytest.approx(ct_r.rate.pre_exponential_factor)
    assert r.b == pytest.approx(ct_r.rate.temperature_exponent)
    assert r.E == pytest.approx(
        ct_r.rate.activation_energy / GAS_CONSTANT_SI,
    )


def test_extract_third_body_reaction_efficiencies(gri30, gri30_gas):
    species, reactions = adapter.extract(gri30, "N2")
    by_name = {sp.name: i for i, sp in enumerate(species)}
    idx = next(
        i for i, r in enumerate(gri30_gas.reactions())
        if r.equation == "2 O + M <=> O2 + M"
    )
    r = reactions[idx]
    assert r.third_body is True
    assert r.pdep_species is None
    eff_by_index = dict(r.third_body_efficiencies)
    assert eff_by_index[by_name["AR"]] == pytest.approx(0.83)
    assert eff_by_index[by_name["H2O"]] == pytest.approx(15.4)
    # only non-default (!= 1.0) efficiencies are listed.
    assert all(eff != 1.0 for eff in eff_by_index.values())


def test_extract_explicit_single_third_body_uses_pdep_species(
    gri30, gri30_gas,
):
    # real gri30 reaction written with one explicit collision-partner
    # species instead of the generic "M" symbol.
    species, reactions = adapter.extract(gri30, "N2")
    by_name = {sp.name: i for i, sp in enumerate(species)}
    idx = next(
        i for i, r in enumerate(gri30_gas.reactions())
        if r.equation == "H + O2 + O2 <=> HO2 + O2"
    )
    r = reactions[idx]
    assert r.third_body is True
    assert r.pdep_species == by_name["O2"]
    assert not r.third_body_efficiencies


def test_extract_falloff_troe_reaction(gri30, gri30_gas):
    species, reactions = adapter.extract(gri30, "N2")
    idx = next(
        i for i, r in enumerate(gri30_gas.reactions())
        if r.reaction_type == "falloff-Troe"
    )
    ct_r = gri30_gas.reactions()[idx]
    r = reactions[idx]
    assert r.pressure_dependent is True
    # pyJac's ReacInfo.thd_body and .pdep are mutually exclusive.
    assert r.third_body is False
    assert r.low_rate_params is not None
    assert r.high_rate_params is None
    assert r.low_rate_params[0] == pytest.approx(
        ct_r.rate.low_rate.pre_exponential_factor,
    )
    assert r.A == pytest.approx(ct_r.rate.high_rate.pre_exponential_factor)
    assert r.troe_params == pytest.approx(
        tuple(float(x) for x in ct_r.rate.falloff_coeffs),
    )
    assert r.sri_params is None


def test_extract_never_sets_explicit_reverse_params(gri30):
    _species, reactions = adapter.extract(gri30, "N2")
    assert all(r.reverse_params is None for r in reactions)


def test_extract_rejects_non_default_reaction_orders(gri30_gas):
    # gri30 species, synthetic reaction with a non-default order.
    h = gri30_gas.species("H")
    h2 = gri30_gas.species("H2")
    reaction = ct.Reaction(
        {"H": 2.0}, {"H2": 1.0}, ct.Arrhenius(1e10, 0.0, 0.0),
    )
    reaction.reversible = False
    reaction.orders = {"H": 1.5}
    gas = ct.Solution(
        thermo="ideal-gas", kinetics="gas", species=[h, h2],
        reactions=[reaction],
    )
    mechanism = Mechanism.from_yaml(gas.write_yaml())
    with pytest.raises(ValueError, match="non-default reaction orders"):
        adapter.extract(mechanism, "H2")


def _three_body_mechanism(gri30_gas, efficiencies):
    names = ["H", "O", "OH", "H2", "N2"]
    reaction = ct.Reaction.from_yaml(
        "{equation: H + O + M <=> OH + M, type: three-body, "
        "rate-constant: {A: 5.0e+17, b: -1.0, Ea: 0.0}, "
        f"default-efficiency: 0.0, efficiencies: {efficiencies}}}",
        ct.Solution(
            thermo="ideal-gas", kinetics="gas",
            species=[gri30_gas.species(n) for n in names],
        ),
    )
    gas = ct.Solution(
        thermo="ideal-gas", kinetics="gas",
        species=[gri30_gas.species(n) for n in names], reactions=[reaction],
    )
    return Mechanism.from_yaml(gas.write_yaml())


def test_extract_expands_a_non_unit_default_efficiency(gri30_gas):
    mechanism = _three_body_mechanism(gri30_gas, "{H2: 2.5, OH: 1.0}")
    species, (reaction,) = adapter.extract(mechanism, "N2")
    index = {sp.name: i for i, sp in enumerate(species)}
    assert reaction.third_body is True
    assert dict(reaction.third_body_efficiencies) == {
        index["H"]: 0.0, index["O"]: 0.0, index["H2"]: 2.5,
        index["N2"]: 0.0,
    }


def test_extract_rebases_entropy_to_one_atmosphere(gri30_gas):
    species = []
    for name in ["O2", "N2"]:
        data = gri30_gas.species(name).input_data
        data["thermo"]["reference-pressure"] = 1e5
        species.append(ct.Species.from_dict(data))
    gas = ct.Solution(thermo="ideal-gas", kinetics="gas", species=species)
    species, _ = adapter.extract(Mechanism.from_yaml(gas.write_yaml()), "N2")
    T = 1500.0
    a = species[0].high
    s_over_r = (
        a[0] * math.log(T) + a[1] * T + a[2] * T**2 / 2
        + a[3] * T**3 / 3 + a[4] * T**4 / 4 + a[6]
    )
    gas.TPX = T, 101325.0, "O2:1"
    assert s_over_r * ct.gas_constant == pytest.approx(
        gas.entropy_mole, rel=1e-12,
    )


def test_extract_rejects_non_nasa7_thermo():
    mechanism = Mechanism("airNASA9.yaml")
    with pytest.raises(ValueError, match="NASA9 thermo, pyJac needs NASA7"):
        adapter.extract(mechanism, "N2")
