import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ct = pytest.importorskip("cantera")
pytest.importorskip("numpy")

from pyfoamy import Case  # noqa: E402
import pyfoamy.cantera.mechanism as mechanism_module  # noqa: E402
from pyfoamy.cantera.mechanism import Mechanism  # noqa: E402
from pyfoamy.cantera.reactions import ReactionConverter  # noqa: E402
from pyfoamy.openfoam_version import use_openfoam_version  # noqa: E402
from pyfoamy.parsing.grammar import String  # noqa: E402

_MECHANISM = Path(__file__).resolve().parent.parent.parent / "converted.yml"


def _skip_unless_mechanism():
    if not _MECHANISM.is_file():
        pytest.skip("converted.yml not present in the repo root")


@pytest.fixture(autouse=True)
def openfoam_14():
    with use_openfoam_version("14"):
        yield


@pytest.fixture
def dev():
    with use_openfoam_version("dev"):
        yield


@pytest.fixture
def mech():
    _skip_unless_mechanism()
    return Mechanism(str(_MECHANISM))


@pytest.fixture
def converter(mech):
    return ReactionConverter(mechanism=mech)


def _fake_mechanism(gas):
    # A real Mechanism instance without __init__, so Mechanism.coerce()
    # passes it through.
    fake = Mechanism.__new__(Mechanism)
    fake._gas = gas
    return fake


def _first_of_type(gas, reaction_type):
    for reaction in gas.reactions():
        if reaction.reaction_type == reaction_type:
            return reaction
    pytest.skip(f"no {reaction_type} reaction in converted.yml")


def test_missing_cantera_raises_import_error(monkeypatch, mech):
    monkeypatch.setattr(mechanism_module, "ct", None)
    monkeypatch.setattr(
        mechanism_module, "_CANTERA_IMPORT_ERROR", ImportError("boom"),
    )
    with pytest.raises(ImportError, match="pyfoamy.cantera requires cantera"):
        ReactionConverter(mechanism=mech)


def test_reaction_converter_shares_a_given_mechanism(mech):
    converter = ReactionConverter(mechanism=mech)
    assert converter.mechanism is mech


def test_arrhenius_reaction_matches_real_dict_shape(converter, mech):
    reaction = _first_of_type(mech.gas, "Arrhenius")
    result = converter.reaction_dict(reaction)
    assert result["type"] in (
        "reversibleArrhenius", "irreversibleArrhenius",
    )
    assert result["A"] == reaction.rate.pre_exponential_factor
    assert result["beta"] == reaction.rate.temperature_exponent
    assert result["Ea"] == reaction.rate.activation_energy
    assert set(result) == {"type", "reaction", "A", "beta", "Ea"}


def _assert_sparse_efficiencies(entry, third_body, gas):
    assert entry["defaultEfficiency"] == third_body.default_efficiency
    listed = dict(entry.get("efficiencies", []))
    assert listed == {
        name: third_body.efficiency(name) for name in gas.species_names
        if third_body.efficiency(name) != third_body.default_efficiency
    }
    assert [name for name in gas.species_names if name in listed] == list(
        listed
    )


def test_three_body_reaction_lists_every_efficiency_for_14(converter, mech):
    reaction = _first_of_type(mech.gas, "three-body-Arrhenius")
    result = converter.reaction_dict(reaction)
    assert result["type"] in (
        "reversibleThirdBodyArrhenius", "irreversibleThirdBodyArrhenius",
    )
    assert "F" not in result
    assert "defaultEfficiency" not in result
    assert result["coeffs"] == [
        [name, reaction.third_body.efficiency(name)]
        for name in mech.gas.species_names
    ]
    assert " M" not in str(result["reaction"])


def test_three_body_reaction_has_default_and_sparse_efficiencies_for_dev(
    converter, mech, dev,
):
    reaction = _first_of_type(mech.gas, "three-body-Arrhenius")
    result = converter.reaction_dict(reaction)
    assert result["type"] in (
        "Reversible<ThirdBody<Arrhenius>>",
        "Irreversible<ThirdBody<Arrhenius>>",
    )
    assert "coeffs" not in result
    _assert_sparse_efficiencies(result, reaction.third_body, mech.gas)


def test_third_body_efficiencies_omitted_when_all_default(dev):
    species = [
        ct.Species.from_yaml(
            f"name: {name}\ncomposition: {{H: {n}}}\n"
            "thermo: {model: constant-cp}"
        )
        for name, n in (("H", 1), ("H2", 2))
    ]
    kinetics = ct.Solution(thermo="ideal-gas", kinetics="gas", species=species)
    reaction = ct.Reaction.from_yaml(
        "equation: 2 H + M <=> H2 + M\ntype: three-body\n"
        "rate-constant: {A: 1.0e+12, b: 0.0, Ea: 0.0}",
        kinetics,
    )
    gas = ct.Solution(
        thermo="ideal-gas", kinetics="gas", species=species,
        reactions=[reaction],
    )
    result = ReactionConverter(
        mechanism=_fake_mechanism(gas),
    ).reaction_dict(gas.reaction(0))
    assert result["defaultEfficiency"] == 1.0
    assert "efficiencies" not in result


def test_falloff_troe_reaction_has_nested_rate_and_thirdbody(
    converter, mech, dev,
):
    reaction = _first_of_type(mech.gas, "falloff-Troe")
    result = converter.reaction_dict(reaction)
    assert result["type"] in (
        "Reversible<FallOff<Arrhenius,Troe>>",
        "Irreversible<FallOff<Arrhenius,Troe>>",
    )
    assert set(result["k0"]) == {"A", "beta", "Ea"}
    assert set(result["kInf"]) == {"A", "beta", "Ea"}
    assert set(result["F"]) == {"alpha", "Tsss", "Ts", "Tss"}
    _assert_sparse_efficiencies(
        result["thirdBodyEfficiencies"], reaction.third_body, mech.gas,
    )
    assert "(+M)" not in str(result["reaction"])


def test_falloff_lindemann_reaction_has_empty_f(converter, mech, dev):
    reaction = _first_of_type(mech.gas, "falloff-Lindemann")
    result = converter.reaction_dict(reaction)
    assert result["F"] == {}
    _assert_sparse_efficiencies(
        result["thirdBodyEfficiencies"], reaction.third_body, mech.gas,
    )


def test_falloff_third_body_for_14_lists_every_species(converter, mech):
    reaction = _first_of_type(mech.gas, "falloff-Troe")
    result = converter.reaction_dict(reaction)
    assert result["type"] in (
        "reversibleArrheniusTroeFallOff", "irreversibleArrheniusTroeFallOff",
    )
    assert [name for name, _ in result["thirdBodyEfficiencies"]["coeffs"]] \
        == mech.gas.species_names


def test_unsupported_reaction_type_raises(converter):
    fake = SimpleNamespace(
        reaction_type="chemically-activated-Arrhenius",
        equation="A = B", orders=None,
    )
    with pytest.raises(ValueError, match="is not supported"):
        converter.reaction_dict(fake)


def test_non_default_reaction_orders_raises(converter, mech):
    reaction = _first_of_type(mech.gas, "Arrhenius")
    fake = SimpleNamespace(
        reaction_type=reaction.reaction_type,
        orders={"A": 2.5},
    )
    with pytest.raises(ValueError, match="non-default reaction orders"):
        converter.reaction_dict(fake)


def test_reactions_dict_converts_every_reaction(converter, mech):
    result = converter.reactions_dict()
    assert len(result) == mech.gas.n_reactions


def _pre_exponential_factors(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "A":
                yield item
            else:
                yield from _pre_exponential_factors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _pre_exponential_factors(item)


def test_rate_multiplier_scales_every_pre_exponential_factor():
    converter = ReactionConverter(mechanism="h2o2.yaml")
    base = list(_pre_exponential_factors(converter.reactions_dict()))
    scaled = list(_pre_exponential_factors(converter.reactions_dict(2.0)))
    # More factors than reactions: falloff reactions carry k0 and kInf.
    assert len(base) > converter.mechanism.gas.n_reactions
    assert scaled == pytest.approx([2.0 * a for a in base])


def test_reactions_dict_groups_identical_failures():
    gas = SimpleNamespace(
        species_names=["A", "B"],
        n_species=2,
        reactions=lambda: [
            SimpleNamespace(
                reaction_type="foo-type", equation=f"foo{i} = bar",
                orders=None,
            )
            for i in range(3)
        ] + [
            SimpleNamespace(
                reaction_type="bar-type", equation="baz = qux", orders=None,
            )
        ],
    )
    converter = ReactionConverter(mechanism=_fake_mechanism(gas))
    with pytest.raises(ValueError) as excinfo:
        converter.reactions_dict()
    message = str(excinfo.value)
    assert "3 reaction(s)" in message
    assert "1 reaction(s)" in message
    assert message.count("reaction type 'foo-type' is not supported") == 1
    assert message.count("reaction type 'bar-type' is not supported") == 1


def _synthetic_gas(reactions):
    yaml_text = f"""
phases:
- name: gas
  thermo: ideal-gas
  elements: [H, O]
  species: [H2, O2, H2O]
  kinetics: gas
  reactions: all
  state: {{T: 300, P: 1 atm}}
species:
- name: H2
  composition: {{H: 2}}
  thermo: {{model: NASA7, temperature-ranges: [200, 1000, 6000],
    data: [[3.3, 0, 0, 0, 0, -1000, -3.3], [3.3, 0, 0, 0, 0, -1000, -3.3]]}}
- name: O2
  composition: {{O: 2}}
  thermo: {{model: NASA7, temperature-ranges: [200, 1000, 6000],
    data: [[3.6, 0, 0, 0, 0, -1000, -3.6], [3.6, 0, 0, 0, 0, -1000, -3.6]]}}
- name: H2O
  composition: {{H: 2, O: 1}}
  thermo: {{model: NASA7, temperature-ranges: [200, 1000, 6000],
    data: [[4.2, 0, 0, 0, 0, -30000, -4.2],
           [4.2, 0, 0, 0, 0, -30000, -4.2]]}}
reactions:
{reactions}
"""
    return ct.Solution(yaml=yaml_text)


def _synthetic_plog_gas(rate_constants):
    return _synthetic_gas(
        "- equation: 2 H2 + O2 <=> 2 H2O\n"
        "  type: pressure-dependent-Arrhenius\n"
        "  rate-constants:\n"
        f"{rate_constants}"
    )


_SINGLE_CHANNEL_PLOG = (
    "  - {P: 1.0 atm, A: 1.0e+10, b: 0.0, Ea: 5000 cal/mol}"
)
# Given high pressure first, so the written table has to be sorted.
_THREE_PRESSURE_PLOG = (
    "  - {P: 100.0 atm, A: 4.0e+14, b: -0.5, Ea: 30000 cal/mol}\n"
    "  - {P: 0.1 atm, A: 1.0e+12, b: 0.3, Ea: 20000 cal/mol}\n"
    "  - {P: 1.0 atm, A: 3.0e+13, b: 0.0, Ea: 25000 cal/mol}"
)
_DUPLICATE_PRESSURE_PLOG = (
    "  - {P: 1.0 atm, A: 1.0e+10, b: 0.0, Ea: 5000 cal/mol}\n"
    "  - {P: 1.0 atm, A: 1.0e+20, b: 0.0, Ea: 80000 cal/mol}"
)


def _plog_k(coeffs, T, p):
    # OpenFOAM's pLogReactionRate: ln k linear in ln p, clamped at the ends
    def lnk(entry):
        _, A, beta, Ta = entry
        return math.log(A) + beta*math.log(T) - Ta/T

    if p <= coeffs[0][0]:
        return math.exp(lnk(coeffs[0]))
    if p >= coeffs[-1][0]:
        return math.exp(lnk(coeffs[-1]))
    i = next(i for i, entry in enumerate(coeffs) if p <= entry[0])
    f = (
        (math.log(p) - math.log(coeffs[i - 1][0]))
        / (math.log(coeffs[i][0]) - math.log(coeffs[i - 1][0]))
    )
    return math.exp((1 - f)*lnk(coeffs[i - 1]) + f*lnk(coeffs[i]))


def _assert_plog_matches_cantera(rate, coeffs):
    pressures = [entry[0] for entry in coeffs]
    probes = (
        [pressures[0]/10, pressures[-1]*10]
        + pressures
        + [math.sqrt(a*b) for a, b in zip(pressures, pressures[1:])]
    )
    for T in (500.0, 1000.0, 2000.0):
        for p in probes:
            assert _plog_k(coeffs, T, p) == pytest.approx(
                rate(T, p), rel=1e-9,
            ), (T, p)


def test_plog_reaction_becomes_sorted_plog_table():
    gas = _synthetic_plog_gas(_THREE_PRESSURE_PLOG)
    converter = ReactionConverter(mechanism=_fake_mechanism(gas))
    result = converter.reaction_dict(gas.reactions()[0])
    assert set(result) == {"type", "reaction", "coeffs"}
    assert result["type"] == "reversiblePLog"
    with use_openfoam_version("dev"):
        assert converter.reaction_dict(gas.reactions()[0])["type"] == (
            "Reversible<pLog>"
        )
    coeffs = result["coeffs"]
    assert [entry[0] for entry in coeffs] == pytest.approx(
        [0.1*ct.one_atm, ct.one_atm, 100*ct.one_atm],
    )
    p, A, beta, Ta = coeffs[0]
    assert (A, beta) == pytest.approx((1.0e12, 0.3))
    assert Ta == pytest.approx(20000*4184.0/ct.gas_constant)


def test_plog_table_reproduces_cantera_rate():
    gas = _synthetic_plog_gas(_THREE_PRESSURE_PLOG)
    converter = ReactionConverter(mechanism=_fake_mechanism(gas))
    reaction = gas.reactions()[0]
    _assert_plog_matches_cantera(
        reaction.rate, converter.reaction_dict(reaction)["coeffs"],
    )


def test_plog_tables_of_a_real_mechanism_reproduce_cantera():
    plog_mechanism = _MECHANISM.with_name("converted2.yml")
    if not plog_mechanism.is_file():
        pytest.skip("converted2.yml not present in the repo root")
    mech = Mechanism(str(plog_mechanism))
    converter = ReactionConverter(mechanism=mech)
    plogs = [
        reaction for reaction in mech.gas.reactions()
        if reaction.reaction_type == "pressure-dependent-Arrhenius"
    ]
    assert plogs
    for reaction in plogs:
        _assert_plog_matches_cantera(
            reaction.rate, converter.reaction_dict(reaction)["coeffs"],
        )


def test_plog_duplicate_pressures_raise():
    gas = _synthetic_plog_gas(_DUPLICATE_PRESSURE_PLOG)
    converter = ReactionConverter(mechanism=_fake_mechanism(gas))
    with pytest.raises(ValueError, match="duplicate pressures"):
        converter.reaction_dict(gas.reactions()[0])


def _explicit_collider_converter(reaction_yaml):
    gas = _synthetic_gas(reaction_yaml)
    converter = ReactionConverter(mechanism=_fake_mechanism(gas))
    return converter, gas.reactions()[0]


def test_explicit_collider_plog_is_not_supported():
    converter, reaction = _explicit_collider_converter(
        "- equation: 2 H2 + O2 + H2O <=> 2 H2O + H2O\n"
        "  type: pressure-dependent-Arrhenius\n"
        "  rate-constants:\n"
        f"{_SINGLE_CHANNEL_PLOG}"
    )
    assert reaction.reaction_type == (
        "three-body-pressure-dependent-Arrhenius"
    )
    with pytest.raises(ValueError, match="is not supported"):
        converter.reaction_dict(reaction)


def test_explicit_collider_three_body_is_stripped_from_equation(dev):
    # Cantera treats an untyped reaction's spectator species as an
    # explicit three-body collider, already excluded from .reactants.
    converter, reaction = _explicit_collider_converter(
        "- equation: 2 H2 + O2 + H2O <=> 2 H2O + H2O\n"
        "  rate-constant: {A: 1.0e+10, b: 0.0, Ea: 0.0}"
    )
    assert reaction.reaction_type == "three-body-Arrhenius"
    result = converter.reaction_dict(reaction)
    assert result["reaction"] == String("2 H2 + O2 = 2 H2O")
    assert result["defaultEfficiency"] == 0
    assert result["efficiencies"] == [["H2O", 1.0]]


def test_explicit_collider_falloff_is_stripped_from_equation(dev):
    converter, reaction = _explicit_collider_converter(
        "- equation: 2 H2 + O2 (+H2O) <=> 2 H2O (+H2O)\n"
        "  type: falloff\n"
        "  low-P-rate-constant: {A: 1.0e+16, b: 0.0, Ea: 0.0}\n"
        "  high-P-rate-constant: {A: 1.0e+10, b: 0.0, Ea: 0.0}"
    )
    result = converter.reaction_dict(reaction)
    assert result["reaction"] == String("2 H2 + O2 = 2 H2O")
    assert result["thirdBodyEfficiencies"] == {
        "defaultEfficiency": 0, "efficiencies": [["H2O", 1.0]],
    }


@pytest.mark.openfoam
def test_case_write_reactions_round_trips_through_foam_dictionary(
    converter, mech, tmp_path,
):
    reactions_by_type = {}
    for reaction in mech.gas.reactions():
        reactions_by_type.setdefault(reaction.reaction_type, reaction)

    entries = {
        rtype.replace("-", "_"): converter.reaction_dict(reaction)
        for rtype, reaction in reactions_by_type.items()
    }

    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case["constant/reactions"] = {"reactions": entries}
    case.write()

    result = subprocess.run(
        ["foamDictionary", "-expand", "constant/reactions"],
        capture_output=True, text=True, cwd=tmp_path / "myCase",
    )
    assert result.returncode == 0, result.stderr
    for name in entries:
        assert name in result.stdout
