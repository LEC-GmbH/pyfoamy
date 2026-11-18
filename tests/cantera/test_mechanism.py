from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

ct = pytest.importorskip("cantera")
pytest.importorskip("numpy")
from cantera import ck2yaml, yaml2ck  # noqa: E402

import pyfoamy.cantera.mechanism as mechanism_module  # noqa: E402
from pyfoamy.cantera.mechanism import Mechanism  # noqa: E402
import chemkin_mechanisms  # noqa: E402

_MECHANISM = Path(__file__).resolve().parent.parent.parent / "converted.yml"


def _download_or_skip(name):
    # Third-party mechanisms are downloaded on first use; offline, the
    # tests skip.
    paths = chemkin_mechanisms.get(name)
    if paths is None:
        pytest.skip(f"could not download the {name!r} mechanism (no network?)")
    return paths


def _redundant_species_chemkin(h2, tmp_path):
    # ck2yaml reports "Found multiple declarations for species 'AR'";
    # permissive=True resolves it.
    text = h2.mechanism.read_text().replace(
        "SPECIES\n", "SPECIES\nAR\n", 1,
    )
    path = tmp_path / "h2_redundant_species.dat"
    path.write_text(text)
    return path


def _duplicate_reaction_chemkin(h2, tmp_path):
    # Duplicate reactions pass ck2yaml.convert() and fail only when
    # loaded (Kinetics::checkDuplicates); permissive=True does not help.
    lines = h2.mechanism.read_text().splitlines()
    out = []
    for line in lines:
        out.append(line)
        if line.startswith("H2 + O <=> H + OH"):
            out.append(line)
    path = tmp_path / "h2_duplicate_reaction.dat"
    path.write_text("\n".join(out) + "\n")
    return path


@pytest.fixture(scope="module")
def h2(tmp_path_factory):
    """Cantera's h2o2.yaml as CHEMKIN files; the transport file spells
    species lowercase, which ck2yaml rejects."""
    out = tmp_path_factory.mktemp("h2_chemkin")
    paths = SimpleNamespace(
        mechanism=out / "mech.ck",
        thermo=out / "thermo.ck",
        transport=out / "tran.ck",
        missing=out / "nope.dat",
    )
    yaml2ck.convert(
        ct.Solution("h2o2.yaml"), mechanism_path=paths.mechanism,
        thermo_path=paths.thermo, transport_path=paths.transport,
    )
    lines = paths.transport.read_text().splitlines()
    paths.transport.write_text("\n".join(
        line if line.startswith("!") else line.lower() for line in lines
    ) + "\n")
    return paths


def _skip_unless_mechanism():
    if not _MECHANISM.is_file():
        pytest.skip("converted.yml not present in the repo root")


@pytest.fixture
def mech():
    _skip_unless_mechanism()
    return Mechanism(str(_MECHANISM))


def test_missing_cantera_raises_import_error(monkeypatch):
    monkeypatch.setattr(mechanism_module, "ct", None)
    monkeypatch.setattr(
        mechanism_module, "_CANTERA_IMPORT_ERROR", ImportError("boom"),
    )
    with pytest.raises(ImportError, match="pyfoamy.cantera requires cantera"):
        Mechanism("converted.yml")


def test_coerce_returns_an_existing_mechanism_unchanged(mech):
    assert Mechanism.coerce(mech) is mech


def test_coerce_constructs_from_a_path():
    _skip_unless_mechanism()
    coerced = Mechanism.coerce(str(_MECHANISM))
    assert isinstance(coerced, Mechanism)


def test_coerce_rejects_reference_pressure_with_an_existing_mechanism(mech):
    with pytest.raises(ValueError, match="already-constructed Mechanism"):
        Mechanism.coerce(mech, reference_pressure=2e5)


def test_coerce_adopts_a_raw_cantera_solution():
    _skip_unless_mechanism()
    gas = ct.Solution(str(_MECHANISM))
    coerced = Mechanism.coerce(gas)
    assert coerced.gas is gas
    assert coerced.mechanism_path is None


def test_coerce_from_solution_accepts_reference_pressure():
    _skip_unless_mechanism()
    gas = ct.Solution(str(_MECHANISM))
    coerced = Mechanism.coerce(gas, reference_pressure=2e5)
    assert coerced.reference_pressure == 200000.0


def test_reference_pressure_override_accepts_decimal_and_stays_float():
    # Decimal from an OpenFOAM dict; the property is declared float.
    _skip_unless_mechanism()
    m = Mechanism(str(_MECHANISM), reference_pressure=Decimal("200000"))
    assert m.reference_pressure == 200000.0
    assert isinstance(m.reference_pressure, float)


def test_from_yaml_reconstructs_an_equivalent_mechanism(mech):
    reconstructed = Mechanism.from_yaml(mech.mechanism_yaml)
    assert reconstructed.gas.species_names == mech.gas.species_names
    assert reconstructed.molecular_weight(
        {"CH4": 1.0},
    ) == pytest.approx(mech.molecular_weight({"CH4": 1.0}))


def test_from_yaml_does_not_leave_a_dangling_temp_file(mech, tmp_path):
    # The temp file is deleted after loading; mechanism_path only
    # reports the origin.
    reconstructed = Mechanism.from_yaml(mech.mechanism_yaml)
    assert not Path(reconstructed.mechanism_path).exists()
    assert reconstructed.mechanism_yaml  # still works with no file


def test_from_yaml_accepts_reference_pressure():
    _skip_unless_mechanism()
    reconstructed = Mechanism.from_yaml(
        Mechanism(str(_MECHANISM)).mechanism_yaml, reference_pressure=2e5,
    )
    assert reconstructed.reference_pressure == 2e5


def test_restricted_contains_only_given_species(mech):
    gas = mech.restricted(["CH4", "N2"])
    assert sorted(gas.species_names) == ["CH4", "N2"]
    assert gas.transport_model == mech.gas.transport_model


def test_restricted_covers_the_full_species_range(mech):
    # gas.max_temp is the intersection over all species, narrower than
    # CH4's range; using it would extrapolate the fit.
    ch4_max = mech.gas.species("CH4").thermo.max_temp
    assert ch4_max > mech.gas.max_temp
    restricted = mech.restricted(["CH4"])
    assert restricted.max_temp >= ch4_max


def test_stoichiometric_air_fuel_ratio_matches_known_ch4_value(mech):
    # ~17.2 kg air / kg CH4
    afr = mech.stoichiometric_air_fuel_ratio(
        {"CH4": 1.0}, mech.air(basis="mole"), basis="mole"
    )
    assert afr == pytest.approx(17.2, rel=0.01)


def test_stoichiometric_air_fuel_ratio_basis_only_affects_input(mech):
    mole = mech.stoichiometric_air_fuel_ratio(
        {"CH4": 1.0}, mech.air(basis="mole"), basis="mole"
    )
    mass = mech.stoichiometric_air_fuel_ratio(
        {"CH4": 1.0}, mech.air(basis="mass"), basis="mass"
    )
    assert mole == pytest.approx(mass, rel=1e-6)


def test_air_dry_mass_matches_real_composition(mech):
    mass = mech.air()
    assert mass["O2"] == pytest.approx(0.2329, abs=1e-3)
    assert mass["N2"] == pytest.approx(0.7671, abs=1e-3)


def test_air_dry_mole_is_the_canonical_definition(mech):
    assert mech.air(basis="mole") == {"O2": 0.21, "N2": 0.79}


def test_air_relative_humidity_requires_t_and_p(mech):
    with pytest.raises(ValueError, match="requires T and P"):
        mech.air(relative_humidity=0.5)


def test_air_rejects_both_humidity_kwargs(mech):
    with pytest.raises(ValueError, match="at most one"):
        mech.air(relative_humidity=0.5, absolute_humidity=0.01)


def test_air_absolute_humidity_mass_fraction(mech):
    mass = mech.air(absolute_humidity=0.01)
    assert mass["H2O"] == pytest.approx(0.01 / 1.01)


def test_air_relative_humidity_uses_real_saturation_pressure(mech):
    mole = mech.air(relative_humidity=0.6, T=298.15, P=101325, basis="mole")
    water = ct.Water()
    water.TP = 298.15, ct.one_atm
    expected_x_h2o = 0.6 * water.P_sat / 101325
    assert mole["H2O"] == pytest.approx(expected_x_h2o)


def test_air_absolute_humidity_accepts_decimal(mech):
    # Decimal from an OpenFOAM dict mixed with Cantera floats in air()
    # raises TypeError unless converted.
    mass = mech.air(absolute_humidity=Decimal("0.01"))
    assert mass["H2O"] == pytest.approx(0.01 / 1.01)


def test_air_relative_humidity_accepts_decimal(mech):
    mole = mech.air(
        relative_humidity=Decimal("0.6"), T=Decimal("298.15"),
        P=Decimal("101325"), basis="mole",
    )
    water = ct.Water()
    water.TP = 298.15, ct.one_atm
    expected_x_h2o = 0.6 * water.P_sat / 101325
    assert mole["H2O"] == pytest.approx(expected_x_h2o)


def test_viscosity_density_thermal_conductivity_cp_match_cantera_directly(
    mech,
):
    # Oracle: an independent Solution restricted to the same species.
    composition = {"CH4": 0.8, "N2": 0.2}
    p, T = 3.5e5, 450.0

    oracle = mech.restricted(["CH4", "N2"])
    oracle.Y = composition
    oracle.TP = T, p

    assert mech.viscosity(p, T, composition) == pytest.approx(
        oracle.viscosity
    )
    assert mech.density(p, T, composition) == pytest.approx(oracle.density)
    assert mech.thermal_conductivity(p, T, composition) == pytest.approx(
        oracle.thermal_conductivity
    )
    assert mech.cp(p, T, composition) == pytest.approx(oracle.cp_mass)


def test_properties_matches_the_individual_methods(mech):
    composition = {"CH4": 0.8, "N2": 0.2}
    p, T = 3.5e5, 450.0

    combined = mech.properties(p, T, composition)

    assert combined == {
        "density": mech.density(p, T, composition),
        "viscosity": mech.viscosity(p, T, composition),
        "thermal_conductivity": mech.thermal_conductivity(p, T, composition),
        "cp": mech.cp(p, T, composition),
    }


def test_viscosity_accepts_mole_basis_composition(mech):
    p, T = 101325.0, 300.0
    mole = {"CH4": 1.0, "N2": 1.0}
    mass = mech.mass_fractions(mole, basis="mole")

    assert mech.viscosity(p, T, mole, basis="mole") == pytest.approx(
        mech.viscosity(p, T, mass, basis="mass")
    )


def test_viscosity_resolves_species_names_case_insensitively(
    lowercase_mech,
):
    p, T = 101325.0, 300.0
    lower = lowercase_mech.viscosity(p, T, {"ch4": 1.0, "n2": 1.0})
    mixed = lowercase_mech.viscosity(p, T, {"Ch4": 1.0, "N2": 1.0})
    assert lower == pytest.approx(mixed)


def test_mass_fractions_converts_mole_to_mass(mech):
    result = mech.mass_fractions({"CH4": 1, "N2": 1}, basis="mole")
    ch4_mw = mech.molecular_weight({"CH4": 1.0}, basis="mole")
    n2_mw = mech.molecular_weight({"N2": 1.0}, basis="mole")
    assert result["CH4"] == pytest.approx(
        ch4_mw / (ch4_mw + n2_mw), rel=1e-3,
    )
    assert sum(result.values()) == pytest.approx(1.0)


def test_mass_fractions_is_identity_under_mass_basis(mech):
    result = mech.mass_fractions({"CH4": 0.3, "N2": 0.7}, basis="mass")
    assert result["CH4"] == pytest.approx(0.3)
    assert result["N2"] == pytest.approx(0.7)


def test_mole_fractions_converts_mass_to_mole(mech):
    mole = mech.mole_fractions({"CH4": 1, "N2": 1}, basis="mole")
    assert mole["CH4"] == pytest.approx(0.5)
    assert mole["N2"] == pytest.approx(0.5)


def test_complete_combustion_products_is_only_co2_h2o_n2(mech):
    products = mech.complete_combustion_products(
        {"CH4": 1.0}, mech.air(basis="mole"), basis="mole"
    )
    # No leftover fuel/oxidizer or dissociated species, above solver
    # noise (~1e-7).
    significant = {name for name, y in products.items() if y > 1e-6}
    assert significant == {"CO2", "H2O", "N2"}


def _write_mechanism(path, species_yaml, elements=("H",)):
    path.write_text(f"""
phases:
- name: gas
  thermo: ideal-gas
  elements: [{", ".join(elements)}]
  species: [{", ".join(species_yaml)}]
  transport: mixture-averaged
  state: {{T: 300, P: 1 atm}}
species:
{chr(10).join(species_yaml.values())}
""")


def _specie_yaml(
    name, low, high, tlow=300.0, tmid=1000.0, thigh=5000.0,
    composition=None, geometry="atom",
):
    composition = composition if composition is not None else {"H": 1}
    composition_yaml = ", ".join(
        f"{el}: {count}" for el, count in composition.items()
    )
    return f"""- name: {name}
  composition: {{{composition_yaml}}}
  thermo:
    model: NASA7
    temperature-ranges: [{tlow}, {tmid}, {thigh}]
    data:
    - {low}
    - {high}
  transport: {{model: gas, geometry: {geometry}, well-depth: 145.0, \
diameter: 2.05}}
"""


def test_combustion_sinks_raises_for_missing_element(tmp_path):
    # Sulfur without SO2 in the mechanism: no sink available.
    coeffs = [3.5, 0.001, 0, 0, 0, -1000, 4.0]
    p = tmp_path / "mech.yaml"
    _write_mechanism(p, {
        "H2S": _specie_yaml(
            "H2S", coeffs, coeffs, composition={"H": 2, "S": 1},
            geometry="linear",
        ),
        "H2O": _specie_yaml(
            "H2O", coeffs, coeffs, composition={"H": 2, "O": 1},
            geometry="linear",
        ),
    }, elements=["H", "O", "S"])
    mech = Mechanism(str(p))
    with pytest.raises(ValueError, match="SO2"):
        mech.complete_combustion_products({"H2S": 1.0}, {"H2O": 1.0})


def _lowercase_mechanism(path):
    # Real CHEMKIN files may spell species lowercase. Flat fake NASA7
    # coefficients: these tests are about names, not thermo.
    coeffs = [3.5, 0.001, 0, 0, 0, -1000, 4.0]
    species = {
        "ch4": _specie_yaml(
            "ch4", coeffs, coeffs, composition={"C": 1, "H": 4},
            geometry="nonlinear",
        ),
        "o2": _specie_yaml(
            "o2", coeffs, coeffs, composition={"O": 2},
            geometry="linear",
        ),
        "n2": _specie_yaml(
            "n2", coeffs, coeffs, composition={"N": 2},
            geometry="linear",
        ),
        "co2": _specie_yaml(
            "co2", coeffs, coeffs, composition={"C": 1, "O": 2},
            geometry="linear",
        ),
        "h2o": _specie_yaml(
            "h2o", coeffs, coeffs, composition={"H": 2, "O": 1},
            geometry="nonlinear",
        ),
        "nh3": _specie_yaml(
            "nh3", coeffs, coeffs, composition={"N": 1, "H": 3},
            geometry="nonlinear",
        ),
    }
    _write_mechanism(path, species, elements=["C", "H", "N", "O"])
    return Mechanism(str(path))


@pytest.fixture
def lowercase_mech(tmp_path):
    return _lowercase_mechanism(tmp_path / "lowercase.yaml")


def test_find_species_resolves_case_insensitively(lowercase_mech):
    assert lowercase_mech.find_species("n2") == "n2"
    assert lowercase_mech.find_species("N2") == "n2"
    assert lowercase_mech.find_species("nH3") == "nh3"
    assert lowercase_mech.find_species("H2o") == "h2o"
    assert lowercase_mech.find_species("NH3") == "nh3"


def test_find_species_raises_for_unknown_species(lowercase_mech):
    with pytest.raises(ValueError, match="no species named"):
        lowercase_mech.find_species("xe")


def test_air_returns_the_mechanisms_own_casing(lowercase_mech):
    assert lowercase_mech.air(basis="mole") == {"o2": 0.21, "n2": 0.79}


def test_molecular_weight_accepts_mixed_case_composition(lowercase_mech):
    lower = lowercase_mech.molecular_weight({"ch4": 1.0}, basis="mole")
    mixed = lowercase_mech.molecular_weight({"Ch4": 1.0}, basis="mole")
    assert lower == pytest.approx(mixed)


def test_mass_fractions_accepts_mixed_case_composition(lowercase_mech):
    result = lowercase_mech.mass_fractions(
        {"N2": 1, "nH3": 1}, basis="mole",
    )
    assert set(result) == {"n2", "nh3"}


def test_stoichiometric_air_fuel_ratio_accepts_mixed_case(lowercase_mech):
    ratio = lowercase_mech.stoichiometric_air_fuel_ratio(
        {"CH4": 1.0}, {"O2": 1.0, "N2": 3.76}, basis="mole",
    )
    assert ratio > 0


def test_complete_combustion_products_resolves_mixed_case(lowercase_mech):
    # The sink table is uppercase; the lookup must ignore case.
    products = lowercase_mech.complete_combustion_products(
        {"CH4": 1.0}, {"o2": 1.0, "N2": 3.76}, basis="mole",
        equivalence_ratio=1.0,
    )
    assert sum(products.values()) == pytest.approx(1.0)
    assert set(products) <= {"ch4", "o2", "n2", "co2", "h2o"}


def test_elements_reports_integer_atom_counts(mech):
    assert mech.elements("CH4") == {"C": 1, "H": 4}


def test_elements_raises_for_non_integer_count(tmp_path):
    # Cantera accepts fractional element counts; elements() must not
    # truncate them.
    p = tmp_path / "mech.yaml"
    _write_mechanism(p, {"FRAC": _specie_yaml(
        "FRAC", [3.5, 0.001, 0, 0, 0, -1000, 4.0],
        [3.5, 0.001, 0, 0, 0, -1000, 4.0],
        composition={"H": 3.5}, geometry="nonlinear",
    )})
    mech = Mechanism(str(p))
    with pytest.raises(ValueError, match="non-integer"):
        mech.elements("FRAC")


def test_from_chemkin_loads_a_converted_mechanism(h2):
    m = Mechanism.from_chemkin(
        str(h2.mechanism), thermo_file=str(h2.thermo),
    )
    assert m.gas.species_names == [
        "H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR", "N2",
    ]


def test_from_chemkin_cleans_up_the_temp_yaml_by_default(h2):
    m = Mechanism.from_chemkin(
        str(h2.mechanism), thermo_file=str(h2.thermo),
    )
    assert not Path(m.mechanism_path).exists()


def test_from_chemkin_keeps_the_yaml_only_when_out_path_given(h2, tmp_path):
    out_path = tmp_path / "h2.yaml"
    Mechanism.from_chemkin(
        str(h2.mechanism), thermo_file=str(h2.thermo),
        out_path=str(out_path),
    )
    assert out_path.is_file()


def test_from_chemkin_accepts_reference_pressure(h2):
    m = Mechanism.from_chemkin(
        str(h2.mechanism), thermo_file=str(h2.thermo),
        reference_pressure=2e5,
    )
    assert m.reference_pressure == 2e5


def test_from_chemkin_rejects_a_missing_mechanism_file(h2):
    with pytest.raises(
        ValueError, match="chemkin mechanism_file not found",
    ):
        Mechanism.from_chemkin(str(h2.missing))


def test_from_chemkin_rejects_a_missing_thermo_file(h2):
    with pytest.raises(ValueError, match="chemkin thermo_file not found"):
        Mechanism.from_chemkin(
            str(h2.mechanism), thermo_file=str(h2.missing),
        )


def test_from_chemkin_rejects_a_missing_transport_file(h2):
    with pytest.raises(
        ValueError, match="chemkin transport_file not found",
    ):
        Mechanism.from_chemkin(
            str(h2.mechanism), thermo_file=str(h2.thermo),
            transport_file=str(h2.missing),
        )


def test_from_chemkin_propagates_ck2yamls_own_input_error(h2):
    # ck2yaml matches transport data case-sensitively, so every species
    # of the lowercased transport file is reported missing.
    with pytest.raises(ck2yaml.InputError, match="No transport data"):
        Mechanism.from_chemkin(
            str(h2.mechanism), thermo_file=str(h2.thermo),
            transport_file=str(h2.transport),
        )


def test_from_chemkin_omits_permissive_hint_when_it_would_not_help(h2):
    # permissive=True does not fix a case mismatch; the hint must not
    # suggest it.
    with pytest.raises(ck2yaml.InputError) as exc_info:
        Mechanism.from_chemkin(
            str(h2.mechanism), thermo_file=str(h2.thermo),
            transport_file=str(h2.transport),
        )
    assert "permissive" not in str(exc_info.value).lower()


def test_from_chemkin_adds_permissive_hint_when_it_would_help(h2, tmp_path):
    redundant = _redundant_species_chemkin(h2, tmp_path)
    with pytest.raises(ck2yaml.InputError, match="permissive=True"):
        Mechanism.from_chemkin(str(redundant), thermo_file=str(h2.thermo))


def test_from_chemkin_permissive_true_resolves_the_redundant_declaration(
    h2, tmp_path,
):
    redundant = _redundant_species_chemkin(h2, tmp_path)
    m = Mechanism.from_chemkin(
        str(redundant), thermo_file=str(h2.thermo), permissive=True,
    )
    assert "AR" in m.gas.species_names


def test_from_chemkin_duplicate_reaction_raises_cantera_error(h2, tmp_path):
    # Caught only when loaded, so a CanteraError, and permissive=True
    # (ck2yaml only) has no effect.
    duplicate = _duplicate_reaction_chemkin(h2, tmp_path)
    with pytest.raises(ct.CanteraError, match="duplicate") as exc_info:
        Mechanism.from_chemkin(str(duplicate), thermo_file=str(h2.thermo))
    assert "permissive" not in str(exc_info.value).lower()


def test_from_chemkin_duplicate_reaction_does_not_leave_a_dangling_temp_file(
    h2, tmp_path,
):
    duplicate = _duplicate_reaction_chemkin(h2, tmp_path)
    with pytest.raises(ct.CanteraError):
        Mechanism.from_chemkin(str(duplicate), thermo_file=str(h2.thermo))
    assert list(tmp_path.glob("*.yaml")) == []


@pytest.mark.slow
def test_from_chemkin_reports_c3mechs_nonstandard_troe_syntax_clearly():
    # C3Mech uses LOWSP/TROESP, which ck2yaml does not support even
    # with permissive=True; the error must name reaction and line.
    paths = _download_or_skip("c3mech")
    with pytest.raises(ck2yaml.InputError, match="could not convert"):
        Mechanism.from_chemkin(
            str(paths["mechanism"]),
            thermo_file=str(paths["thermo"]),
            transport_file=str(paths["transport"]),
        )


@pytest.mark.slow
def test_from_chemkin_converts_c3mechs_portable_variant_matching_oracle():
    # C3Mech's portable variant (LOWSP/TROESP expanded) converts; the
    # counts are checked against C3Mech's own YAML (5044 species, 22161
    # reactions), which is only counted, not loaded (~5 min each).
    paths = _download_or_skip("c3mech_cantera")
    m = Mechanism.from_chemkin(
        str(paths["mechanism"]),
        thermo_file=str(paths["thermo"]),
        transport_file=str(paths["transport"]),
    )
    with paths["oracle"].open() as oracle_file:
        oracle = ck2yaml.yaml.YAML(typ="safe").load(oracle_file)
    assert m.gas.n_species == len(oracle["species"])
    assert m.gas.n_reactions == len(oracle["reactions"])


@pytest.mark.slow
@pytest.mark.parametrize("name", ["syngas_nox", "methanol"])
def test_from_chemkin_converts_other_real_nui_galway_mechanisms(name):
    # Other file-name conventions (.MECH/.THERM/.TRAN vs .inp/
    # _therm.dat/_tran.dat) and sizes (44/173 species).
    paths = _download_or_skip(name)
    m = Mechanism.from_chemkin(
        str(paths["mechanism"]),
        thermo_file=str(paths["thermo"]),
        transport_file=str(paths["transport"]),
    )
    assert len(m.gas.species_names) > 0


_CH4 = {"CH4": 1.0}
_AIR = {"O2": 0.233, "N2": 0.767}


@pytest.fixture(scope="module")
def gri():
    return Mechanism("gri30.yaml")


def test_mixture_fraction_is_stoichiometric_at_unity(gri):
    afr_st = gri.stoichiometric_air_fuel_ratio(_CH4, _AIR)
    Z_st = 1 / (afr_st + 1)
    assert gri.mixture_fraction(
        _CH4, _AIR, equivalence_ratio=1,
    ) == pytest.approx(Z_st)
    assert gri.mixture_fraction(
        _CH4, _AIR, air_fuel_equivalence_ratio=1,
    ) == pytest.approx(Z_st)


def test_mixture_fraction_lambda_is_inverse_equivalence_ratio(gri):
    assert gri.mixture_fraction(
        _CH4, _AIR, air_fuel_equivalence_ratio=Decimal("1.6"),
    ) == pytest.approx(
        gri.mixture_fraction(_CH4, _AIR, equivalence_ratio=1/1.6)
    )


@pytest.mark.parametrize("ratios", [
    {}, {"equivalence_ratio": 1, "air_fuel_equivalence_ratio": 1},
])
def test_mixture_fraction_needs_exactly_one_ratio(gri, ratios):
    with pytest.raises(ValueError, match="exactly one"):
        gri.mixture_fraction(_CH4, _AIR, **ratios)


def test_residual_fuel_fraction_is_zero_unless_rich(gri):
    Z_st = gri.mixture_fraction(_CH4, _AIR, equivalence_ratio=1)
    assert gri.residual_fuel_fraction(Z_st/2, _CH4, _AIR) == 0
    assert gri.residual_fuel_fraction(Z_st, _CH4, _AIR) == (
        pytest.approx(0, abs=1e-12)
    )
    assert gri.residual_fuel_fraction(1, _CH4, _AIR) == 1


def test_partially_burnt_ends_are_reactants_and_products(gri):
    Z = gri.mixture_fraction(_CH4, _AIR, equivalence_ratio=0.8)
    unburnt = gri.partially_burnt(_CH4, _AIR, Z, 0)
    assert unburnt["CH4"] == pytest.approx(Z)
    assert set(unburnt) == {"CH4", "O2", "N2"}
    burnt = gri.partially_burnt(_CH4, _AIR, Z, 1)
    products = gri.complete_combustion_products(
        _CH4, _AIR, mixture_fraction=Z,
    )
    assert burnt == pytest.approx(products, abs=1e-8)


def test_partially_burnt_blends_linearly_and_sums_to_one(gri):
    Z = gri.mixture_fraction(_CH4, _AIR, equivalence_ratio=0.8)
    half = gri.partially_burnt(_CH4, _AIR, Z, 0.5)
    assert sum(half.values()) == pytest.approx(1)
    assert half["CH4"] == pytest.approx(Z/2, rel=1e-6)
    assert half["CO2"] == pytest.approx(
        gri.partially_burnt(_CH4, _AIR, Z, 1)["CO2"]/2, rel=1e-6,
    )
