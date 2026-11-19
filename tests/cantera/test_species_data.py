import subprocess
import warnings
from decimal import Decimal
from pathlib import Path

import pytest

ct = pytest.importorskip("cantera")
np = pytest.importorskip("numpy")

from pyfoamy import Case  # noqa: E402
from pyfoamy.cantera.mechanism import Mechanism  # noqa: E402
from pyfoamy.cantera.species_data import ThermoDatabase  # noqa: E402

_MECHANISM = Path(__file__).resolve().parent.parent.parent / "converted.yml"


def test_missing_cantera_raises_import_error(monkeypatch):
    import pyfoamy.cantera.mechanism as mechanism_module
    monkeypatch.setattr(mechanism_module, "ct", None)
    monkeypatch.setattr(
        mechanism_module, "_CANTERA_IMPORT_ERROR", ImportError("boom"),
    )
    with pytest.raises(ImportError, match="pyfoamy.cantera requires cantera"):
        ThermoDatabase("converted.yml")


def _skip_unless_mechanism():
    if not _MECHANISM.is_file():
        pytest.skip("converted.yml not present in the repo root")


@pytest.fixture
def db():
    _skip_unless_mechanism()
    return ThermoDatabase(str(_MECHANISM))


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


def _assert_continuous_per_cantera(tmp_path, thermo):
    """Round-trips through Cantera; no NasaPoly2::validate warnings."""
    p = tmp_path / "check.yaml"
    low = [float(x) for x in thermo["lowCpCoeffs"]]
    high = [float(x) for x in thermo["highCpCoeffs"]]
    entries = {"CHECK": _specie_yaml(
        "CHECK", low, high, thermo["Tlow"], thermo["Tcommon"],
        thermo["Thigh"],
    )}
    _write_mechanism(p, entries)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ct.Solution(str(p))
    assert not caught, [str(w.message) for w in caught]


def test_real_specie_passthrough_matches_cantera(db):
    # Cp coefficients and a5 pass through verbatim; a6 (entropy) does
    # not, see test_real_specie_passthrough_entropy_is_pstd_referenced.
    n2 = db.species(["N2"])["N2"]
    coeffs = db.mechanism.gas.species("N2").thermo.coeffs
    assert n2["thermodynamics"]["Tcommon"] == coeffs[0]
    assert n2["thermodynamics"]["highCpCoeffs"][:6] == list(coeffs[1:7])
    assert n2["thermodynamics"]["lowCpCoeffs"][:6] == list(coeffs[8:14])
    assert n2["specie"]["molWeight"] > 0
    assert n2["transport"]["mu"] > 0


def test_real_specie_passthrough_entropy_is_pstd_referenced(db):
    # Entropy must refer to OpenFOAM's pStd=1e5 (janafThermo::s()),
    # not the mechanism's reference_pressure; oracle: Solution at 1e5.
    name = "N2"
    n2 = db.species([name])[name]["thermodynamics"]
    T = 298.15
    a = n2["lowCpCoeffs"] if T <= n2["Tcommon"] else n2["highCpCoeffs"]
    s_R = (
        a[0]*np.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3
        + a[4]*T**4/4 + a[6]
    )
    fitted_s = s_R * ct.gas_constant

    oracle_gas = ct.Solution(str(_MECHANISM))
    oracle_gas.TPX = T, 1e5, {name: 1.0}
    true_s = oracle_gas.entropy_mole

    assert fitted_s == pytest.approx(true_s, rel=1e-9)


def test_no_split_specie_has_tcommon_equal_thigh(db):
    ar = db.species(["AR"])["AR"]
    assert ar["thermodynamics"]["Tcommon"] == ar["thermodynamics"]["Thigh"]


def test_lump_is_continuous_per_cantera(db, tmp_path):
    lumped = db.lump(
        {"CH4": 0.9, "C2H6": 0.05, "C3H8": 0.02, "CO2": 0.02, "N2": 0.01}
    )
    assert lumped["thermodynamics"]["Tlow"] < \
        lumped["thermodynamics"]["Tcommon"] < \
        lumped["thermodynamics"]["Thigh"]
    _assert_continuous_per_cantera(tmp_path, lumped["thermodynamics"])


def test_lump_mass_and_mole_basis_agree(db):
    mole = db.lump({"CH4": 0.9, "N2": 0.1}, basis="mole")
    gas = db.mechanism.gas
    gas.X = {"CH4": 0.9, "N2": 0.1}
    mass = db.lump(dict(zip(["CH4", "N2"], gas["CH4", "N2"].Y)), basis="mass")
    assert mole["specie"]["molWeight"] == pytest.approx(
        mass["specie"]["molWeight"]
    )
    assert mole["thermodynamics"]["Tcommon"] == pytest.approx(
        mass["thermodynamics"]["Tcommon"], rel=1e-3
    )


def test_lump_accepts_decimal_t_low_t_high(db):
    # Decimal from an OpenFOAM dict mixed with floats in
    # _optimize_tcommon() raises TypeError unless converted.
    composition = {"CH4": 0.9, "N2": 0.1}
    with_decimal = db.lump(
        composition, T_low=Decimal("300"), T_high=Decimal("2000"),
    )
    with_float = db.lump(composition, T_low=300.0, T_high=2000.0)
    assert with_decimal["thermodynamics"]["Tcommon"] == pytest.approx(
        with_float["thermodynamics"]["Tcommon"]
    )


def test_lump_repairs_small_discontinuity(tmp_path):
    p = tmp_path / "mech.yaml"
    _write_mechanism(p, {"SMALL": _specie_yaml(
        "SMALL", [3.5, 0.001, 0, 0, 0, -1000, 4.0],
        [3.502, 0.001, 0, 0, 0, -1000, 4.0],
    )})
    db = ThermoDatabase(str(p))
    with pytest.warns(UserWarning, match="repaired"):
        result = db.species(["SMALL"])
    assert result["SMALL"]["thermodynamics"]["Tcommon"] != 1000.0


def test_lump_raises_for_large_discontinuity(tmp_path):
    p = tmp_path / "mech.yaml"
    _write_mechanism(p, {"BIG": _specie_yaml(
        "BIG", [3.5, 0.001, 0, 0, 0, -1000, 4.0],
        [3.9, 0.001, 0, 0, 0, -1000, 4.0],
    )})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Cantera's own load-time warning
        db = ThermoDatabase(str(p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="BIG"):
            db.species(["BIG"])


def test_unknown_thermo_model_raises_on_assignment(db):
    with pytest.raises(ValueError, match="hLogPolynomial"):
        db.thermo_model = "hLogPolynomial"


def test_unknown_transport_model_raises_on_assignment(db):
    with pytest.raises(ValueError, match="tabulated"):
        db.transport_model = "tabulated"


def test_constructor_defaults_and_overrides_thermo_transport_model():
    _skip_unless_mechanism()
    default = ThermoDatabase(str(_MECHANISM))
    assert default.thermo_model == "janaf"
    assert default.transport_model == "const"

    configured = ThermoDatabase(
        str(_MECHANISM), thermo_model="janaf",
        transport_model="logPolynomial",
    )
    assert configured.transport_model == "logPolynomial"
    lumped = configured.lump({"CH4": 1.0})
    assert set(lumped["transport"]) == {"muLogCoeffs<8>", "kappaLogCoeffs<8>"}


def test_setting_transport_model_changes_subsequent_lump_shape(db):
    assert db.transport_model == "const"
    assert set(db.lump({"CH4": 1.0})["transport"]) == {"mu", "Pr"}

    db.transport_model = "sutherland"
    assert set(db.lump({"CH4": 1.0})["transport"]) == {"As", "Ts"}


@pytest.fixture
def hpoly_db():
    _skip_unless_mechanism()
    return ThermoDatabase(str(_MECHANISM), thermo_model="hPolynomial")


def test_h_polynomial_lump_has_the_real_dict_shape(hpoly_db):
    lumped = hpoly_db.lump(
        {"CH4": 0.9, "C2H6": 0.05, "C3H8": 0.02, "CO2": 0.02, "N2": 0.01}
    )
    thermodynamics = lumped["thermodynamics"]
    assert set(thermodynamics) == {
        "hf", "sf", "Tlow", "Thigh", "CpCoeffs<8>",
    }
    assert len(thermodynamics["CpCoeffs<8>"]) == 8
    assert thermodynamics["Tlow"] < thermodynamics["Thigh"]


def test_h_polynomial_cp_fit_matches_cantera(hpoly_db):
    n2 = hpoly_db.species(["N2"])["N2"]
    coeffs = n2["thermodynamics"]["CpCoeffs<8>"]
    gas = hpoly_db.mechanism.gas
    for T in (400, 1000, 3000):
        gas.TPX = T, ct.one_atm, {"N2": 1.0}
        real = gas.cp_mass
        pred = sum(c * T**p for p, c in enumerate(coeffs))
        assert pred == pytest.approx(real, rel=0.02)


def test_h_polynomial_sf_matches_real_absolute_entropy(hpoly_db):
    # N2: ~191.6 J/mol/K -> ~6840 J/kg/K (absolute, not formation).
    n2 = hpoly_db.species(["N2"])["N2"]
    assert n2["thermodynamics"]["sf"] == pytest.approx(6840, rel=0.01)
    assert n2["thermodynamics"]["hf"] == pytest.approx(0, abs=1e3)


def test_species_skips_janaf_passthrough_for_other_thermo_models(hpoly_db):
    # Only janaf converts directly from Cantera; others are fitted.
    n2 = hpoly_db.species(["N2"])["N2"]
    assert "CpCoeffs<8>" in n2["thermodynamics"]
    assert "highCpCoeffs" not in n2["thermodynamics"]


@pytest.mark.openfoam
def test_h_polynomial_dict_round_trips_through_foam_dictionary(
    hpoly_db, tmp_path
):
    lumped = hpoly_db.lump({"CH4": 1.0})
    case = Case(tmp_path / "myCase")
    case.write_dict(
        "constant/testThermo", {"thermodynamics": lumped["thermodynamics"]}
    )
    path = tmp_path / "myCase" / "constant" / "testThermo"
    for entry, expected in (
        ("thermodynamics/hf", lumped["thermodynamics"]["hf"]),
        ("thermodynamics/sf", lumped["thermodynamics"]["sf"]),
    ):
        result = subprocess.run(
            ["foamDictionary", "-entry", entry, "-value", str(path)],
            capture_output=True, text=True, check=True,
        )
        assert float(result.stdout.strip()) == pytest.approx(expected)
    result = subprocess.run(
        [
            "foamDictionary", "-entry", "thermodynamics/CpCoeffs<8>",
            "-value", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    coeffs = result.stdout.strip().strip("()").split()
    assert len(coeffs) == 8
    # foamDictionary prints with limited precision.
    assert [float(c) for c in coeffs] == pytest.approx(
        lumped["thermodynamics"]["CpCoeffs<8>"], rel=1e-4
    )


def test_h_polynomial_thermo_type_entries():
    _skip_unless_mechanism()
    db = ThermoDatabase(str(_MECHANISM), thermo_model="hPolynomial")
    assert db.thermo_type_entries == {
        "thermo": "hPolynomial", "transport": "const",
    }


def test_thermo_type_entries_matches_configured_models():
    _skip_unless_mechanism()
    db = ThermoDatabase(
        str(_MECHANISM), thermo_model="janaf",
        transport_model="logPolynomial",
    )
    assert db.thermo_type_entries == {
        "thermo": "janaf", "transport": "logPolynomial",
    }


def test_lump_accepts_mixed_case_composition(tmp_path):
    # Mechanisms differ in name capitalization (find_species()).
    coeffs = [3.5, 0.001, 0, 0, 0, -1000, 4.0]
    p = tmp_path / "lowercase.yaml"
    _write_mechanism(p, {
        "ch4": _specie_yaml(
            "ch4", coeffs, coeffs, composition={"C": 1, "H": 4},
            geometry="nonlinear",
        ),
        "n2": _specie_yaml(
            "n2", coeffs, coeffs, composition={"N": 2}, geometry="linear",
        ),
    }, elements=["C", "H", "N"])
    db = ThermoDatabase(str(p))
    lumped = db.lump({"CH4": 0.9, "N2": 0.1}, basis="mole")
    assert lumped["specie"]["molWeight"] > 0


def test_complete_combustion_products_can_be_lumped(db):
    products = db.mechanism.complete_combustion_products(
        {"CH4": 1.0}, db.mechanism.air(basis="mole"), basis="mole"
    )
    lumped = db.lump(products, basis="mass")
    assert set(lumped["thermodynamics"]) == {
        "Tlow", "Thigh", "Tcommon", "highCpCoeffs", "lowCpCoeffs",
    }
    assert lumped["specie"]["molWeight"] == pytest.approx(27.6, abs=0.1)


def test_thermo_database_shares_a_given_mechanism(db):
    mech = Mechanism(str(_MECHANISM))
    shared = ThermoDatabase(mech)
    assert shared.mechanism is mech


def test_lump_group_shares_one_tcommon(db, tmp_path):
    # CH4 and N2 alone optimize to very different Tcommon;
    # lump_group() must fit both with a shared one.
    independent = {
        name: db.lump({name: 1.0}, basis="mole")
        for name in ("CH4", "N2")
    }
    assert independent["CH4"]["thermodynamics"]["Tcommon"] != pytest.approx(
        independent["N2"]["thermodynamics"]["Tcommon"]
    )

    grouped = db.lump_group({
        "fuel": ({"CH4": 1.0}, "mole"), "oxidant": ({"N2": 1.0}, "mole"),
    })
    assert grouped["fuel"]["thermodynamics"]["Tcommon"] == pytest.approx(
        grouped["oxidant"]["thermodynamics"]["Tcommon"]
    )
    for name in ("fuel", "oxidant"):
        _assert_continuous_per_cantera(tmp_path, grouped[name]["thermodynamics"])


def test_lump_group_shared_tcommon_is_within_both_ranges(db):
    grouped = db.lump_group({
        "fuel": ({"CH4": 1.0}, "mole"), "oxidant": ({"N2": 1.0}, "mole"),
    })
    for entry in grouped.values():
        thermo = entry["thermodynamics"]
        assert thermo["Tlow"] < thermo["Tcommon"] < thermo["Thigh"]


def test_lump_group_result_shape_matches_lump(db):
    grouped = db.lump_group({"fuel": ({"CH4": 1.0}, "mole")})
    single = db.lump({"CH4": 1.0}, basis="mole")
    assert set(grouped["fuel"]) == set(single)
    assert set(grouped["fuel"]["thermodynamics"]) == set(
        single["thermodynamics"]
    )


def test_lump_group_non_janaf_lumps_each_independently(hpoly_db):
    # hPolynomial has no Tcommon: lump_group() delegates to lump().
    real_species = {"fuel": "CH4", "oxidant": "N2"}
    grouped = hpoly_db.lump_group({
        "fuel": ({"CH4": 1.0}, "mole"), "oxidant": ({"N2": 1.0}, "mole"),
    })
    for name, specie in real_species.items():
        single = hpoly_db.lump({specie: 1.0}, basis="mole")
        assert grouped[name]["thermodynamics"] == single["thermodynamics"]


def test_lump_group_raises_for_incompatible_t_range(db):
    with pytest.raises(ValueError, match="T_low"):
        db.lump_group(
            {
                "fuel": ({"CH4": 1.0}, "mole"),
                "oxidant": ({"N2": 1.0}, "mole"),
            },
            T_low=5000, T_high=1000,
        )


_INERT_COEFFS = [3.5, 0, 0, 0, 0, 0, 4.0]


def test_lump_group_repairs_small_discontinuity(tmp_path):
    p = tmp_path / "small.yaml"
    _write_mechanism(p, {
        "SMALL": _specie_yaml(
            "SMALL", [3.5, 0.001, 0, 0, 0, -1000, 4.0],
            [3.502, 0.001, 0, 0, 0, -1000, 4.0],
        ),
        "N2": _specie_yaml(
            "N2", _INERT_COEFFS, _INERT_COEFFS, composition={"N": 2},
            geometry="linear",
        ),
    }, elements=("H", "N"))
    db = ThermoDatabase(str(p))
    with pytest.warns(UserWarning, match="repaired"):
        db.lump_group({
            "fuel": ({"SMALL": 1.0}, "mole"), "oxidant": ({"N2": 1.0}, "mole"),
        })


def test_lump_group_raises_for_large_discontinuity(tmp_path):
    p = tmp_path / "big.yaml"
    _write_mechanism(p, {
        "BIG": _specie_yaml(
            "BIG", [3.5, 0.001, 0, 0, 0, -1000, 4.0],
            [3.9, 0.001, 0, 0, 0, -1000, 4.0],
        ),
        "N2": _specie_yaml(
            "N2", _INERT_COEFFS, _INERT_COEFFS, composition={"N": 2},
            geometry="linear",
        ),
    }, elements=("H", "N"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Cantera's own load-time warning
        db = ThermoDatabase(str(p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="BIG"):
            db.lump_group({
                "fuel": ({"BIG": 1.0}, "mole"), "oxidant": ({"N2": 1.0}, "mole"),
            })


def test_species_group_shares_one_tcommon(db, tmp_path):
    independent = db.species(["CH4", "N2"])
    assert independent["CH4"]["thermodynamics"]["Tcommon"] != pytest.approx(
        independent["N2"]["thermodynamics"]["Tcommon"]
    )

    grouped = db.species_group(["CH4", "N2"])
    assert grouped["CH4"]["thermodynamics"]["Tcommon"] == pytest.approx(
        grouped["N2"]["thermodynamics"]["Tcommon"]
    )
    for name in ("CH4", "N2"):
        _assert_continuous_per_cantera(tmp_path, grouped[name]["thermodynamics"])


def test_species_group_includes_elements_like_species(db):
    grouped = db.species_group(["CH4", "N2"])
    assert grouped["CH4"]["elements"] == db.mechanism.elements("CH4")
    assert grouped["N2"]["elements"] == db.mechanism.elements("N2")


def test_species_group_defaults_to_every_specie(tmp_path):
    # Small synthetic mechanism: a joint Tcommon optimization over
    # every species of a real mechanism is slow.
    p = tmp_path / "small.yaml"
    _write_mechanism(p, {
        "A": _specie_yaml(
            "A", _INERT_COEFFS, _INERT_COEFFS, composition={"H": 1},
        ),
        "B": _specie_yaml(
            "B", _INERT_COEFFS, _INERT_COEFFS, composition={"H": 1},
        ),
    })
    small_db = ThermoDatabase(str(p))
    grouped = small_db.species_group()
    assert set(grouped) == {"A", "B"}
    tcommons = {e["thermodynamics"]["Tcommon"] for e in grouped.values()}
    assert len(tcommons) == 1


def test_species_group_resolves_case_insensitively(db):
    grouped = db.species_group(["ch4", "n2"])
    assert set(grouped) == {"CH4", "N2"}


def test_species_refits_non_nasa7_thermo():
    gri30 = ct.Solution("gri30.yaml")
    o2 = gri30.species("O2").input_data
    o2["thermo"] = ct.Solution("airNASA9.yaml").species("O2").input_data[
        "thermo"
    ]
    gas = ct.Solution(
        thermo="ideal-gas", transport_model="mixture-averaged",
        species=[ct.Species.from_dict(o2), gri30.species("N2")],
    )
    db = ThermoDatabase(Mechanism.from_yaml(gas.write_yaml()))
    thermo = db.species(["O2"])["O2"]["thermodynamics"]
    for T in (300.0, 1500.0, 6000.0):
        gas.TPX = T, 101325.0, "O2:1"
        a = (
            thermo["highCpCoeffs"] if T > thermo["Tcommon"]
            else thermo["lowCpCoeffs"]
        )
        cp_over_r = sum(a[i] * T**i for i in range(5))
        assert cp_over_r == pytest.approx(
            gas.cp_mole / ct.gas_constant, rel=0.03,
        )


def test_sutherland_fit_minimises_relative_error():
    from pyfoamy.cantera.species_data import _sutherland_fit

    gas = ct.Solution("gri30.yaml")

    def mu(T):
        gas.TPX = T, ct.one_atm, "N2:1"
        return gas.viscosity

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _sutherland_fit(mu, None, None, 300.0, 5000.0, "X")
    Ts = np.linspace(300.0, 5000.0, 50)
    real = np.array([mu(T) for T in Ts])
    pred = fit["As"] * np.sqrt(Ts) / (1 + fit["Ts"] / Ts)
    assert np.max(np.abs(pred - real) / real) < 0.08


def test_transport_fit_warnings_name_the_specie():
    db = ThermoDatabase("h2o2.yaml", transport_model="sutherland")
    with pytest.warns(UserWarning, match="^H2: sutherland transport fit"):
        db.species(["H2"])


def test_transport_fit_rejects_a_mechanism_without_transport_data(h2):
    mech = Mechanism.from_chemkin(h2.mechanism, thermo_file=h2.thermo)
    with pytest.raises(ValueError, match="has no transport data"):
        ThermoDatabase(mech).species(["H2"])


def test_combustion_group_is_the_lump_group_of_the_three_streams():
    db = ThermoDatabase("gri30.yaml")
    fuel, oxidant = {"CH4": 1.0}, {"O2": 0.233, "N2": 0.767}
    group = db.combustion_group(fuel, oxidant, T_low=300, T_high=3000)
    products = db.mechanism.complete_combustion_products(fuel, oxidant)
    assert group == db.lump_group(
        {
            "fuel": (fuel, "mass"), "oxidant": (oxidant, "mass"),
            "products": (products, "mass"),
        },
        T_low=300, T_high=3000,
    )
    assert list(group) == ["fuel", "oxidant", "products"]
