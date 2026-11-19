from pathlib import Path

import math

import pytest

ct = pytest.importorskip("cantera")
pytest.importorskip("numpy")

from pyfoamy.cantera import critical  # noqa: E402
from pyfoamy.cantera.mechanism import Mechanism  # noqa: E402


@pytest.fixture(scope="module")
def mech():
    return Mechanism("gri30.yaml")


def test_sources_in_lookup_order(mech):
    data = critical.lookup(mech)
    assert data["H2O"].source == critical.Source.OPENFOAM
    assert data["NH3"].source == critical.Source.OPENFOAM
    assert data["CO2"].source == critical.Source.CANTERA
    assert data["OH"].source == critical.Source.LENNARD_JONES
    assert data["OH"].omega == 0
    assert list(data) == mech.gas.species_names


def test_normal_isomer_prefix_finds_the_database_entry(tmp_path):
    source = next(
        Path(d)/"gri30.yaml" for d in ct.get_data_directories()
        if (Path(d)/"gri30.yaml").is_file()
    )
    path = tmp_path/"normal.yaml"
    path.write_text(source.read_text().replace("C3H8", "NC3H8"))
    data = critical.lookup(str(path))
    assert data["NC3H8"].source == critical.Source.CANTERA
    assert data["NC3H8"].Tc == pytest.approx(369.8, abs=1)


def test_given_data_come_first(mech):
    mine = critical.Critical(400, 1e7, 0.25)
    data = critical.lookup(mech, {"H2O": mine})
    assert data["H2O"] is mine
    assert data["H2O"].source == critical.Source.GIVEN


def test_given_data_for_unknown_specie_raise(mech):
    with pytest.raises(ValueError, match="not in the mechanism: FOO"):
        critical.lookup(mech, {"FOO": critical.Critical(1, 1, 0)})


def _coefficients(c):
    # A0, A1, b of pyfoamyPengRobinsonGas
    sqrt_a = math.sqrt(critical._OMEGA_A/c.Pc)*critical._RR*c.Tc
    k = critical.kappa(c.omega)
    b = critical._OMEGA_B*critical._RR*c.Tc/c.Pc
    return sqrt_a*(1 + k), sqrt_a*k/math.sqrt(c.Tc), b


def test_lump_of_one_specie_is_that_specie():
    c = critical.OPENFOAM["N2"]
    lumped = critical.lump({"N2": c}, {"N2": 1})
    assert (lumped.Tc, lumped.Pc, lumped.omega) == (
        pytest.approx((c.Tc, c.Pc, c.omega), rel=1e-12)
    )


def test_lump_is_the_one_fluid_mixture():
    data = {n: critical.OPENFOAM[n] for n in ("N2", "H2O", "CH3OH")}
    X = {"N2": 3, "H2O": 1, "CH3OH": 0.5}
    lumped = _coefficients(critical.lump(data, X))
    mixed = [
        sum(x*_coefficients(data[n])[i] for n, x in X.items())/4.5
        for i in range(3)
    ]
    assert lumped == pytest.approx(mixed, rel=1e-12)


def test_lump_of_lumps_is_the_lump_of_the_species():
    data = {n: critical.OPENFOAM[n] for n in ("N2", "H2O", "C3H8")}
    inner = critical.lump(data, {"N2": 1, "H2O": 1})
    outer = critical.lump({"A": inner, "C3H8": data["C3H8"]},
                          {"A": 2, "C3H8": 1})
    direct = critical.lump(data, {"N2": 1, "H2O": 1, "C3H8": 1})
    assert (outer.Tc, outer.Pc, outer.omega) == pytest.approx(
        (direct.Tc, direct.Pc, direct.omega), rel=1e-12
    )


def test_lump_in_the_kappa_gap_takes_omega_0491():
    # Equal Tc and Pc: the mixture's kappa is the mole-weighted kappa.
    a = critical.Critical(500, 3e6, 0.4)
    b = critical.Critical(500, 3e6, 0.6)
    ka, kb = critical.kappa(a.omega), critical.kappa(b.omega)
    x = (kb - 1.067)/(kb - ka)
    lumped = critical.lump({"A": a, "B": b}, {"A": x, "B": 1 - x})
    assert lumped.omega == 0.491
    assert (lumped.Tc, lumped.Pc) == pytest.approx((500, 3e6), rel=1e-12)


@pytest.mark.parametrize("omega", [-0.2, 0.0, 0.3, 0.491, 0.5, 0.645, 1.2])
def test_lump_recovers_omega_on_both_fits(omega):
    c = critical.Critical(500, 3e6, omega)
    assert critical.lump({"A": c}, {"A": 1}).omega == pytest.approx(
        omega, abs=1e-12
    )


def test_to_dict_is_the_equation_of_state_entry():
    assert critical.Critical(1.0, 2.0, 4.0).to_dict() == {
        "Tc": 1.0, "Pc": 2.0, "omega": 4.0,
    }


def test_summary_counts_per_source(mech):
    text = critical.summary(critical.lookup(mech))
    assert "critical data from OpenFOAM liquidProperties: " in text
    assert "given values" not in text
