from pathlib import Path

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
    mine = critical.Critical(400, 1e7, 0.07, 0.25)
    data = critical.lookup(mech, {"H2O": mine})
    assert data["H2O"] is mine
    assert data["H2O"].source == critical.Source.GIVEN


def test_given_data_for_unknown_specie_raise(mech):
    with pytest.raises(ValueError, match="not in the mechanism: FOO"):
        critical.lookup(mech, {"FOO": critical.Critical(1, 1, 1, 0)})


def test_lump_of_one_specie_is_that_specie():
    c = critical.OPENFOAM["N2"]
    lumped = critical.lump({"N2": c}, {"N2": 1})
    assert (lumped.Tc, lumped.Pc, lumped.Vc, lumped.omega) == (
        pytest.approx((c.Tc, c.Pc, c.Vc, c.omega))
    )


def test_lump_is_kays_rule():
    a, b = critical.OPENFOAM["N2"], critical.OPENFOAM["H2O"]
    lumped = critical.lump({"A": a, "B": b}, {"A": 3, "B": 1})
    Zc = 0.75*a.Pc*a.Vc/a.Tc + 0.25*b.Pc*b.Vc/b.Tc
    assert lumped.Tc == pytest.approx(0.75*a.Tc + 0.25*b.Tc)
    assert lumped.Vc == pytest.approx(0.75*a.Vc + 0.25*b.Vc)
    assert lumped.omega == pytest.approx(0.75*a.omega + 0.25*b.omega)
    assert lumped.Pc == pytest.approx(Zc*lumped.Tc/lumped.Vc)


def test_to_dict_is_the_equation_of_state_entry():
    assert critical.Critical(1.0, 2.0, 3.0, 4.0).to_dict() == {
        "Tc": 1.0, "Vc": 3.0, "Pc": 2.0, "omega": 4.0,
    }


def test_summary_counts_per_source(mech):
    text = critical.summary(critical.lookup(mech))
    assert "critical data from OpenFOAM liquidProperties: " in text
    assert "given values" not in text
