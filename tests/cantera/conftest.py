from types import SimpleNamespace

import pytest

_H2_CTI = """\
units(length='cm', time='s', quantity='mol', act_energy='cal/mol')
ideal_gas(name='gas', elements='H O', species='gri30: H2 O2 H2O OH H O',
          reactions='all', transport='Mix')
reaction('H2 + O2 <=> 2 OH', [1.7e13, 0.0, 47780.0])
"""


@pytest.fixture(scope="module")
def h2(tmp_path_factory):
    """Cantera's h2o2.yaml as CHEMKIN files; `transport` spells species
    lowercase, which ck2yaml rejects, `valid_transport` does not."""
    ct = pytest.importorskip("cantera")
    from cantera import yaml2ck

    out = tmp_path_factory.mktemp("h2_chemkin")
    paths = SimpleNamespace(
        mechanism=out / "mech.ck",
        thermo=out / "thermo.ck",
        transport=out / "tran.ck",
        valid_transport=out / "tran_valid.ck",
        missing=out / "nope.dat",
    )
    yaml2ck.convert(
        ct.Solution("h2o2.yaml"), mechanism_path=paths.mechanism,
        thermo_path=paths.thermo, transport_path=paths.valid_transport,
    )
    lines = paths.valid_transport.read_text().splitlines()
    paths.transport.write_text("\n".join(
        line if line.startswith("!") else line.lower() for line in lines
    ) + "\n")
    return paths


@pytest.fixture
def h2_cti(tmp_path):
    """A one-reaction H2/O2 CTI mechanism (Cantera 3 ships none)."""
    path = tmp_path / "h2.cti"
    path.write_text(_H2_CTI)
    return path
