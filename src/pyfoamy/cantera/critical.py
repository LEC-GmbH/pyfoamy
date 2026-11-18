"""Critical data of a mechanism's species for a cubic equation of state."""

import dataclasses
import enum
import re
from pathlib import Path
from typing import Dict, Optional

from pyfoamy.cantera import mechanism as _mechanism
from pyfoamy.cantera.mechanism import Mechanism, require_cantera

# Boltzmann constant [J/K], universal gas constant [J/kmol/K]
_KB = 1.380649e-23
_RR = 8314.46


class Source(enum.Enum):
    """Where a specie's critical data come from, in lookup order."""

    GIVEN = "given values"
    MECHANISM = "mechanism"
    OPENFOAM = "OpenFOAM liquidProperties"
    CANTERA = "Cantera critical-properties.yaml"
    LENNARD_JONES = "Lennard-Jones estimate"


@dataclasses.dataclass(frozen=True)
class Critical:
    """Critical data of one specie or lumped mixture.

    Args:
        Tc: Critical temperature [K].
        Pc: Critical pressure [Pa].
        Vc: Critical molar volume [m^3/kmol].
        omega: Acentric factor [-].
        source: Where the data come from.
    """
    Tc: float
    Pc: float
    Vc: float
    omega: float
    source: Source = Source.GIVEN

    def to_dict(self) -> dict:
        """Returns the `equationOfState` sub-dict of `PengRobinsonGas`."""
        return {
            "Tc": self.Tc, "Vc": self.Vc, "Pc": self.Pc,
            "omega": self.omega,
        }


#: OpenFOAM's liquidProperties: the species Cantera's database lacks.
OPENFOAM = {
    "H2O": Critical(647.13, 2.2055e7, 0.05595, 0.3449),
    "N2": Critical(126.10, 3.3944e6, 0.0901, 0.0403),
    "AR": Critical(150.86, 4.8981e6, 0.07459, 0.0),
    "NH3": Critical(405.65, 1.1278e7, 0.07247, 0.2520),
    "C2H6": Critical(305.32, 4.872e6, 0.14550, 0.0995),
    "C3H8": Critical(369.83, 4.248e6, 0.2, 0.1523),
    "CH3OH": Critical(512.58, 8.0959e6, 0.1178, 0.5656),
}

_DATABASE = "critical-properties.yaml"

# Mechanism spellings of Cantera database entries; a bare formula there
# is the normal isomer.
_ALIASES = {"IC8H18": "C8H18-14"}
_NORMAL = re.compile(r"^N-?(C\d.*)$")


def _cantera_database() -> dict:
    # Cantera's own YAML dependency; YAML 1.2 keeps the specie NO a string.
    from ruamel import yaml

    dirs = map(Path, _mechanism.ct.get_data_directories())
    path = next(d/_DATABASE for d in dirs if (d/_DATABASE).is_file())
    species = yaml.YAML(typ="safe").load(path)["species"]
    return {s["name"].upper(): s["critical-parameters"] for s in species}


def _from_parameters(params, source: Source) -> Critical:
    Tc = float(params["critical-temperature"])
    Pc = float(params["critical-pressure"])
    if "critical-molar-volume" in params:
        Vc = float(params["critical-molar-volume"])
    else:
        Zc = float(params.get("critical-compressibility", 0.29))
        Vc = Zc*_RR*Tc/Pc
    return Critical(Tc, Pc, Vc, float(params["acentric-factor"]), source)


def lookup(
    mechanism, given: Optional[Dict[str, Critical]] = None,
) -> Dict[str, Critical]:
    """Critical data of every specie of a mechanism.

    The first found of: `given`, the mechanism's `critical-parameters`,
    `OPENFOAM`, Cantera's database, an estimate from the Lennard-Jones
    parameters (Tc and Vc as Chung et al., omega 0).

    Args:
        mechanism: A `Mechanism` or a mechanism path.
        given: `{specie: Critical}` taking precedence, in the mechanism's
            spelling.

    Returns:
        `{specie: Critical}` in the mechanism's order.

    Raises:
        ValueError: If `given` names a specie the mechanism lacks.
    """
    require_cantera()
    gas = Mechanism.coerce(mechanism).gas
    given = given or {}
    unknown = set(given) - set(gas.species_names)
    if unknown:
        raise ValueError(
            "critical data given for species not in the mechanism: "
            f"{', '.join(sorted(unknown))}"
        )

    database = _cantera_database()
    result = {}
    for sp in gas.species():
        name = sp.name.upper()
        normal = _NORMAL.match(name)
        key = _ALIASES.get(name, normal.group(1) if normal else name)
        if sp.name in given:
            result[sp.name] = given[sp.name]
        elif "critical-parameters" in sp.input_data:
            result[sp.name] = _from_parameters(
                sp.input_data["critical-parameters"], Source.MECHANISM,
            )
        elif name in OPENFOAM:
            result[sp.name] = dataclasses.replace(
                OPENFOAM[name], source=Source.OPENFOAM,
            )
        elif key in database:
            result[sp.name] = _from_parameters(database[key], Source.CANTERA)
        else:
            transport = sp.transport
            Tc = 1.2593*transport.well_depth/_KB
            Vc = (transport.diameter*1e10/0.809)**3*1e-3
            result[sp.name] = Critical(
                Tc, 0.291*_RR*Tc/Vc, Vc, 0.0, Source.LENNARD_JONES,
            )
    return result


def lump(
    data: Dict[str, Critical], mole_fractions: Dict[str, float],
) -> Critical:
    """Critical data of a mixture, mixed as OpenFOAM's
    `PengRobinsonGas::operator+=` does (Kay's rule): Tc, Vc, Zc and omega
    linear in the mole fractions, Pc from them.

    Args:
        data: `{specie: Critical}`, e.g. from `lookup()`.
        mole_fractions: `{specie: fraction}`; need not sum to 1.
    """
    total = sum(mole_fractions.values())
    Tc = Vc = Zc = omega = 0
    for name, x in mole_fractions.items():
        c = data[name]
        x /= total
        Tc += x*c.Tc
        Vc += x*c.Vc
        Zc += x*c.Pc*c.Vc/(_RR*c.Tc)
        omega += x*c.omega
    return Critical(Tc, _RR*Zc*Tc/Vc, Vc, omega)


def summary(data: Dict[str, Critical]) -> str:
    """One line per source: how many species and which."""
    lines = []
    for source in Source:
        names = [n for n, c in data.items() if c.source == source]
        if names:
            lines.append(
                f"critical data from {source.value}: {len(names)}: "
                f"{' '.join(names)}"
            )
    return "\n".join(lines)
