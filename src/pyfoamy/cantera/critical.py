"""Critical data of a mechanism's species for a cubic equation of state."""

import dataclasses
import enum
import math
import re
from pathlib import Path
from typing import Dict, Optional

from pyfoamy.cantera import mechanism as _mechanism
from pyfoamy.cantera.mechanism import Mechanism, require_cantera

# Boltzmann constant [J/K], universal gas constant [J/kmol/K]
_KB = 1.380649e-23
_RR = 8314.46

# Cantera's PengRobinsonPhase, as pyfoamyPengRobinsonGas
_OMEGA_A = 0.45723552892138218
_OMEGA_B = 0.0777960739038885
_OMEGA_SPLIT = 0.491


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
        omega: Acentric factor [-].
        source: Where the data come from.
    """
    Tc: float
    Pc: float
    omega: float
    source: Source = Source.GIVEN

    def to_dict(self) -> dict:
        """Returns the `equationOfState` sub-dict of `pyfoamyPengRobinsonGas`."""
        return {"Tc": self.Tc, "Pc": self.Pc, "omega": self.omega}


#: OpenFOAM's liquidProperties: the species Cantera's database lacks.
OPENFOAM = {
    "H2O": Critical(647.13, 2.2055e7, 0.3449),
    "N2": Critical(126.10, 3.3944e6, 0.0403),
    "AR": Critical(150.86, 4.8981e6, 0.0),
    "NH3": Critical(405.65, 1.1278e7, 0.2520),
    "C2H6": Critical(305.32, 4.872e6, 0.0995),
    "C3H8": Critical(369.83, 4.248e6, 0.1523),
    "CH3OH": Critical(512.58, 8.0959e6, 0.5656),
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
    return Critical(
        float(params["critical-temperature"]),
        float(params["critical-pressure"]),
        float(params["acentric-factor"]),
        source,
    )


def lookup(
    mechanism, given: Optional[Dict[str, Critical]] = None,
) -> Dict[str, Critical]:
    """Critical data of every specie of a mechanism.

    The first found of: `given`, the mechanism's `critical-parameters`,
    `OPENFOAM`, Cantera's database, an estimate from the Lennard-Jones
    parameters (Tc and Vc as Chung et al., Zc 0.291, omega 0).

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
                Tc, 0.291*_RR*Tc/Vc, 0.0, Source.LENNARD_JONES,
            )
    return result


def kappa(omega: float) -> float:
    """Peng-Robinson kappa of an acentric factor, Cantera's two fits."""
    if omega <= _OMEGA_SPLIT:
        return 0.37464 + omega*(1.54226 - 0.26992*omega)
    return 0.374642 + omega*(1.487503 + omega*(-0.164423 + 0.016666*omega))


def _omega(kappa_: float) -> float:
    # Inverse of kappa(); the fits do not meet at the split, between them
    # no omega exists and the split is taken.
    if kappa_ <= kappa(_OMEGA_SPLIT):
        c = kappa_ - 0.37464
        return 2*c/(1.54226 + math.sqrt(1.54226**2 - 4*0.26992*c))
    omega = _OMEGA_SPLIT
    if kappa_ < kappa(math.nextafter(_OMEGA_SPLIT, 1)):
        return omega
    # The cubic fit rises monotonically: Newton from the split converges.
    for _ in range(50):
        step = (kappa(omega) - kappa_)/(
            1.487503 + omega*(-0.328846 + 0.049998*omega)
        )
        omega -= step
        if abs(step) < 1e-15*max(1.0, abs(omega)):
            break
    return omega


def lump(
    data: Dict[str, Critical], mole_fractions: Dict[str, float],
) -> Critical:
    """Critical data of a mixture, mixed as `pyfoamyPengRobinsonGas` does.

    Van der Waals one-fluid mixing with zero binary interaction parameters:
    sqrt(a alpha) = A0 - A1 sqrt(T) and b are mole-fraction linear. The
    result is the pseudo-specie whose Tc, Pc and omega give the mixture's
    A0, A1 and b, so mixing it again in OpenFOAM is exact. Where the
    mixture's kappa falls between Cantera's two fits (omega 0.491, a gap of
    5.2e-4 in kappa) omega is 0.491.

    Args:
        data: `{specie: Critical}`, e.g. from `lookup()`.
        mole_fractions: `{specie: fraction}`; need not sum to 1.
    """
    total = sum(mole_fractions.values())
    A0 = A1 = b = 0.0
    for name, x in mole_fractions.items():
        c = data[name]
        x /= total
        sqrt_a = math.sqrt(_OMEGA_A/c.Pc)*_RR*c.Tc
        k = kappa(c.omega)
        A0 += x*sqrt_a*(1 + k)
        A1 += x*sqrt_a*k/math.sqrt(c.Tc)
        b += x*_OMEGA_B*_RR*c.Tc/c.Pc
    Tc = (A0/(A1 + math.sqrt(_OMEGA_A*_RR*b/_OMEGA_B)))**2
    Pc = _OMEGA_B*_RR*Tc/b
    sqrt_a = math.sqrt(_OMEGA_A/Pc)*_RR*Tc
    return Critical(Tc, Pc, _omega(A1*math.sqrt(Tc)/sqrt_a))


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
