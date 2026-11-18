"""Builds the single-cell chemFoam cases of the pyJac comparison tests.

chemFoam's dictionaries do not fit `MulticomponentFluidSetup`, which
targets foamRun.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cantera as ct

from pyfoamy.case import Case
from pyfoamy.cantera.mechanism import Mechanism
from pyfoamy.cantera.reactions import ReactionConverter
from pyfoamy.cantera.species_data import ThermoDatabase
from pyfoamy.models import ReactionSpec
from pyfoamy.models.compressible_fluid import append_include
from pyfoamy.parsing.grammar import String

MECHANISM_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "KAUST2023_Mod" / "converted.yml"
)
LAST_SPEC = "N2"
INITIAL_PRESSURE = 1.36789e6
INITIAL_TEMPERATURE = 1000
INITIAL_FRACTIONS = {"CH4": 0.5, "O2": 1, "N2": 3.76}
END_TIME = 0.07

# Excluded: HOCO/CH2OCH have Cp/H/S jumps at Tcommon above the 1 %
# threshold of ThermoDatabase.species() (mechanism data).
_EXCLUDED_SPECIES = {"HOCO", "CH2OCH"}


# PLOG pairs the pyJac port cannot differentiate (equal pressures,
# all-zero b/E) or emits invalid C for (b differs, E equal).
def _plog_unsupported(reaction: "ct.Reaction") -> bool:
    if reaction.reaction_type != "pressure-dependent-Arrhenius":
        return False
    points = sorted(reaction.rate.rates, key=lambda pr: pr[0])
    pressures = [p for p, _ in points]
    if len(set(pressures)) != len(pressures):
        return True
    for (_, r1), (_, r2) in zip(points, points[1:]):
        b1, e1 = r1.temperature_exponent, r1.activation_energy
        b2, e2 = r2.temperature_exponent, r2.activation_energy
        if not (b1 or e1 or b2 or e2):
            return True
        if (b2 - b1 != 0.0) and (e2 - e1 == 0.0):
            return True
    return False


def load_filtered_mechanism() -> Mechanism:
    """Loads KAUST2023_Mod without the unsupported species/reactions.

    Three-body pressure-dependent Arrhenius is not supported by the
    adapter; 150 of 152 species and 1,360 of 1,385 reactions remain.
    """
    gas = ct.Solution(str(MECHANISM_PATH))
    excluded_reactions = {
        i for i, r in enumerate(gas.reactions())
        if r.reaction_type == "three-body-pressure-dependent-Arrhenius"
        or _plog_unsupported(r)
        or (_EXCLUDED_SPECIES & (set(r.reactants) | set(r.products)))
    }
    kept_species = [
        sp for sp in gas.species() if sp.name not in _EXCLUDED_SPECIES
    ]
    kept_reactions = [
        r for i, r in enumerate(gas.reactions())
        if i not in excluded_reactions
    ]
    filtered = ct.Solution(
        thermo=gas.thermo_model, kinetics=gas.kinetics_model,
        transport_model=gas.transport_model,
        species=kept_species, reactions=kept_reactions,
    )
    return Mechanism.coerce(filtered)


def build_species_dict(mechanism: Mechanism) -> dict:
    db = ThermoDatabase(
        mechanism, thermo_model="janaf", transport_model="const",
    )
    return db.species()


def build_reactions_dict(mechanism: Mechanism) -> dict:
    return ReactionConverter(mechanism).reactions_dict()


def write_chemfoam_case(
    root: Path, chemistry_properties: dict, species: dict,
    reactions: dict, extra_libs: Tuple[str, ...] = (),
    species_order: Optional[Tuple[str, ...]] = None,
    last_spec: str = LAST_SPEC,
    initial_fractions: Optional[dict] = None,
    initial_pressure: float = INITIAL_PRESSURE,
    initial_temperature: float = INITIAL_TEMPERATURE,
    end_time: float = END_TIME,
) -> Case:
    """Writes a single-cell chemFoam case like `test/chemistry/gri`.

    Args:
        species_order: Species order of `physicalProperties`. For
            `loadBalanced_pyJac` it must be the order the generator
            returned (`last_spec` last): it indexes concentrations by
            position (`loadBalanced_pyJac_chemistryModel.C`).
    """
    if species_order is None:
        species_order = tuple(species)
    if initial_fractions is None:
        initial_fractions = INITIAL_FRACTIONS
    case = Case.create(root)

    case.add_dict("constant/physicalProperties", {
        "thermoType": {
            "type": "heRhoThermo",
            "mixture": "multicomponentMixture",
            "transport": "const",
            "thermo": "janaf",
            "energy": "sensibleEnthalpy",
            "equationOfState": "perfectGas",
            "specie": "specie",
        },
        "defaultSpecie": last_spec,
    })
    species_content = {"species": list(species_order)}
    species_content.update(species)
    case.add_dict(
        "constant/speciesThermo", species_content, header=False,
    )
    append_include(
        case["constant/physicalProperties"], "speciesThermo",
    )

    case.add_dict("constant/chemistryProperties", chemistry_properties)
    case.add_dict(
        "constant/reactions", {"reactions": reactions}, header=False,
    )
    append_include(case["constant/chemistryProperties"], "reactions")

    case.add_dict("constant/initialConditions", {
        "constantProperty": "pressure",
        "fractionBasis": "mole",
        "fractions": initial_fractions,
        "p": initial_pressure,
        "T": initial_temperature,
    })

    case["system/controlDict"] = {
        "startFrom": "startTime",
        "startTime": 0,
        "stopAt": "endTime",
        "endTime": end_time,
        "deltaT": 1e-05,
        "maxDeltaT": 1e-4,
        "adjustTimeStep": "on",
        "writeControl": "adjustableRunTime",
        "writeInterval": 0.01,
        "purgeWrite": 0,
        "writeFormat": "ascii",
        "writeCompression": "off",
        "timeFormat": "general",
        "timePrecision": 6,
        "runTimeModifiable": "yes",
    }
    case["system/fvSchemes"] = {
        "ddtSchemes": {"default": "Euler"},
        "gradSchemes": {}, "divSchemes": {}, "laplacianSchemes": {},
    }
    case["system/fvSolution"] = {
        "solvers": {
            String("Yi.*"): {
                "solver": "PBiCGStab", "preconditioner": "DILU",
                "tolerance": 1e-12, "relTol": 0,
            },
        },
    }
    case.add_dict(
        "system/configDict", {"DebugSwitches": {"SolverPerformance": 0}},
    )

    case.add_libs(*extra_libs)
    return case


def run_chemfoam(case: Case) -> float:
    """Runs `chemFoam` for `case`, returning wall-clock seconds.

    Raises:
        RuntimeError: If `chemFoam` exits nonzero (via `Case.run()`).
    """
    start = time.monotonic()
    case.run("chemFoam")
    return time.monotonic() - start


def read_chemfoam_out(root: Path) -> List[Tuple[float, float]]:
    """Reads `chemFoam.out`'s `(time, temperature)` columns."""
    rows = []
    for line in (root / "chemFoam.out").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        rows.append((float(cols[0]), float(cols[1])))
    return rows


def ignition_delay(rows: List[Tuple[float, float]]) -> float:
    """Ignition delay: the time of steepest `dT/dt`."""
    best_time, best_slope = rows[0][0], float("-inf")
    for (t0, T0), (t1, T1) in zip(rows, rows[1:]):
        slope = (T1 - T0) / (t1 - t0)
        if slope > best_slope:
            best_slope, best_time = slope, (t0 + t1) / 2
    return best_time
