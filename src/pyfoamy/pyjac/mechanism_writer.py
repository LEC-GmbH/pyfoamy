# Derived from pyJac 1.0.6, Copyright (c) 2016 Kyle E Niemeyer, Nicholas J
# Curtis; MIT licence, see LICENSE-pyJac.

"""Mechanism metadata C source (port of pyJac's `mech_auxiliary.py`)."""

from pathlib import Path
from typing import List

from pyfoamy.pyjac.records import Reaction, Species

_HEADER_GUARD = "#ifndef {0}\n#define {0}\n"


def write_mechanism_header(
    path: Path, species: List[Species], reactions: List[Reaction],
) -> None:
    """Writes `mechanism.h`: the `NSP`/`NN`/`FWD_RATES`/`REV_RATES`/
    `PRES_MOD_RATES` defines the other generated files include.

    Port of the `mechanism.h` part of pyJac's
    `mech_auxiliary.write_mechanism_initializers()`.

    Args:
        species: In the order the generated code expects (dependent
            species last).
    """
    num_s = len(species)
    num_r = len(reactions)
    num_rev = sum(1 for r in reactions if r.reversible)
    num_pdep = sum(
        1 for r in reactions if r.third_body or r.pressure_dependent
    )

    species_index_comment = "\n".join(
        f"{i}  {sp.name}" for i, sp in enumerate(species)
    )

    (path / "mechanism.h").write_text(
        _HEADER_GUARD.format("MECHANISM_h") + "\n"
        f"//last_spec {species[-1].name}\n"
        "/* Species Indexes\n"
        f"{species_index_comment}\n"
        "*/\n\n"
        "//Number of species\n"
        f"#define NSP {num_s}\n"
        "//Number of variables. NN = NSP + 1 (temperature)\n"
        f"#define NN {num_s + 1}\n"
        "//Number of forward reactions\n"
        f"#define FWD_RATES {num_r}\n"
        "//Number of reversible reactions\n"
        f"#define REV_RATES {num_rev}\n"
        "//Number of reactions with pressure modified rates\n"
        f"#define PRES_MOD_RATES {num_pdep}\n\n"
        "#endif\n\n"
    )


def write_header(path: Path) -> None:
    """Writes `header.h`: defines `CONP` and includes `mechanism.h`.

    Always constant pressure (never `CONV`): OpenFOAM integrates
    chemistry at constant pressure.
    """
    (path / "header.h").write_text(
        "#ifndef HEAD\n"
        "#define HEAD\n"
        "#include <stdlib.h>\n"
        "#include <math.h>\n"
        "/** Constant pressure or volume. */\n"
        "#define CONP\n"
        "//#define CONV\n"
        "\n"
        "/** Include mechanism header to get NSP and NN **/\n"
        '#include "mechanism.h"\n'
        "// OpenMP\n"
        "#ifdef _OPENMP\n"
        " #include <omp.h>\n"
        "#else\n"
        " #define omp_get_max_threads() 1\n"
        " #define omp_get_num_threads() 1\n"
        "#endif\n"
        "#endif\n"
    )
