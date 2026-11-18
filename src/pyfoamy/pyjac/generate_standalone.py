"""Standalone pyJac C-source generator, without importing `pyfoamy.case`.

Usage:
    python generate_standalone.py <mechanism.yaml> <last_spec> \\
        <case_root>

Example:
    python generate_standalone.py gri30.yaml N2 ./myCase
"""

import argparse
from pathlib import Path

from pyfoamy.cantera.mechanism import Mechanism
from pyfoamy.pyjac._source_tree import write_source_tree as generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mechanism", help="Mechanism file (e.g. .yaml)")
    parser.add_argument(
        "last_spec", help="Species to move to the last index (e.g. N2)",
    )
    parser.add_argument("case_root", help="OpenFOAM case directory")
    args = parser.parse_args()

    case_root = Path(args.case_root).resolve()
    mechanism = Mechanism(args.mechanism)

    species_names = generate(mechanism, case_root, args.last_spec)

    output_dir = case_root / "constant" / "foam" / "pyJacSrc"
    lib_path = case_root / "constant" / "foam" / "libc_pyjac"

    print(f"Wrote {len(species_names)} species, in order:")
    print(species_names)
    print(f"\nBuild with:\n  cd {output_dir} && ./Allwmake")
    print(f"-> {lib_path}.so")


if __name__ == "__main__":
    main()
