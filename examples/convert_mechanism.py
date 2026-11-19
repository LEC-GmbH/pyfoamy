"""Example: OpenFOAM `reactions` and `thermo` files from a Cantera mechanism.

The same as `python -m pyfoamy.cantera gri30.yaml`, step by step. A CHEMKIN
or CTI mechanism loads with `Mechanism.from_chemkin()`/`from_cti()`.
"""

import warnings
from pathlib import Path

from pyfoamy.cantera import ReactionConverter, ThermoDatabase
from pyfoamy.cantera.mechanism import Mechanism
from pyfoamy.parsing.grammar import Dictionary

# Shipped with Cantera.
MECHANISM = "gri30.yaml"


def main():
    mech = Mechanism(MECHANISM)

    # 1. Every reaction, with the type names of the sourced OpenFOAM.
    reactions = ReactionConverter(mech).reactions_dict()
    print("reaction0:", reactions["reaction0"]["type"])

    # 2. Every species refit to one shared Tcommon, as coefficient-blending
    #    mixtures need; species() would keep each NASA7 polynomial as is.
    db = ThermoDatabase(mech, transport_model="sutherland")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sutherland's fit-quality warnings
        species = db.species_group()
    print("Tcommon:", species["CH4"]["thermodynamics"]["Tcommon"])

    # 3. The files for `#include` in physicalProperties/chemistryProperties.
    Path("reactions").write_text(
        Dictionary({"reactions": reactions}).as_text + "\n"
    )
    Path("thermo").write_text(
        Dictionary({"species": list(species), **species}).as_text + "\n"
    )
    print(f"wrote {len(reactions)} reactions and {len(species)} species")


if __name__ == "__main__":
    main()
