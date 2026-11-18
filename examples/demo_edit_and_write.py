"""Demo: reading, editing and writing an OpenFOAM dictionary.

Needs a sourced OpenFOAM ($FOAM_TUTORIALS).
"""

import os
from pathlib import Path

from pyfoamy.parsing.grammar import Dictionary

CONTROL_DICT = (
    Path(os.environ["FOAM_TUTORIALS"])
    / "incompressibleFluid" / "pitzDaily" / "system" / "controlDict"
)
# Written to the working directory, not over the tutorial file.
OUTPUT_PATH = Path("controlDict.edited")


def main():
    d = Dictionary.from_file(CONTROL_DICT)

    print("--- before edits ---")
    print("endTime:       ", float(d["endTime"]))
    print("adjustTimeStep:", d["adjustTimeStep"].as_switch().value)
    print("purgeWrite:    ", int(d["purgeWrite"]))

    # 1. change an existing scalar value
    d["endTime"] = 1.0

    # 2. change a boolean value
    d["adjustTimeStep"] = False

    # 3. add a brand new scalar entry
    d["maxDeltaT"] = 0.01

    # 4. add a sub-dictionary from a plain dict
    d["functions"] = {"probe1": {"type": "probes", "fields": ["p", "U"]}}

    # 5. remove an entry entirely
    del d["purgeWrite"]

    print("\n--- after edits (still in memory) ---")
    print("endTime:       ", float(d["endTime"]))
    print("adjustTimeStep:", d["adjustTimeStep"].as_switch().value)
    print("maxDeltaT:     ", float(d["maxDeltaT"]))
    print("'purgeWrite' in d:", "purgeWrite" in d)

    written = d.as_text
    OUTPUT_PATH.write_text(written)

    print(f"\n--- wrote edited dictionary to {OUTPUT_PATH} ---")
    print(written)

    # Round trip: reparse the written file.
    d2 = Dictionary.from_file(OUTPUT_PATH)
    assert float(d2["endTime"]) == 1.0
    assert d2["adjustTimeStep"].as_switch().value is False
    assert float(d2["maxDeltaT"]) == 0.01
    assert "purgeWrite" not in d2
    fields = [str(v) for v in d2["functions"]["probe1"]["fields"]]
    assert fields == ["p", "U"]
    print("reparsed the written file - all edits round-tripped correctly.")


if __name__ == "__main__":
    main()
