"""Demo: reading OpenFOAM dictionaries.

Needs a sourced OpenFOAM ($FOAM_TUTORIALS).
"""

import os
from pathlib import Path

from pyfoamy.parsing.grammar import Dictionary, Directive, KeywordEntry

CONTROL_DICT = (
    Path(os.environ["FOAM_TUTORIALS"])
    / "incompressibleFluid" / "pitzDaily" / "system" / "controlDict"
)


def demo_real_file():
    print(f"--- parsing a real file: {CONTROL_DICT} ---")
    d = Dictionary.from_file(CONTROL_DICT)

    top_level_keys = [
        node.key.data for node in d if isinstance(node, KeywordEntry)
    ]
    print("top-level keys:", top_level_keys)
    print("solver:         ", str(d["solver"]))
    print("endTime:        ", float(d["endTime"]))
    print("writeInterval:  ", float(d["writeInterval"]))
    print(
        "runTimeModifiable:",
        d["runTimeModifiable"].as_switch().value,
    )
    print("FoamFile.class: ", str(d["FoamFile"]["class"]))


def demo_structure():
    print("\n--- parsing a snippet to show the node/entry structure ---")
    text = """
    #includeEtc "caseDicts/setConstraintTypes"

    internalField   uniform (10 0 0);

    someBareVector  (10 0 0);

    boundaryField
    {
        inlet
        {
            type    fixedValue;
            value   $internalField;
        }
    }
    """
    d = Dictionary(text)

    for node in d:
        if isinstance(node, KeywordEntry) and isinstance(
            node.value, Dictionary
        ):
            print(
                f"KeywordEntry(key={node.key!r}, "
                f"subdict with {len(node.value)} entries)"
            )
        elif isinstance(node, KeywordEntry):
            print(f"KeywordEntry(key={node.key!r}, value={node.value!r})")
        elif isinstance(node, Directive):
            print(f"Directive(name={node.name!r}, args={node.args!r})")

    # Parsed "uniform (10 0 0)" is a plain Compound, not a Uniform.
    print(
        "\ninternalField, as parsed (a Compound, not auto-recognized "
        "as Uniform):",
        d["internalField"],
    )
    print(
        "someBareVector, as a real List:",
        [int(v) for v in d["someBareVector"]],
    )

    # A sub-dictionary value indexes like a Dictionary.
    inlet = d["boundaryField"]["inlet"]
    print("boundaryField.inlet.type: ", str(inlet["type"]))
    print(
        "boundaryField.inlet.value (a Variable, since it's a "
        "$-macro reference):",
        inlet["value"],
    )


if __name__ == "__main__":
    demo_real_file()
    demo_structure()
