"""Regions of a multi-region case (`regionSolvers` in controlDict)."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pyfoamy.parsing.grammar import Dictionary, KeywordEntry


@dataclass(frozen=True)
class Region:
    """A region name and its solver module, e.g. `"fluid"`, `"solid"`."""

    name: str
    solver_type: str


def list_regions(case_dir) -> List[Region]:
    """Lists the regions of `regionSolvers` in declaration order.

    Raises:
        KeyError: If the case has no `regionSolvers` (single region).
    """
    return control_dict_regions(
        Dictionary.from_file(Path(case_dir) / "system" / "controlDict")
    )


def control_dict_regions(control_dict: Dictionary) -> List[Region]:
    """Lists the regions of a parsed controlDict's `regionSolvers`.

    Raises:
        KeyError: If it has no `regionSolvers` (single region).
    """
    return [
        Region(str(entry.key), str(entry.value))
        for entry in control_dict["regionSolvers"]
        if isinstance(entry, KeywordEntry)
    ]
