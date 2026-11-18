"""Writes a pyJac C source tree for a `Case`, adding it to `libs`."""

from pathlib import Path
from typing import List, Union

from pyfoamy.cantera.mechanism import Mechanism
from pyfoamy.case import Case
from pyfoamy.pyjac._source_tree import write_source_tree


def generate_pyjac_mechanism(
    mechanism: Mechanism, case: Union[str, Path, Case], last_spec: str,
    debug: bool = False, clamp_state: bool = False,
) -> List[str]:
    """Writes a pyJac C source tree for `mechanism`, built via `Allwmake`.

    Does not write `chemistryProperties`/`thermophysicalProperties`;
    that is `models.multicomponent_fluid.MulticomponentFluidSetup`.

    Args:
        case: A `Case` (gets `add_libs` for
            `$FOAM_CASE/constant/foam/libc_pyjac.so`) or a case root
            path. The library is case-local, never `$FOAM_USER_LIBBIN`.
        last_spec: Species moved to the last index, typically an inert
            bath gas (N2, Ar, He).
        debug: Compile at `-O0 -g` and make `dydt()`/`eval_jacob()`
            overwrite a per-process state file on every call; see
            `write_source_tree()`.
        clamp_state: Clamp the input state of `dydt()`/`eval_jacob()`;
            see `write_source_tree()`.

    Returns:
        Species names in the order the generated code expects; use it
        instead of `mechanism.gas.species_names`.
    """
    case_obj = case if isinstance(case, Case) else None
    case_root = case_obj.root if case_obj is not None else case

    names = write_source_tree(
        mechanism, case_root, last_spec, debug=debug,
        clamp_state=clamp_state,
    )

    if case_obj is not None:
        case_obj.add_libs("$FOAM_CASE/constant/foam/libc_pyjac.so")

    return names
