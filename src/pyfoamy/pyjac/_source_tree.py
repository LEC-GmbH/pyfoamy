"""Writes the pyJac C source tree, shared by `generate*.py`."""

from pathlib import Path
from typing import List, Union

from pyfoamy.cantera.mechanism import Mechanism
from pyfoamy.pyjac import adapter, jacobian, mechanism_writer, rates

_BRIDGE_C = (
    '#include "header.h"\n\n'
    "int PYJAC_NSP() { return NSP; }\n"
    "int PYJAC_FWD_RATES() { return FWD_RATES; }\n"
)


def _write_bridge(path: Path) -> None:
    (path / "pyjac_bridge.c").write_text(_BRIDGE_C)


def _write_make_files(
    path: Path, lib_path: Path, debug: bool = False,
) -> None:
    sources = sorted(p.name for p in path.glob("*.c"))
    make_dir = path / "Make"
    make_dir.mkdir(exist_ok=True)
    (make_dir / "files").write_text(
        "\n".join(sources) + f"\n\nLIB = {lib_path}\n"
    )
    exe_inc_flags = "    -O0 \\\n    -g \\\n" if debug else ""
    (make_dir / "options").write_text(
        "EXE_INC = \\\n"
        f"{exe_inc_flags}"
        "    -I.\n"
        "\n"
        "LIB_LIBS = \\\n"
        "    -lm\n"
    )


def _write_allwmake(path: Path) -> None:
    script = path / "Allwmake"
    script.write_text(
        "#!/bin/sh\n"
        'cd "${0%/*}" || exit 1\n'
        "wmake -j libso .\n"
    )
    script.chmod(script.stat().st_mode | 0o111)


def write_source_tree(
    mechanism: Mechanism, case_root: Union[str, Path], last_spec: str,
    debug: bool = False, clamp_state: bool = False,
) -> List[str]:
    """Writes a pyJac C source tree for `mechanism`, built via `Allwmake`.

    Sources go to `<case_root>/constant/foam/pyJacSrc/`, the library to
    `<case_root>/constant/foam/libc_pyjac.so` (case-local).

    Args:
        last_spec: Species moved to the last index, typically an inert
            bath gas (N2, Ar, He).
        debug: Compile with `-O0 -g` (much faster build, slower run;
            wmake puts `EXE_INC` after its own `-O3`, so the last `-O`
            wins) and make `dydt()` and `eval_jacob()` overwrite
            `$FOAM_CASE/pyjac_debug_state_<pid>.log` with the state fed to
            `eval_rxn_rates()`, to reproduce a crash. Overwrites rather
            than appends: it fires on every evaluation.
        clamp_state: Clamp the input state of `dydt()` and
            `eval_jacob()` (temperature floored at a small positive
            value, mass fractions at zero) against non-physical trial
            states of the ODE integrator. Off by default as it changes
            the generated math.

    Returns:
        Species names in the order the generated code expects; use it
        instead of `mechanism.gas.species_names` for
        `chemistryProperties`/`thermophysicalProperties`.
    """
    case_root = Path(case_root).resolve()
    src_dir = case_root / "constant" / "foam" / "pyJacSrc"
    src_dir.mkdir(parents=True, exist_ok=True)

    species, reactions = adapter.extract(mechanism, last_spec)

    rates.write_mass_mole(src_dir, species)
    rates.write_chem_utils(src_dir, species)
    rates.write_rxn_rates(src_dir, species, reactions)
    if any(r.third_body or r.pressure_dependent for r in reactions):
        rates.write_rxn_pressure_mod(src_dir, reactions)
    seen_sp = rates.write_spec_rates(src_dir, species, reactions)
    rates.write_derivs(
        src_dir, species, reactions, seen_sp, debug=debug,
        clamp_state=clamp_state,
    )

    touched = jacobian.write_jacobian(
        src_dir, species, reactions, seen_sp, debug=debug,
        clamp_state=clamp_state,
    )
    jacobian.write_sparse_multiplier(src_dir, touched, len(species))

    mechanism_writer.write_mechanism_header(src_dir, species, reactions)
    mechanism_writer.write_header(src_dir)

    _write_bridge(src_dir)
    lib_path = case_root / "constant" / "foam" / "libc_pyjac"
    _write_make_files(src_dir, lib_path, debug=debug)
    _write_allwmake(src_dir)

    return [sp.name for sp in species]
