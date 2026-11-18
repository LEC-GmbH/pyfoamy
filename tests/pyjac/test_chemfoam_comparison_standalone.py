"""As test_chemfoam_comparison, with `generate_standalone.generate()`.

It does not touch a `Case`, so `libc_pyjac.so` is added to controlDict
by hand, as DLBFoam's `tutorials/chemFoam/gri_pyJac/Allrun` does.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import pytest

pytest.importorskip("cantera")
pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyfoamy.models import DLB_PYJAC, ReactionSpec  # noqa: E402
from pyfoamy.pyjac.generate_standalone import generate  # noqa: E402

from _chemfoam_case import (  # noqa: E402
    LAST_SPEC, MECHANISM_PATH, build_reactions_dict, build_species_dict,
    ignition_delay, load_filtered_mechanism, read_chemfoam_out,
    run_chemfoam, write_chemfoam_case,
)

_ODE = {"solver": "seulex", "absTol": 1e-12, "relTol": 1e-2}


@pytest.mark.openfoam
def test_standard_vs_pyjac_via_standalone_generation(tmp_path):
    if not MECHANISM_PATH.is_file():
        pytest.skip(f"{MECHANISM_PATH} not present in the repo root")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mechanism = load_filtered_mechanism()
        species = build_species_dict(mechanism)
    reactions = build_reactions_dict(mechanism)

    spec_exact = ReactionSpec(
        "laminar", chemistry_type="standard",
        initial_chemical_time_step=1e-7, ode=_ODE,
        chemistry_coefficients={"jacobian": "exact"},
    )
    spec_pyjac = ReactionSpec(
        "laminar", chemistry_type=DLB_PYJAC,
        initial_chemical_time_step=1e-7, ode=_ODE,
    )

    case_exact = write_chemfoam_case(
        tmp_path / "exact", spec_exact.chemistry_properties_dict(),
        species, reactions,
    )
    case_exact.write()

    # The returned species order (last_spec last) must be used for the
    # case: loadBalanced_pyJac indexes concentrations by position.
    pyjac_root = tmp_path / "pyjac"
    src_dir = pyjac_root / "constant" / "foam" / "pyJacSrc"
    lib_path = pyjac_root.resolve() / "constant" / "foam" / "libc_pyjac"
    names = generate(mechanism, pyjac_root, LAST_SPEC, debug=True)
    assert names[-1] == LAST_SPEC

    case_pyjac = write_chemfoam_case(
        pyjac_root, spec_pyjac.chemistry_properties_dict(),
        species, reactions, species_order=names,
        extra_libs=(
            *spec_pyjac.required_libs(),
            "$FOAM_CASE/constant/foam/libc_pyjac.so",
        ),
    )
    case_pyjac.write()

    result = subprocess.run(
        ["./Allwmake"], cwd=src_dir, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert lib_path.with_suffix(".so").is_file()

    elapsed = {}
    rows = {}
    for label, case in (("exact", case_exact), ("pyjac", case_pyjac)):
        elapsed[label] = run_chemfoam(case)
        rows[label] = read_chemfoam_out(case.root)

    delay = {label: ignition_delay(r) for label, r in rows.items()}
    print("\nchemFoam wall-clock (s):", elapsed)
    print("ignition delay (s):", delay)

    gap = abs(delay["exact"] - delay["pyjac"]) / delay["exact"]
    print(f"exact-vs-pyjac (standalone) ignition-delay gap: {gap:.3%}")
    assert gap < 0.05
