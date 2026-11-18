"""chemFoam ignition delay: standard (fast, exact) vs loadBalanced_pyJac."""

import subprocess
import sys
import warnings
from pathlib import Path

import pytest

pytest.importorskip("cantera")
pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyfoamy.models import DLB_PYJAC, ReactionSpec  # noqa: E402
from pyfoamy.pyjac.generate import generate_pyjac_mechanism  # noqa: E402

from _chemfoam_case import (  # noqa: E402
    LAST_SPEC, MECHANISM_PATH, build_reactions_dict, build_species_dict,
    ignition_delay, load_filtered_mechanism, read_chemfoam_out,
    run_chemfoam, write_chemfoam_case,
)

_ODE = {"solver": "seulex", "absTol": 1e-12, "relTol": 1e-2}


@pytest.mark.openfoam
def test_standard_fast_vs_exact_vs_pyjac_chemfoam(tmp_path):
    if not MECHANISM_PATH.is_file():
        pytest.skip(f"{MECHANISM_PATH} not present in the repo root")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mechanism = load_filtered_mechanism()
        species = build_species_dict(mechanism)
    reactions = build_reactions_dict(mechanism)

    spec_fast = ReactionSpec(
        "laminar", chemistry_type="standard",
        initial_chemical_time_step=1e-7, ode=_ODE,
    )
    spec_exact = ReactionSpec(
        "laminar", chemistry_type="standard",
        initial_chemical_time_step=1e-7, ode=_ODE,
        chemistry_coefficients={"jacobian": "exact"},
    )
    spec_pyjac = ReactionSpec(
        "laminar", chemistry_type=DLB_PYJAC,
        initial_chemical_time_step=1e-7, ode=_ODE,
    )

    case_fast = write_chemfoam_case(
        tmp_path / "fast", spec_fast.chemistry_properties_dict(),
        species, reactions,
    )
    case_exact = write_chemfoam_case(
        tmp_path / "exact", spec_exact.chemistry_properties_dict(),
        species, reactions,
    )

    # The returned species order (last_spec last) must be used for the
    # case: loadBalanced_pyJac indexes concentrations by position.
    pyjac_root = tmp_path / "pyjac"
    names = generate_pyjac_mechanism(
        mechanism, pyjac_root, LAST_SPEC, debug=True,
    )
    assert names[-1] == LAST_SPEC
    case_pyjac = write_chemfoam_case(
        pyjac_root, spec_pyjac.chemistry_properties_dict(),
        species, reactions, species_order=names,
        extra_libs=(
            *spec_pyjac.required_libs(),
            "$FOAM_CASE/constant/foam/libc_pyjac.so",
        ),
    )

    for case in (case_fast, case_exact, case_pyjac):
        case.write()

    src_dir = case_pyjac.root / "constant" / "foam" / "pyJacSrc"
    result = subprocess.run(
        ["./Allwmake"], cwd=src_dir, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lib = case_pyjac.root / "constant" / "foam" / "libc_pyjac.so"
    assert lib.is_file()

    elapsed = {}
    rows = {}
    for label, case in (
        ("fast", case_fast), ("exact", case_exact), ("pyjac", case_pyjac),
    ):
        elapsed[label] = run_chemfoam(case)
        rows[label] = read_chemfoam_out(case.root)

    delay = {
        label: ignition_delay(r) for label, r in rows.items()
    }
    print("\nchemFoam wall-clock (s):", elapsed)
    print("ignition delay (s):", delay)

    exact_pyjac_gap = abs(
        delay["exact"] - delay["pyjac"]
    ) / delay["exact"]
    fast_pyjac_gap = abs(
        delay["fast"] - delay["pyjac"]
    ) / delay["fast"]
    print(
        f"exact-vs-pyjac ignition-delay gap: {exact_pyjac_gap:.3%}, "
        f"fast-vs-pyjac gap: {fast_pyjac_gap:.3%}",
    )

    # B (exact) and C (pyJac) are both unapproximated analytical
    # Jacobians of the same rate laws - expected to closely agree.
    assert exact_pyjac_gap < 0.05
