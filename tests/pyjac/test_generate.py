import ctypes
import os
import subprocess

import pytest

ct = pytest.importorskip("cantera")
pytest.importorskip("numpy")

from pyfoamy.case import Case  # noqa: E402
from pyfoamy.cantera.mechanism import Mechanism  # noqa: E402
from pyfoamy.pyjac.generate import generate_pyjac_mechanism  # noqa: E402


@pytest.fixture(scope="module")
def gri30():
    return Mechanism("gri30.yaml")


@pytest.fixture(scope="module")
def tiny_mechanism():
    # Compiles in under a second; full gri30 takes ~100 s.
    gas30 = ct.Solution("gri30.yaml")
    species = [gas30.species(n) for n in ["H2", "O2", "H2O", "N2"]]
    reaction = ct.Reaction(
        {"H2": 2.0, "O2": 1.0}, {"H2O": 2.0},
        ct.Arrhenius(1e13, 0.0, 5.0e7),
    )
    gas = ct.Solution(
        thermo="ideal-gas", kinetics="gas", species=species,
        reactions=[reaction],
    )
    return Mechanism.from_yaml(gas.write_yaml())


def test_generate_pyjac_mechanism_returns_reordered_species_names(
    tmp_path, gri30,
):
    names = generate_pyjac_mechanism(gri30, tmp_path, "N2")
    assert names[-1] == "N2"
    assert names[0] == gri30.gas.species_names[0]
    assert set(names) == set(gri30.gas.species_names)


def test_generate_pyjac_mechanism_writes_expected_files(
    tmp_path, tiny_mechanism,
):
    generate_pyjac_mechanism(tiny_mechanism, tmp_path, "N2")
    src_dir = tmp_path / "constant" / "foam" / "pyJacSrc"
    for name in [
        "mass_mole.c", "mass_mole.h", "chem_utils.c", "chem_utils.h",
        "rates.h", "rxn_rates.c", "spec_rates.c", "dydt.c", "dydt.h",
        "jacob.c", "jacob.h", "sparse_multiplier.c",
        "sparse_multiplier.h", "mechanism.h", "header.h",
        "pyjac_bridge.c", "Make/files", "Make/options", "Allwmake",
    ]:
        assert (src_dir / name).is_file(), name
    # no pressure dependence in the tiny mechanism, so no
    # rxn_rates_pres_mod.c is written.
    assert not (src_dir / "rxn_rates_pres_mod.c").exists()


def test_generate_pyjac_mechanism_allwmake_is_executable(
    tmp_path, tiny_mechanism,
):
    generate_pyjac_mechanism(tiny_mechanism, tmp_path, "N2")
    script = tmp_path / "constant" / "foam" / "pyJacSrc" / "Allwmake"
    assert os.access(script, os.X_OK)


def test_generate_pyjac_mechanism_make_files_lib_path(
    tmp_path, tiny_mechanism,
):
    generate_pyjac_mechanism(tiny_mechanism, tmp_path, "N2")
    make_files = (
        tmp_path / "constant" / "foam" / "pyJacSrc" / "Make" / "files"
    ).read_text()
    expected_lib = (
        tmp_path.resolve() / "constant" / "foam" / "libc_pyjac"
    )
    assert f"LIB = {expected_lib}\n" in make_files
    assert "jacob.c" in make_files
    assert "pyjac_bridge.c" in make_files


def test_generate_pyjac_mechanism_pyjac_bridge_content(
    tmp_path, tiny_mechanism,
):
    generate_pyjac_mechanism(tiny_mechanism, tmp_path, "N2")
    bridge = (
        tmp_path / "constant" / "foam" / "pyJacSrc" / "pyjac_bridge.c"
    ).read_text()
    assert bridge == (
        '#include "header.h"\n\n'
        "int PYJAC_NSP() { return NSP; }\n"
        "int PYJAC_FWD_RATES() { return FWD_RATES; }\n"
    )


def test_generate_pyjac_mechanism_with_case_adds_libs(
    tmp_path, tiny_mechanism,
):
    case = Case.create(tmp_path)
    generate_pyjac_mechanism(tiny_mechanism, case, "N2")
    control_dict = case["system/controlDict"]
    libs = [v.data for v in control_dict["libs"]]
    assert libs == ["$FOAM_CASE/constant/foam/libc_pyjac.so"]


def test_generate_pyjac_mechanism_with_path_does_not_touch_case(
    tmp_path, tiny_mechanism,
):
    # a bare path/string still works exactly as before - no
    # controlDict is written at all unless a real Case is passed.
    generate_pyjac_mechanism(tiny_mechanism, tmp_path, "N2")
    assert not (tmp_path / "system" / "controlDict").exists()


@pytest.mark.openfoam
def test_generate_pyjac_mechanism_builds_and_runs(
    tmp_path, tiny_mechanism,
):
    generate_pyjac_mechanism(tiny_mechanism, tmp_path, "N2")
    src_dir = tmp_path / "constant" / "foam" / "pyJacSrc"

    result = subprocess.run(
        ["./Allwmake"], cwd=src_dir, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    lib_path = tmp_path / "constant" / "foam" / "libc_pyjac.so"
    assert lib_path.is_file()

    lib = ctypes.CDLL(str(lib_path))
    lib.PYJAC_NSP.restype = ctypes.c_int
    lib.PYJAC_FWD_RATES.restype = ctypes.c_int
    assert lib.PYJAC_NSP() == 4
    assert lib.PYJAC_FWD_RATES() == 1

    nsp = lib.PYJAC_NSP()
    y = (ctypes.c_double * nsp)(1500.0, 0.05, 0.2, 0.0)
    jac = (ctypes.c_double * (nsp * nsp))()
    lib.eval_jacob.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.eval_jacob(1.0, 101325.0, y, jac)
    assert all(v == v for v in jac)  # no NaNs
    assert any(v != 0.0 for v in jac)
