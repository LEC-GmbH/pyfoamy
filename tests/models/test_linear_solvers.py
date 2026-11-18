import pytest

from pyfoamy.models.linear_solvers import (
    LinearSolver,
    check_solver_matches_symmetry,
)


def test_multigrid_resolves_to_gamg_regardless_of_symmetry():
    solver = LinearSolver("multigrid", tolerance=1e-7, rel_tol=0.01)
    assert solver.block(symmetric=True)["solver"] == "GAMG"
    assert solver.block(symmetric=False)["solver"] == "GAMG"


def test_smooth_resolves_to_smooth_solver_regardless_of_symmetry():
    solver = LinearSolver("smooth", tolerance=1e-6, rel_tol=0.1)
    assert solver.block(symmetric=True)["solver"] == "smoothSolver"
    assert solver.block(symmetric=False)["solver"] == "smoothSolver"


def test_conjugate_gradient_resolves_by_symmetry():
    solver = LinearSolver("conjugate_gradient", tolerance=1e-6, rel_tol=0.1)
    assert solver.block(symmetric=True)["solver"] == "PCG"
    assert solver.block(symmetric=False)["solver"] == "PBiCGStab"


def test_literal_solver_name_passes_through_unresolved():
    solver = LinearSolver("diagonal", tolerance=1e-6, rel_tol=0)
    assert solver.block(symmetric=True)["solver"] == "diagonal"
    assert solver.block(symmetric=False)["solver"] == "diagonal"


@pytest.mark.parametrize(
    "solver_name,symmetric,expected_key,expected_value",
    [
        ("PCG", True, "preconditioner", "DIC"),
        ("PBiCGStab", False, "preconditioner", "DILU"),
        ("GAMG", True, "smoother", "GaussSeidel"),
        ("smoothSolver", True, "smoother", "symGaussSeidel"),
    ],
)
def test_default_preconditioner_or_smoother_per_solver(
    solver_name, symmetric, expected_key, expected_value
):
    solver = LinearSolver(solver_name, tolerance=1e-6, rel_tol=0.1)
    block = solver.block(symmetric=symmetric)
    assert block[expected_key] == expected_value


def test_preconditioner_and_smoother_are_overridable():
    solver = LinearSolver(
        "GAMG", tolerance=1e-6, rel_tol=0.1, smoother="DICGaussSeidel"
    )
    assert solver.block(symmetric=True)["smoother"] == "DICGaussSeidel"


def test_preconditioner_is_overridable():
    solver = LinearSolver(
        "PCG", tolerance=1e-6, rel_tol=0.1, preconditioner="FDIC"
    )
    assert solver.block(symmetric=True)["preconditioner"] == "FDIC"


def test_tolerance_and_rel_tol_always_present():
    solver = LinearSolver("multigrid", tolerance=1e-7, rel_tol=0.01)
    block = solver.block(symmetric=True)
    assert block["tolerance"] == 1e-7
    assert block["relTol"] == 0.01


def test_final_block_forces_rel_tol_zero():
    solver = LinearSolver("multigrid", tolerance=1e-7, rel_tol=0.01)
    assert solver.final_block(symmetric=True)["relTol"] == 0


def test_min_iter_and_max_iter_omitted_by_default():
    solver = LinearSolver("multigrid", tolerance=1e-7, rel_tol=0.01)
    block = solver.block(symmetric=True)
    assert "minIter" not in block
    assert "maxIter" not in block


def test_min_iter_and_max_iter_included_when_given():
    solver = LinearSolver(
        "multigrid", tolerance=1e-7, rel_tol=0.01, min_iter=1, max_iter=50
    )
    block = solver.block(symmetric=True)
    assert block["minIter"] == 1
    assert block["maxIter"] == 50


def test_min_iter_final_defaults_to_min_iter():
    solver = LinearSolver(
        "multigrid", tolerance=1e-7, rel_tol=0.01, min_iter=3
    )
    assert solver.final_block(symmetric=True)["minIter"] == 3


def test_min_iter_final_independently_overridable():
    solver = LinearSolver(
        "multigrid",
        tolerance=1e-7,
        rel_tol=0.01,
        min_iter=3,
        min_iter_final=7,
    )
    assert solver.final_block(symmetric=True)["minIter"] == 7


def test_check_solver_matches_symmetry_rejects_pcg_for_asymmetric():
    with pytest.raises(ValueError, match="asymmetric"):
        check_solver_matches_symmetry("PCG", symmetric=False, context="U")


def test_check_solver_matches_symmetry_rejects_pbicgstab_for_symmetric():
    with pytest.raises(ValueError, match="symmetric"):
        check_solver_matches_symmetry(
            "PBiCGStab", symmetric=True, context="p"
        )


def test_check_solver_matches_symmetry_allows_matching_combinations():
    check_solver_matches_symmetry("PCG", symmetric=True, context="p")
    check_solver_matches_symmetry("PBiCGStab", symmetric=False, context="U")
    check_solver_matches_symmetry("GAMG", symmetric=True, context="p")
    check_solver_matches_symmetry("GAMG", symmetric=False, context="U")


def test_conjugate_gradient_family_never_triggers_mismatch():
    # Resolving through the abstraction should be structurally safe -
    # block() shouldn't ever raise for the family names themselves.
    solver = LinearSolver("conjugate_gradient", tolerance=1e-6, rel_tol=0.1)
    solver.block(symmetric=True)
    solver.block(symmetric=False)
