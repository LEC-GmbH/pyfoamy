"""Resolves linear-solver choices into `fvSolution` blocks."""

from dataclasses import dataclass
from typing import Optional

_FAMILY_TO_SOLVER = {
    "multigrid": {True: "GAMG", False: "GAMG"},
    "smooth": {True: "smoothSolver", False: "smoothSolver"},
    "conjugate_gradient": {True: "PCG", False: "PBiCGStab"},
}

_DEFAULT_PRECONDITIONER = {
    "PCG": "DIC",
    "PBiCG": "DILU",
    "PBiCGStab": "DILU",
}

_DEFAULT_SMOOTHER = {
    "GAMG": "GaussSeidel",
    "smoothSolver": "symGaussSeidel",
}

_SYMMETRIC_ONLY = {"PCG"}
_ASYMMETRIC_ONLY = {"PBiCG", "PBiCGStab"}


def check_solver_matches_symmetry(
    solver: str, symmetric: bool, context: str
) -> None:
    """Checks a resolved solver name against the matrix symmetry.

    Args:
        solver: OpenFOAM solver name, e.g. `"PCG"`.
        symmetric: Whether the matrix is symmetric.
        context: Field/equation name for the error message.

    Raises:
        ValueError: If `solver` needs the opposite symmetry.
    """
    if symmetric and solver in _ASYMMETRIC_ONLY:
        raise ValueError(
            f"{solver!r} requires an asymmetric matrix, but {context} "
            "is symmetric"
        )
    if not symmetric and solver in _SYMMETRIC_ONLY:
        raise ValueError(
            f"{solver!r} requires a symmetric matrix, but {context} "
            "is asymmetric"
        )


@dataclass
class LinearSolver:
    """A linear-solver choice for one or more `fvSolution` entries.

    Args:
        family: `"multigrid"` (`GAMG`), `"smooth"` (`smoothSolver`),
            `"conjugate_gradient"` (`PCG` if symmetric, else `PBiCGStab`,
            resolved per `.block()` call), or a literal solver name.
        tolerance: Absolute tolerance.
        rel_tol: Relative tolerance.
        min_iter: Omitted from the block if `None` (OpenFOAM default `0`).
        max_iter: Omitted if `None` (OpenFOAM default `1000`).
        min_iter_final: For `.final_block()`; `None` uses `min_iter`.
        preconditioner: `None` uses the per-solver default
            (`_DEFAULT_PRECONDITIONER`).
        smoother: `None` uses the per-solver default
            (`_DEFAULT_SMOOTHER`).
    """
    family: str
    tolerance: float
    rel_tol: float
    min_iter: Optional[int] = None
    max_iter: Optional[int] = None
    min_iter_final: Optional[int] = None
    preconditioner: Optional[str] = None
    smoother: Optional[str] = None

    def _resolve_solver_name(self, symmetric: bool) -> str:
        resolved = _FAMILY_TO_SOLVER.get(self.family)
        if resolved is not None:
            return resolved[symmetric]
        return self.family

    def block(
        self,
        symmetric: bool,
        context: str = "this equation",
        rel_tol: Optional[float] = None,
        min_iter: Optional[int] = None,
    ) -> dict:
        """Resolves this family into an `fvSolution` block dict.

        Args:
            symmetric: Selects PCG vs. PBiCGStab for `"conjugate_gradient"`;
                ignored for `"multigrid"`/`"smooth"`.
            context: Field/equation name for the error message.
            rel_tol: Overrides `self.rel_tol` for this block.
            min_iter: Overrides `self.min_iter` for this block.

        Raises:
            ValueError: If the solver name doesn't match `symmetric`.
        """
        solver = self._resolve_solver_name(symmetric)
        check_solver_matches_symmetry(solver, symmetric, context)

        result = {
            "solver": solver,
            "tolerance": self.tolerance,
        }

        if self.preconditioner is not None:
            result["preconditioner"] = self.preconditioner
        else:
            preconditioner = _DEFAULT_PRECONDITIONER.get(solver)
            if preconditioner is not None:
                result["preconditioner"] = preconditioner

        if self.smoother is not None:
            result["smoother"] = self.smoother
        else:
            smoother = _DEFAULT_SMOOTHER.get(solver)
            if smoother is not None:
                result["smoother"] = smoother

        if rel_tol is not None:
            result["relTol"] = rel_tol
        else:
            result["relTol"] = self.rel_tol

        if min_iter is not None:
            result["minIter"] = min_iter
        elif self.min_iter is not None:
            result["minIter"] = self.min_iter

        if self.max_iter is not None:
            result["maxIter"] = self.max_iter

        return result

    def final_block(
        self, symmetric: bool, context: str = "this equation"
    ) -> dict:
        """Like `block()` for a `Final` block, with `relTol` forced to `0`.

        Args:
            symmetric: Whether the matrix is symmetric.
            context: Field/equation name for the error message.

        Raises:
            ValueError: If the solver name doesn't match `symmetric`.
        """
        min_iter_final = self.min_iter_final
        if min_iter_final is None:
            min_iter_final = self.min_iter
        return self.block(
            symmetric, context, rel_tol=0, min_iter=min_iter_final
        )
