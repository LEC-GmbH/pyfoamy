"""Shared base classes for the solver setup classes in this package."""

import os
from typing import Optional

from pyfoamy.case import Case
from pyfoamy.custom_lib import ensure_built
from pyfoamy.casefiles import foam_file_header
from pyfoamy.models.linear_solvers import LinearSolver
from pyfoamy.models.schemes import (
    Order, resolve_ddt_scheme, resolve_div_scheme, tokenize_scheme,
    unblended,
)
from pyfoamy.openfoam_version import openfoam_version
from pyfoamy.parsing.grammar import Bracket


class SolverSetup:
    """Shared base for solver setups; `U` is required.

    The pressure field defaults to `"p"`; subclasses may override it.
    Subclasses give `solver_name`, the constant dicts, the solved fields and
    the div terms; `build()` assembles the case from them.
    """

    solver_name: str = ""

    @property
    def pressure_field(self) -> str:
        return "p"

    def required_fields(self):
        """Returns the fields every setup needs: `["p", "U"]`."""
        return ["p", "U"]

    @property
    def _symmetric_pressure(self) -> bool:
        return True

    def _check_gravity(self, gravity) -> None:
        if gravity is not None:
            raise ValueError(f"{self.solver_name} takes no gravity")

    def _constant_dicts(self, case: Case, gravity) -> None:
        raise NotImplementedError

    def _momentum_group_fields(self):
        raise NotImplementedError

    def _div_order_categories(self) -> dict:
        raise NotImplementedError

    def _fixed_div_terms(self) -> dict:
        return {}

    def _fixed_solvers(self) -> dict:
        return {}

    def _grad_name(self, term: str) -> Optional[str]:
        # Gradient scheme a linearUpwind/LUST div scheme of `term` names;
        # None: grad(<field>).
        return None

    def _pimple_entries(self) -> dict:
        return {}

    def build(
        self,
        root,
        ddt_order: str,
        div_order: Order,
        p_solver: LinearSolver,
        momentum_solver: LinearSolver,
        gravity=None,
        grad_scheme="Gauss linear",
        grad_schemes: Optional[dict] = None,
        interpolation_scheme="linear",
        sn_grad_scheme="corrected",
        wall_dist: Optional[str] = "meshWave",
        n_outer_correctors: Optional[int] = None,
        n_correctors: Optional[int] = None,
        n_non_orthogonal_correctors: Optional[int] = None,
        momentum_predictor: Optional[bool] = None,
        pressure_reference=None,
        relaxation_factors: Optional[dict] = None,
        pimple: Optional[dict] = None,
        solvers: Optional[dict] = None,
        start_time=0,
        end_time=1,
        delta_t=1,
        write_control="timeStep",
        write_interval=1,
        default_backend="python",
        pyfoamy_lib: bool = True,
    ) -> Case:
        """Builds a populated `Case`; nothing is written until `case.write()`.

        Args:
            ddt_order: `ddtSchemes` order name (e.g. `"steady"`) or literal
                text.
            div_order: `divSchemes` order for `U` and the energy field: a
                name, literal text or `schemes.CourantBlend`.
            p_solver: Linear solver for the pressure field.
            momentum_solver: Linear solver for every other solved field,
                resolved asymmetric.
            gravity: `(x, y, z)`; required if the setup is buoyant.
            grad_schemes: Field name -> scheme text of its `grad(<field>)`
                entry, beside the default `grad_scheme`.
            interpolation_scheme: Also feeds `laplacianSchemes`.
            sn_grad_scheme: Also feeds `laplacianSchemes`.
            wall_dist: `wallDist.method`, which wall-distance turbulence
                models read; `None` writes no `wallDist`.
            n_outer_correctors: `PIMPLE.nOuterCorrectors`; omitted if `None`.
            n_correctors: `PIMPLE.nCorrectors`; omitted if `None`.
            n_non_orthogonal_correctors: `PIMPLE.nNonOrthogonalCorrectors`;
                omitted if `None`.
            momentum_predictor: `PIMPLE.momentumPredictor`; omitted if `None`.
            pressure_reference: `(cell, value)` for `PIMPLE.pRefCell`/
                `pRefValue`; omitted if `None`.
            relaxation_factors: `relaxationFactors` dict; omitted if `None`.
            pimple: Further `PIMPLE` entries, e.g. `consistent`.
            solvers: Further `solvers` entries, e.g. `MeshPhi`; a key with
                a regular expression is a `String`.
            default_backend: `Case.default_backend`.
            pyfoamy_lib: Load `libpyfoamy.so` (`controlDict` `libs`), built
                now if an OpenFOAM is sourced.

        Raises:
            ValueError: If `gravity` does not fit the setup's buoyancy, or
                the OpenFOAM version is unsupported.
            pyfoamy.custom_lib.CustomLibError: If building the library
                fails.
        """
        self._check_gravity(gravity)
        openfoam_version()
        if pyfoamy_lib and os.environ.get("WM_PROJECT_VERSION"):
            ensure_built()

        case = Case(root, default_backend=default_backend)
        case["system/controlDict"] = {
            "solver": self.solver_name,
            "startFrom": "startTime",
            "startTime": start_time,
            "endTime": end_time,
            "deltaT": delta_t,
            "writeControl": write_control,
            "writeInterval": write_interval,
        }
        if pyfoamy_lib:
            case.add_libs("libpyfoamy.so")
        self._constant_dicts(case, gravity)
        case["system/fvSchemes"] = self._fv_schemes(
            ddt_order,
            div_order,
            grad_scheme,
            interpolation_scheme,
            sn_grad_scheme,
            grad_schemes,
            wall_dist,
        )
        case["system/fvSolution"] = self._fv_solution(
            p_solver,
            momentum_solver,
            n_outer_correctors,
            n_correctors,
            n_non_orthogonal_correctors,
            momentum_predictor,
            pressure_reference,
            relaxation_factors,
            pimple,
            solvers,
        )
        return case

    def _fv_schemes(
        self,
        ddt_order,
        div_order,
        grad_scheme,
        interpolation_scheme,
        sn_grad_scheme,
        grad_schemes=None,
        wall_dist=None,
    ) -> dict:
        div_scheme_dict = {"default": "none"}
        for key, category in self._div_order_categories().items():
            vector = category == "defaultV"
            order = (
                unblended(key, div_order)
                if category in ("default", "defaultV") else category
            )
            div_scheme_dict[f"div({key})"] = resolve_div_scheme(
                key.rsplit(",", 1)[-1], order, vector,
                grad=self._grad_name(key),
            )
        for key, scheme_text in self._fixed_div_terms().items():
            div_scheme_dict[f"div({key})"] = tokenize_scheme(
                scheme_text
            )

        grad_scheme_dict = {"default": tokenize_scheme(grad_scheme)}
        for field, scheme_text in (grad_schemes or {}).items():
            grad_scheme_dict[f"grad({field})"] = tokenize_scheme(
                scheme_text
            )

        result = {
            "ddtSchemes": {"default": resolve_ddt_scheme(ddt_order)},
            "gradSchemes": grad_scheme_dict,
            "divSchemes": div_scheme_dict,
            "laplacianSchemes": {
                "default": tokenize_scheme(
                    f"Gauss {interpolation_scheme} {sn_grad_scheme}"
                )
            },
            "interpolationSchemes": {
                "default": tokenize_scheme(interpolation_scheme),
                "interpolate(Co)": tokenize_scheme("localMax"),
            },
            "snGradSchemes": {
                "default": tokenize_scheme(sn_grad_scheme)
            },
        }
        if wall_dist is not None:
            result["wallDist"] = {"method": wall_dist}
        return result

    def _fv_solution(
        self,
        p_solver,
        momentum_solver,
        n_outer_correctors,
        n_correctors,
        n_non_orthogonal_correctors,
        momentum_predictor,
        pressure_reference,
        relaxation_factors,
        pimple_entries=None,
        solver_entries=None,
    ) -> dict:
        symmetric_pressure = self._symmetric_pressure
        pressure_field = self.pressure_field
        solvers = {
            **self._fixed_solvers(),
            pressure_field: p_solver.block(
                symmetric=symmetric_pressure, context=pressure_field
            ),
            f"{pressure_field}Final": p_solver.final_block(
                symmetric=symmetric_pressure, context=pressure_field
            ),
        }
        for field in self._momentum_group_fields():
            solvers[field] = momentum_solver.block(
                symmetric=False, context=f"{field} (advective)"
            )
            solvers[f"{field}Final"] = momentum_solver.final_block(
                symmetric=False, context=f"{field} (advective)"
            )

        pimple = {}
        if n_outer_correctors is not None:
            pimple["nOuterCorrectors"] = n_outer_correctors
        if n_correctors is not None:
            pimple["nCorrectors"] = n_correctors
        if n_non_orthogonal_correctors is not None:
            pimple["nNonOrthogonalCorrectors"] = (
                n_non_orthogonal_correctors
            )
        if momentum_predictor is not None:
            pimple["momentumPredictor"] = momentum_predictor
        if pressure_reference is not None:
            cell, value = pressure_reference
            pimple["pRefCell"] = cell
            pimple["pRefValue"] = value
        pimple.update(self._pimple_entries())
        pimple.update(pimple_entries or {})
        solvers.update(solver_entries or {})

        result = {"solvers": solvers, "PIMPLE": pimple}
        if relaxation_factors is not None:
            result["relaxationFactors"] = relaxation_factors
        return result


class BuoyantSolverSetup(SolverSetup):
    """Shared base for solver setups with optional buoyancy and transonic.

    `fluid`, `multicomponentFluid` and `XiFluid` derive from
    `isothermalFluid`, which provides both: buoyancy is selected by a
    `p_rgh` field (`Foam::solvers::buoyancy`), and the transonic
    `pressureCorrector()` (adding `div(phid,p)`/`div(phid,p_rgh)`) is not
    overridden by any of them.
    """

    def __init__(self, buoyant: bool = False, transonic: bool = False):
        self.buoyant = buoyant
        self.transonic = transonic

    @property
    def pressure_field(self) -> str:
        return "p_rgh" if self.buoyant else "p"

    def required_fields(self):
        """Returns `["p", "U"]`, plus `"p_rgh"` when buoyant.

        `p` stays required: `isothermalFluid/correctBuoyantPressure.C`
        recomputes it as `p_rgh + rho*gh + pRef` and never solves for it, and
        every buoyant tutorial ships both `0/p` and `0/p_rgh`. So `p` needs no
        `fvSchemes`/`fvSolution` entry, and this is not built from
        `pressure_field`, which names only the solved field.
        """
        fields = ["p", "U"]
        if self.buoyant:
            fields = fields + ["p_rgh"]
        return fields

    @property
    def _symmetric_pressure(self) -> bool:
        return not self.transonic

    def _check_gravity(self, gravity) -> None:
        if self.buoyant and gravity is None:
            raise ValueError(
                "buoyant=True requires gravity=(x, y, z)"
            )

    def _gravity_entry(self, gravity) -> Optional[dict]:
        if not self.buoyant:
            return None
        return {
            "FoamFile": foam_file_header(
                "constant/g", "uniformDimensionedVectorField"
            ),
            "dimensions": Bracket("acceleration"),
            "value": list(gravity),
        }

    def _fixed_solvers(self) -> dict:
        return {"rho": {"solver": "diagonal"}}

    def _pimple_entries(self) -> dict:
        return {"transonic": True} if self.transonic else {}

    def _transonic_div_categories(self) -> dict:
        if self.transonic:
            return {f"phid,{self.pressure_field}": "default"}
        return {}
