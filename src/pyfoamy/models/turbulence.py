"""The field and dict requirements of a `momentumTransport` choice."""

from typing import Dict, List, Optional

from pyfoamy.boundary_conditions import (
    Calculated,
    EpsilonWallFunction,
    FixedValue,
    KqRWallFunction,
    NutkWallFunction,
    NutUSpaldingWallFunction,
    OmegaWallFunction,
    PatchRole,
    ZeroGradient,
)
from pyfoamy.enums import SimulationType, TurbulenceModel
from pyfoamy.models.coefficients import coefficients_entry

_RAS_MODEL_FIELDS = {
    TurbulenceModel.K_EPSILON: ["k", "epsilon"],
    TurbulenceModel.REALIZABLE_KE: ["k", "epsilon"],
    TurbulenceModel.RNG_K_EPSILON: ["k", "epsilon"],
    TurbulenceModel.LAUNDER_SHARMA_KE: ["k", "epsilon"],
    TurbulenceModel.K_OMEGA: ["k", "omega"],
    TurbulenceModel.K_OMEGA_2006: ["k", "omega"],
    TurbulenceModel.K_OMEGA_SST: ["k", "omega"],
    TurbulenceModel.K_OMEGA_SST_LM: [
        "k", "omega", "ReThetat", "gammaInt"
    ],
    TurbulenceModel.V2F: ["k", "epsilon", "v2", "f"],
    TurbulenceModel.SPALART_ALLMARAS: ["nuTilda"],
}

_LES_MODEL_FIELDS = {
    TurbulenceModel.SMAGORINSKY: [],
    TurbulenceModel.WALE: [],
    TurbulenceModel.K_EQN: ["k"],
    TurbulenceModel.DYNAMIC_K_EQN: ["k"],
}

_MODEL_FIELDS_BY_TYPE = {
    SimulationType.RAS: _RAS_MODEL_FIELDS,
    SimulationType.LES: _LES_MODEL_FIELDS,
}


def _wall_and_slip_patches(
    wall_bc: dict, slip_bc: dict, include_inlet: bool = False
) -> dict:
    patches = {
        PatchRole.WALL: wall_bc,
        PatchRole.MOVING_WALL: wall_bc,
        PatchRole.SLIP: slip_bc,
        PatchRole.MOVING_SLIP: slip_bc,
    }
    if include_inlet:
        # Only for `nut`: algebraic, so its inlet is `calculated` like
        # the slip roles. k/epsilon/omega/... need a case-specific
        # inlet BC (intensity/mixing length).
        patches[PatchRole.INLET] = slip_bc
    return patches


class TurbulenceSpec:
    """A `simulationType`/model choice for `constant/momentumTransport`.

    `nut` is required for any non-laminar choice; models may need more
    fields (see `_RAS_MODEL_FIELDS`/`_LES_MODEL_FIELDS`). `alphat` is not
    here: it belongs to the compressible solvers, not `momentumTransport`.

    Args:
        simulation_type: `"laminar"`, `"RAS"` or `"LES"`.
        model: Required (and validated) unless laminar, where it must be
            omitted.
        coefficients: Model constant overrides, e.g. `{"Cmu": 0.09}`,
            written to a bare `<model>` sub-dict
            (`pyfoamy.models.coefficients`); omitted if empty. Not allowed
            for laminar.
        entries: Further entries beside `model`, e.g. the LES
            `{"delta": "cubeRootVol", "cubeRootVolCoeffs": {...}}`. Not
            allowed for laminar.

    Raises:
        ValueError: If `simulation_type` is invalid, `model`/
            `coefficients`/`entries` is given for laminar, or `model` is
            not registered for the type.
    """

    def __init__(
        self,
        simulation_type: SimulationType,
        model: Optional[TurbulenceModel] = None,
        coefficients: Optional[dict] = None,
        entries: Optional[dict] = None,
    ):
        simulation_type = SimulationType.coerce(
            simulation_type, "simulation_type"
        )
        if simulation_type == SimulationType.LAMINAR:
            if model is not None:
                raise ValueError(
                    "model must not be given for simulation_type="
                    "'laminar'"
                )
            if coefficients is not None:
                raise ValueError(
                    "coefficients must not be given for "
                    "simulation_type='laminar'"
                )
            if entries is not None:
                raise ValueError(
                    "entries must not be given for "
                    "simulation_type='laminar'"
                )
        else:
            valid_models = _MODEL_FIELDS_BY_TYPE[simulation_type]
            if model not in valid_models:
                raise ValueError(
                    f"unknown {simulation_type} model {model!r}, "
                    f"registered models are "
                    f"{sorted(m.value for m in valid_models)}"
                )
            model = TurbulenceModel(model)
        self.simulation_type = simulation_type
        self.model = model
        self.coefficients = coefficients
        self.entries = entries

    def required_fields(self) -> List[str]:
        """Returns `nut` plus the model's transported fields, or `[]`."""
        if self.simulation_type == SimulationType.LAMINAR:
            return []
        extra = _MODEL_FIELDS_BY_TYPE[self.simulation_type][self.model]
        return ["nut"] + extra

    def required_div_categories(self) -> dict:
        """Maps each transported field to `"firstOrder"`; `{}` if laminar.

        Transported fields are `required_fields()` minus the algebraic `nut`.
        `firstOrder` regardless of the case's `div_order` keeps turbulence
        quantities bounded.
        """
        return {
            field: "firstOrder"
            for field in self.required_fields()
            if field != "nut"
        }

    def momentum_transport_dict(self) -> dict:
        """Returns the `constant/momentumTransport` content."""
        if self.simulation_type == SimulationType.LAMINAR:
            return {"simulationType": "laminar"}
        model_dict = {
            "model": self.model,
            "turbulence": True,
            "printCoeffs": True,
            **coefficients_entry(self.model, self.coefficients),
            **(self.entries or {}),
        }
        return {
            "simulationType": self.simulation_type,
            self.simulation_type: model_dict,
        }

    def wall_patches(self, values: Optional[dict] = None
                     ) -> Dict[str, dict]:
        """Returns default wall/slip BCs for every field of this choice.

        Covers the wall roles of `PatchRole`, plus `INLET` for `nut`
        (algebraic, no case-specific inlet value).

        Unlike `U`/`p`, a wall-function BC does not depend on wall motion:
        `tutorials/incompressibleFluid/impeller/0/k` uses one `kqRWallFunction`
        for moving and static walls. Slip roles get `zeroGradient`/
        `calculated(0)`, per `simpleRushtonNCC/system/nonCouplePatchFields` and
        <https://cfd.direct/openfoam/free-software/using-non-conformal-coupling/>.

        Args:
            values: Field name -> initial value for the wall-function entries,
                e.g. `{"k": 0.1, "omega": 1}`; omitted fields get `0`.

        Returns:
            Field name -> `{PatchRole: bc}`, for `BoundaryField(..., roles=)`
            per field.

        Raises:
            ValueError: For laminar, or `"v2f"` (no tutorial to base a
                convention on).
        """
        if self.simulation_type == SimulationType.LAMINAR:
            raise ValueError(
                "wall_patches() needs a non-laminar TurbulenceSpec"
            )
        if self.model == TurbulenceModel.V2F:
            raise ValueError(
                "no confirmed wall-function convention for v2f - no "
                "real tutorial in this OpenFOAM-dev checkout uses it"
            )

        values = values or {}

        def value(name):
            return values.get(name, 0)

        if self.model == TurbulenceModel.SPALART_ALLMARAS:
            return {
                "nuTilda": _wall_and_slip_patches(
                    FixedValue(value("nuTilda")), ZeroGradient()
                ),
                "nut": _wall_and_slip_patches(
                    NutUSpaldingWallFunction(value("nut")),
                    Calculated(0), include_inlet=True,
                ),
            }

        if self.simulation_type == SimulationType.LES:
            result = {
                "nut": _wall_and_slip_patches(
                    ZeroGradient(), ZeroGradient(), include_inlet=True,
                ),
            }
            if "k" in self.required_fields():
                result["k"] = _wall_and_slip_patches(
                    FixedValue(value("k")), ZeroGradient()
                )
            return result

        # RAS k-epsilon/k-omega family.
        result = {
            "k": _wall_and_slip_patches(
                KqRWallFunction(value("k")), ZeroGradient()
            ),
            "nut": _wall_and_slip_patches(
                NutkWallFunction(value("nut")), Calculated(0),
                include_inlet=True,
            ),
        }
        if "epsilon" in self.required_fields():
            result["epsilon"] = _wall_and_slip_patches(
                EpsilonWallFunction(value("epsilon")), ZeroGradient()
            )
        else:
            result["omega"] = _wall_and_slip_patches(
                OmegaWallFunction(value("omega")), ZeroGradient()
            )
        if self.model == TurbulenceModel.K_OMEGA_SST_LM:
            result["ReThetat"] = _wall_and_slip_patches(
                ZeroGradient(), ZeroGradient()
            )
            result["gammaInt"] = _wall_and_slip_patches(
                ZeroGradient(), ZeroGradient()
            )
        return result
