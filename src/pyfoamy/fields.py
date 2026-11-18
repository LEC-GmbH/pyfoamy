"""Default field-file templates, to scaffold a case's `0/` directory.

Values are numerically safe starting points (e.g. turbulence
quantities are never exactly `0`), not tuned initial conditions.
"""

import copy
import inspect
from dataclasses import dataclass, field as _dataclass_field
from typing import Optional

from pyfoamy.boundary_conditions import (
    AlphatWallFunction,
    BoundaryCondition,
    BoundaryField,
    Calculated,
    InletOutlet,
    PatchRole,
    UniformInletOutlet,
    WALL_ROLES,
    ZeroGradient,
    pressure_wall_patches,
    scalar_wall_patches,
    u_wall_patches,
)
from pyfoamy.parsing.grammar import Bracket, Dictionary, Uniform


# FoamFile `class` per value type; add the other vol*Field types when a
# template needs one.
_FIELD_CLASS_BY_TYPE = {
    "scalar": "volScalarField",
    "vector": "volVectorField",
    "tensor": "volTensorField",
}


def _takes_value(bc_class: type) -> bool:
    return "value" in inspect.signature(bc_class.__init__).parameters


def _build_patch(bc, value):
    # A patches= entry is an instance (used as-is; needed when the
    # constructor takes no plain "value", e.g. InletOutlet), or a class,
    # called with the field's value if its constructor takes one, else
    # with no arguments.
    if isinstance(bc, BoundaryCondition):
        return bc
    return bc(value) if _takes_value(bc) else bc()


@dataclass(frozen=True)
class FieldTemplate:
    """A default starting point for one field file.

    Args:
        class_name: `"scalar"`/`"vector"`/`"tensor"`, or a full `FoamFile`
            class such as `"volScalarField"`.
        dimensions: Named-dimension string, brackets optional.
        internal_value: Default `internalField` value.
        patches: `PatchRole` -> `BoundaryCondition` class or instance,
            see `_build_patch()`. Empty means no default
            `boundaryField`. `U`/`p` stay empty since their moving-wall
            values are per-case, see `default_wall_boundary_field()`.
    """

    class_name: str
    dimensions: str
    internal_value: object = 0
    patches: dict = _dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "class_name",
            _FIELD_CLASS_BY_TYPE.get(self.class_name, self.class_name),
        )
        dimensions = self.dimensions
        if not dimensions.startswith("["):
            dimensions = f"[{dimensions}]"
        object.__setattr__(self, "dimensions", dimensions)

    def boundary_field(self, value: Optional[object] = None) -> dict:
        """Returns the default `boundaryField` content, `PatchRole` -> BC.

        Args:
            value: Overrides `internal_value` for a `patches` entry given as a
                bare class.
        """
        value = self.internal_value if value is None else value
        return {
            role: _build_patch(bc, value)
            for role, bc in self.patches.items()
        }

    def to_dict(
        self, internal_value: Optional[object] = None, roles=None,
    ) -> dict:
        """Returns the field-file content for `Case.write_dict`.

        Args:
            internal_value: Overrides `self.internal_value`.
            roles: `PatchRole` -> patch names for the default
                `boundaryField`; without it only the constraint types.
        """
        value = (
            self.internal_value if internal_value is None
            else internal_value
        )
        result = {
            "dimensions": Bracket(self.dimensions[1:-1]),
            "internalField": Uniform(value),
        }
        if self.patches:
            result["boundaryField"] = BoundaryField(
                self.boundary_field(value), roles=roles or {},
            )
        return result


class FieldFile(Dictionary):
    """A registered field's dictionary, as returned by `Case.add_field()`.

    Adds `set_bc()` to `Dictionary`.
    """

    def set_bc(self, patch: str, bc: BoundaryCondition) -> None:
        """Sets one patch BC, adding `boundaryField` if absent.

        Overrides any template default for that patch.

        Args:
            patch: Patch or mesh group name.

        Raises:
            TypeError: If `bc` is not a `BoundaryCondition`.
        """
        if not isinstance(bc, BoundaryCondition):
            raise TypeError(
                f"set_bc() needs a BoundaryCondition instance, got "
                f"{type(bc).__name__}"
            )
        if "boundaryField" not in self:
            self["boundaryField"] = BoundaryField({})
        self["boundaryField"][patch] = copy.deepcopy(bc)


# `p`/`p_rgh` are compressible (`[pressure]`); an incompressible case
# must override with `[kinematicPressure]`. Keyed by base name; a
# phase-suffixed field (`T.u`) shares its base template. `U`, `p`,
# `p_rgh` and turbulence fields declare no `patches`: their wall
# treatment is per-case/per-model (see `default_wall_boundary_field()`).
_GENERIC_SCALAR_PATCHES = {role: ZeroGradient for role in WALL_ROLES}

_FIELD_TEMPLATES = {
    "U": FieldTemplate("vector", "velocity", (0, 0, 0)),
    "p": FieldTemplate("scalar", "pressure", 0),
    "p_rgh": FieldTemplate("scalar", "pressure", 0),
    "T": FieldTemplate(
        "scalar", "temperature", 300, patches=_GENERIC_SCALAR_PATCHES,
    ),
    "k": FieldTemplate("scalar", "turbulentKineticEnergy", 0.1),
    "epsilon": FieldTemplate("scalar", "turbulentEpsilon", 0.01),
    "omega": FieldTemplate("scalar", "turbulentOmega", 1),
    "nut": FieldTemplate("scalar", "turbulentViscosity", 0),
    "nuTilda": FieldTemplate("scalar", "turbulentViscosity", 0),
    "alphat": FieldTemplate(
        "scalar", "dynamicDiffusivity", 0,
        patches={
            PatchRole.WALL: AlphatWallFunction,
            PatchRole.MOVING_WALL: AlphatWallFunction,
            PatchRole.SLIP: Calculated,
            PatchRole.MOVING_SLIP: Calculated,
            PatchRole.INLET: Calculated,
        },
    ),
    "v2": FieldTemplate("scalar", "turbulentKineticEnergy", 0),
    "f": FieldTemplate("scalar", "rate", 0),
    "ReThetat": FieldTemplate("scalar", "", 0),
    "gammaInt": FieldTemplate("scalar", "", 1),
    # XiFluid's fixed field names (unlike MulticomponentFluidSetup's
    # species).
    "b": FieldTemplate(
        "scalar", "", 1,
        patches={
            **_GENERIC_SCALAR_PATCHES,
            PatchRole.INLET: UniformInletOutlet(1),
        },
    ),
    "Xi": FieldTemplate(
        "scalar", "", 1,
        patches={
            **_GENERIC_SCALAR_PATCHES,
            PatchRole.INLET: UniformInletOutlet(1),
        },
    ),
    "Su": FieldTemplate(
        "scalar", "velocity", 1,
        patches={
            **_GENERIC_SCALAR_PATCHES, PatchRole.INLET: InletOutlet(1),
        },
    ),
    "egr": FieldTemplate("scalar", "", 0, patches=_GENERIC_SCALAR_PATCHES),
    "ft": FieldTemplate("scalar", "", 0, patches=_GENERIC_SCALAR_PATCHES),
    "fu": FieldTemplate("scalar", "", 0, patches=_GENERIC_SCALAR_PATCHES),
}


def default_field(name: str) -> FieldTemplate:
    """Returns the default template for a field's base name.

    Pass `"T"`, not `"T.u"`, for a phase-suffixed field.

    Raises:
        KeyError: If `name` has no registered default.
    """
    if name not in _FIELD_TEMPLATES:
        raise KeyError(
            f"no default field template for {name!r}, registered "
            f"names are {sorted(_FIELD_TEMPLATES)}"
        )
    return _FIELD_TEMPLATES[name]


# The wall function of a turbulence field depends on the active model
# (see `TurbulenceSpec.wall_patches()`), not on the name alone.
_TURBULENCE_FIELD_NAMES = {
    "k", "epsilon", "omega", "nut", "nuTilda", "v2", "f", "ReThetat",
    "gammaInt",
}


def default_wall_boundary_field(name: str, **kwargs) -> dict:
    """Returns the default conditions per `PatchRole` for a field name.

    `U`, `p`, `p_rgh` and turbulence fields are special-cased, since
    their BCs need per-case data. Others use the template's `patches`,
    else `scalar_wall_patches()`.

    Args:
        name: Base name, or any unregistered name such as a species.
        **kwargs: `moving_wall_value` for `"U"`; `value`/
            `moving_wall_pressure` for `"p"`/`"p_rgh"`; `value` for
            template `patches` (default: the template's `internal_value`).
            Ignored otherwise.

    Returns:
        The wall roles, plus `INLET` if the template has one, for
        `BoundaryField(..., roles=)`.

    Raises:
        ValueError: For a turbulence field; its BC depends on the model,
            use `TurbulenceSpec.wall_patches()`.
        KeyError: If `moving_wall_value` is missing for `"U"`.
    """
    if name == "U":
        return u_wall_patches(kwargs["moving_wall_value"])
    if name in ("p", "p_rgh"):
        return pressure_wall_patches(
            kwargs.get("value", 0),
            moving_wall_pressure=kwargs.get(
                "moving_wall_pressure", "fixedFluxPressure"
            ),
        )
    if name in _TURBULENCE_FIELD_NAMES:
        raise ValueError(
            f"{name!r}'s wall boundary condition is turbulence-model-"
            "dependent - use TurbulenceSpec.wall_patches() instead"
        )
    template = _FIELD_TEMPLATES.get(name)
    if template is not None and template.patches:
        return template.boundary_field(kwargs.get("value"))
    return scalar_wall_patches()
