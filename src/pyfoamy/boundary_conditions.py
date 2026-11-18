"""Boundary-condition classes for a field's `boundaryField` entries."""

import copy
import enum

from pyfoamy.parsing.grammar import (
    Compound,
    Dictionary,
    Directive,
    String,
    Uniform,
    Variable,
)


class Applicability(enum.Enum):
    """Where a boundary condition is physically meaningful.

    OpenFOAM has no inlet/outlet patch type, only a generic `patch`, so
    `INLET_OUTLET` means "non-wall".
    """

    WALL = "wall"
    INLET_OUTLET = "inlet_outlet"
    ANY = "any"


class PatchRole(enum.Enum):
    """Physical role of a patch, the key of the default condition sets.

    Which patches (or patch groups) of a mesh play a role is case data:
    a `roles` mapping `PatchRole` -> names, see `patches_by_name()`.
    `SLIP`/`MOVING_SLIP` are frictionless walls, common at non-conformal
    interfaces.
    """

    WALL = "wall"
    MOVING_WALL = "movingWall"
    SLIP = "slip"
    MOVING_SLIP = "movingSlip"
    INLET = "inlet"


def patches_by_name(patches: dict, roles: dict) -> dict:
    """Replaces each `PatchRole` key by the patch names playing that role.

    Args:
        patches: `PatchRole` or patch name -> boundary condition.
        roles: `PatchRole` -> patch or group names; a role without names
            is dropped.
    """
    result = {}
    for key, bc in patches.items():
        names = roles.get(key, ()) if isinstance(key, PatchRole) else (key,)
        for name in names:
            result[name] = bc
    return result


def _uniform(value):
    # a macro or a Function1 form is written as it is
    if isinstance(value, (Variable, Compound, dict)):
        return value
    return Uniform(value)


class BoundaryCondition(Dictionary):
    """Base class for a `boundaryField` patch entry.

    A `Dictionary` subclass, so the instance itself is what a field's
    `boundaryField` holds. Subclasses set `applicability` as a class
    attribute; `GenericBoundaryCondition` sets it per instance.
    """

    applicability: Applicability = Applicability.ANY

    def __init__(self, **entries) -> None:
        super().__init__(entries)

    def __eq__(self, other) -> bool:
        if isinstance(other, dict):
            other = Dictionary(other)
        return super().__eq__(other)


class GenericBoundaryCondition(BoundaryCondition):
    """Escape hatch for an OpenFOAM patchField type without a class.

    Args:
        type_name: The OpenFOAM `type`.
        applicability: Defaults to `ANY` (no mesh-coupling check).
        **fields: The BC's other entries.
    """

    def __init__(
        self, type_name: str,
        applicability: Applicability = Applicability.ANY,
        **fields,
    ) -> None:
        super().__init__(type=type_name, **fields)
        self.applicability = applicability


class ZeroGradient(BoundaryCondition):
    """A `zeroGradient` boundary condition."""

    applicability = Applicability.ANY

    def __init__(self) -> None:
        super().__init__(type="zeroGradient")


class NoSlip(BoundaryCondition):
    """A `noSlip` boundary condition (zero-velocity wall)."""

    applicability = Applicability.WALL

    def __init__(self) -> None:
        super().__init__(type="noSlip")


class FixedValue(BoundaryCondition):
    """A `fixedValue` boundary condition."""

    applicability = Applicability.ANY

    def __init__(self, value) -> None:
        super().__init__(type="fixedValue", value=Uniform(value))


class Calculated(BoundaryCondition):
    """A `calculated` boundary condition, for derived fields.

    E.g. `p` in a buoyant case, recomputed from `p_rgh`
    (`tutorials/fluid/hotRoomBoussinesqSteady/0/p`). Needs an initial
    `value`.
    """

    applicability = Applicability.ANY

    def __init__(self, value) -> None:
        super().__init__(type="calculated", value=Uniform(value))


class FixedFluxPressure(BoundaryCondition):
    """A `fixedFluxPressure` boundary condition.

    Standard pressure BC for closed and/or buoyant domains; adjusts the
    boundary value to match the boundary flux
    (`tutorials/fluid/hotRoomBoussinesqSteady/0/p_rgh`).

    Args:
        value: The initial value, before the first flux correction; `None`
            omits it.
    """

    applicability = Applicability.ANY

    def __init__(self, value=None) -> None:
        fields = {"type": "fixedFluxPressure"}
        if value is not None:
            fields["value"] = _uniform(value)
        super().__init__(**fields)


class FixedFluxExtrapolatedPressure(BoundaryCondition):
    """A `fixedFluxExtrapolatedPressure` boundary condition.

    Alternative to `FixedFluxPressure` for a moving wall
    (`tutorials/incompressibleFluid/simpleRushtonNCC/0/p.orig`); less
    stable, so opt-in.
    """

    applicability = Applicability.ANY

    def __init__(self, value) -> None:
        super().__init__(
            type="fixedFluxExtrapolatedPressure", value=Uniform(value),
        )


class Slip(BoundaryCondition):
    """A `slip` boundary condition (frictionless, static wall)."""

    applicability = Applicability.ANY

    def __init__(self) -> None:
        super().__init__(type="slip")


class MovingWallVelocity(BoundaryCondition):
    """A `movingWallVelocity` boundary condition (moving no-slip wall).

    Args:
        value: The wall velocity.
    """

    applicability = Applicability.WALL

    def __init__(self, value) -> None:
        super().__init__(type="movingWallVelocity", value=Uniform(value))


class MovingWallSlipVelocity(BoundaryCondition):
    """A `movingWallSlipVelocity` boundary condition (moving slip wall).

    For a moving NCC interface, see
    `tutorials/incompressibleFluid/impeller/0/U` and
    <https://cfd.direct/openfoam/free-software/using-non-conformal-coupling/>.

    Args:
        value: The wall velocity.
    """

    applicability = Applicability.WALL

    def __init__(self, value) -> None:
        super().__init__(
            type="movingWallSlipVelocity", value=Uniform(value),
        )


class KqRWallFunction(BoundaryCondition):
    """A `kqRWallFunction` boundary condition, the wall function for `k`.

    Independent of the RAS model.
    """

    applicability = Applicability.WALL

    def __init__(self, value) -> None:
        super().__init__(type="kqRWallFunction", value=Uniform(value))


class EpsilonWallFunction(BoundaryCondition):
    """An `epsilonWallFunction` boundary condition (k-epsilon family)."""

    applicability = Applicability.WALL

    def __init__(self, value) -> None:
        super().__init__(type="epsilonWallFunction", value=Uniform(value))


class OmegaWallFunction(BoundaryCondition):
    """An `omegaWallFunction` boundary condition (k-omega family)."""

    applicability = Applicability.WALL

    def __init__(self, value) -> None:
        super().__init__(type="omegaWallFunction", value=Uniform(value))


class NutkWallFunction(BoundaryCondition):
    """A `nutkWallFunction` boundary condition.

    For RAS models that transport `k`; not valid for `SpalartAllmaras`,
    see `NutUSpaldingWallFunction`.
    """

    applicability = Applicability.WALL

    def __init__(self, value) -> None:
        super().__init__(type="nutkWallFunction", value=Uniform(value))


class NutUSpaldingWallFunction(BoundaryCondition):
    """A `nutUSpaldingWallFunction` boundary condition.

    The `nut` wall function for `SpalartAllmaras`, which has no `k`
    (`tutorials/incompressibleFluid/airFoil2D/0/nut`).
    """

    applicability = Applicability.WALL

    def __init__(self, value) -> None:
        super().__init__(
            type="nutUSpaldingWallFunction", value=Uniform(value),
        )


class InletOutlet(BoundaryCondition):
    """An `inletOutlet` boundary condition.

    Fixed value on inflow, `zeroGradient` on outflow.

    Args:
        inlet_value: The value where flow enters the domain.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(self, inlet_value) -> None:
        super().__init__(
            type="inletOutlet", inletValue=Uniform(inlet_value),
        )


class PressureInletOutletVelocity(BoundaryCondition):
    """A `pressureInletOutletVelocity` boundary condition.

    Derives the tangential velocity from the pressure solution.

    Args:
        value: Optional initial value.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(self, value=None) -> None:
        fields = {"type": "pressureInletOutletVelocity"}
        if value is not None:
            fields["value"] = Uniform(value)
        super().__init__(**fields)


class AlphatWallFunction(BoundaryCondition):
    """A `compressible::alphatWallFunction` boundary condition.

    Args:
        prt: Turbulent Prandtl number; `0.85` is the usual tutorial value.
    """

    applicability = Applicability.WALL

    def __init__(self, value, prt=0.85) -> None:
        super().__init__(
            type="compressible::alphatWallFunction",
            Prt=prt,
            value=Uniform(value),
        )


class AlphatJayatillekeWallFunction(BoundaryCondition):
    """A `compressible::alphatJayatillekeWallFunction` boundary condition.

    Args:
        prt: Turbulent Prandtl number; `0.85` is the usual tutorial value.
    """

    applicability = Applicability.WALL

    def __init__(self, value, prt=0.85) -> None:
        super().__init__(
            type="compressible::alphatJayatillekeWallFunction",
            Prt=prt,
            value=Uniform(value),
        )


class TurbulentKineticEnergy(BoundaryCondition):
    """A `turbulentKineticEnergy` boundary condition.

    Derives `k` from a turbulence intensity.

    Args:
        intensity: Turbulence intensity as a fraction, e.g. `0.05`.
        value: The initial value, e.g. `Variable("internalField")`; `None`
            omits it.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(self, intensity, value=None) -> None:
        fields = {
            "type": "turbulentKineticEnergy", "intensity": intensity,
        }
        if value is not None:
            fields["value"] = _uniform(value)
        super().__init__(**fields)


class TurbulentEpsilon(BoundaryCondition):
    """A `turbulentEpsilon` boundary condition.

    Derives `epsilon` from a mixing length.

    Args:
        value: The initial value, or `None` to omit.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(self, mixing_length, value=None) -> None:
        fields = {
            "type": "turbulentEpsilon", "mixingLength": mixing_length,
        }
        if value is not None:
            fields["value"] = _uniform(value)
        super().__init__(**fields)


class TurbulentOmega(BoundaryCondition):
    """A `turbulentOmega` boundary condition.

    Derives `omega` from a mixing length.

    Args:
        value: The initial value, or `None` to omit.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(self, mixing_length, value=None) -> None:
        fields = {
            "type": "turbulentOmega", "mixingLength": mixing_length,
        }
        if value is not None:
            fields["value"] = _uniform(value)
        super().__init__(**fields)


class TurbulentInlet(BoundaryCondition):
    """A `turbulentInlet` boundary condition.

    Superimposes a random fluctuation on a reference field.

    Args:
        reference_field: Field to fluctuate around, e.g.
            `"internalField"`.
        fluctuation_scale: Relative fluctuation amplitude.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(self, reference_field, fluctuation_scale, value) -> None:
        super().__init__(
            type="turbulentInlet",
            referenceField=reference_field,
            fluctuationScale=fluctuation_scale,
            value=Uniform(value),
        )


class Freestream(BoundaryCondition):
    """A `freestream` boundary condition.

    Like `InletOutlet`, for external aerodynamics.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(self, freestream_value) -> None:
        super().__init__(
            type="freestream", freestreamValue=Uniform(freestream_value),
        )


class FreestreamVelocity(BoundaryCondition):
    """A `freestreamVelocity` boundary condition, the far field for `U`."""

    applicability = Applicability.INLET_OUTLET

    def __init__(self, freestream_value, value) -> None:
        super().__init__(
            type="freestreamVelocity",
            freestreamValue=Uniform(freestream_value),
            value=Uniform(value),
        )


class FreestreamPressure(BoundaryCondition):
    """A `freestreamPressure` boundary condition, the far field for `p`."""

    applicability = Applicability.INLET_OUTLET

    def __init__(self, freestream_value) -> None:
        super().__init__(
            type="freestreamPressure",
            freestreamValue=Uniform(freestream_value),
        )


class RotatingWallVelocity(BoundaryCondition):
    """A `rotatingWallVelocity` boundary condition.

    A wall rotating at a fixed angular velocity about an axis
    (`tutorials/incompressibleVoF/mixerVessel/0/U`). Unit literals such
    as `126 [rpm]` are not supported; pass `omega` in SI or `rpm`.

    Args:
        origin: A point on the rotation axis.
        axis: The axis direction.
        omega: Angular velocity in rad/s, a value or Function1 form.
        value: Optional initial value.
        rpm: Rotational speed in 1/min instead of `omega`, written as
            OpenFOAM's backwards compatible `rpm` entry.

    Raises:
        ValueError: Unless exactly one of `omega`, `rpm` is given.
    """

    applicability = Applicability.WALL

    def __init__(
        self, origin, axis, omega=None, value=None, *, rpm=None,
    ) -> None:
        if (omega is None) == (rpm is None):
            raise ValueError(
                "RotatingWallVelocity() needs exactly one of omega/rpm"
            )
        fields = {
            "type": "rotatingWallVelocity",
            "origin": origin,
            "axis": axis,
        }
        if omega is not None:
            fields["omega"] = omega
        else:
            fields["rpm"] = rpm
        if value is not None:
            fields["value"] = Uniform(value)
        super().__init__(**fields)


class UniformInletOutlet(BoundaryCondition):
    """A `uniformInletOutlet` boundary condition.

    Like `InletOutlet`, but the inlet value may be a Function1
    (`tutorials/multicomponentFluid/filter/0/T`).

    Args:
        uniform_inlet_value: The value where flow enters the domain.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(self, uniform_inlet_value) -> None:
        super().__init__(
            type="uniformInletOutlet",
            uniformInletValue=uniform_inlet_value,
        )


class FlowRateInletVelocity(BoundaryCondition):
    """A `flowRateInletVelocity` boundary condition.

    Derives the inlet velocity from a flow rate (see
    `tutorials/fluid/angledDuct/0/U`).

    Args:
        mass_flow_rate: A plain value or Function1 form, e.g.
            `Compound("constant", 0.1)`.
        volumetric_flow_rate: Same forms as `mass_flow_rate`.
        mean_velocity: Same forms as `mass_flow_rate`.
        rho_inlet: Initial `rho` estimate, only with `mass_flow_rate`.
        profile: Named inlet profile, e.g. `"turbulentBL"`.
        value: Optional initial value.

    Raises:
        ValueError: Unless exactly one of `mass_flow_rate`,
            `volumetric_flow_rate`, `mean_velocity` is given.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(
        self, *, mass_flow_rate=None, volumetric_flow_rate=None,
        mean_velocity=None, rho_inlet=None, profile=None, value=None,
    ) -> None:
        given = [
            name for name, val in (
                ("mass_flow_rate", mass_flow_rate),
                ("volumetric_flow_rate", volumetric_flow_rate),
                ("mean_velocity", mean_velocity),
            )
            if val is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "FlowRateInletVelocity() needs exactly one of "
                "mass_flow_rate/volumetric_flow_rate/mean_velocity, "
                f"got {given}"
            )
        fields = {"type": "flowRateInletVelocity"}
        if mass_flow_rate is not None:
            fields["massFlowRate"] = mass_flow_rate
        if volumetric_flow_rate is not None:
            fields["volumetricFlowRate"] = volumetric_flow_rate
        if mean_velocity is not None:
            fields["meanVelocity"] = mean_velocity
        if rho_inlet is not None:
            fields["rhoInlet"] = rho_inlet
        if profile is not None:
            fields["profile"] = profile
        if value is not None:
            fields["value"] = Uniform(value)
        super().__init__(**fields)


class ExternalTemperature(BoundaryCondition):
    """An `externalTemperature` boundary condition.

    Heat exchange with an environment via `ta` + `h`, a heat flux `q`, or
    a heat power `Q`, optionally plus radiative flux `qr` (see
    `tutorials/multiRegion/CHT/{hotWire,wallBoiling}`).

    Args:
        ta: Ambient temperature as Function1, e.g.
            `Compound("constant", 300)`. Requires `h`.
        h: Heat transfer coefficient. Requires `ta`.
        q: Fixed heat flux, a value or Function1 form.
        Q: Total heat power, a value or Function1 form.
        qr: Name of an externally supplied radiative flux field.
        thickness_layers: Multi-layer wall thicknesses.
        kappa_layers: Multi-layer wall conductivities.

    Raises:
        ValueError: If none of `ta`+`h`, `q`, `Q` is given, or only one of
            `ta`/`h`.
    """

    applicability = Applicability.WALL

    def __init__(
        self, value, *, ta=None, h=None, q=None, Q=None, qr=None,
        thickness_layers=None, kappa_layers=None,
    ) -> None:
        if (ta is None) != (h is None):
            raise ValueError(
                "ExternalTemperature() ta/h must be given together"
            )
        if ta is None and q is None and Q is None:
            raise ValueError(
                "ExternalTemperature() needs ta+h, q, or Q"
            )
        fields = {"type": "externalTemperature"}
        if ta is not None:
            fields["Ta"] = ta
            fields["h"] = Uniform(h)
        if q is not None:
            fields["q"] = _uniform(q)
        if Q is not None:
            fields["Q"] = _uniform(Q)
        if qr is not None:
            fields["qr"] = qr
        if thickness_layers is not None:
            fields["thicknessLayers"] = thickness_layers
        if kappa_layers is not None:
            fields["kappaLayers"] = kappa_layers
        fields["value"] = _uniform(value)
        super().__init__(**fields)


class UniformTotalPressure(BoundaryCondition):
    """A `uniformTotalPressure` boundary condition.

    Args:
        p0: Total pressure as Function1, e.g. `Compound("constant", 1e5)`
            or a full `{"type": "table", ...}` dict.
        psi: Name of the compressibility field, e.g. `"psi"`, for the
            compressible relation; `None` is the incompressible one.
        gamma: Ratio of specific heats. Requires `psi`.

    Raises:
        ValueError: If only one of `psi`/`gamma` is given.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(self, p0, value=None, *, psi=None, gamma=None) -> None:
        if (psi is None) != (gamma is None):
            raise ValueError(
                "UniformTotalPressure() psi/gamma must be given together"
            )
        fields = {"type": "uniformTotalPressure", "p0": p0}
        if psi is not None:
            fields["psi"] = psi
            fields["gamma"] = gamma
        if value is not None:
            fields["value"] = Uniform(value)
        super().__init__(**fields)


class EntrainmentPressure(BoundaryCondition):
    """An `entrainmentPressure` boundary condition.

    Total pressure for a locally entraining flow
    (`tutorials/multicomponentFluid/SandiaD_LTS/0/p`).

    Args:
        p0: Entrainment total pressure, often `Variable("internalField")`.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(self, p0, value=None) -> None:
        fields = {"type": "entrainmentPressure", "p0": p0}
        if value is not None:
            fields["value"] = Uniform(value)
        super().__init__(**fields)


class WaveTransmissive(BoundaryCondition):
    """A `waveTransmissive` boundary condition (non-reflecting outlet).

    Args:
        field: Name of the field on this patch, e.g. `"p"`.
        gamma: Ratio of specific heats.
        field_inf: Far-field value.
        l_inf: Far-field distance the correction relaxes over.
    """

    applicability = Applicability.INLET_OUTLET

    def __init__(self, field, gamma, field_inf, l_inf, value=None) -> None:
        fields = {
            "type": "waveTransmissive",
            "field": field,
            "gamma": gamma,
            "fieldInf": field_inf,
            "lInf": l_inf,
        }
        if value is not None:
            fields["value"] = Uniform(value)
        super().__init__(**fields)


class UniformFixedValue(BoundaryCondition):
    """A `uniformFixedValue` boundary condition.

    Args:
        uniform_value: A constant or Function1 dict, e.g.
            `{"type": "table", "values": [...]}`; not wrapped in
            `Uniform(...)`.
    """

    applicability = Applicability.ANY

    def __init__(self, uniform_value) -> None:
        super().__init__(
            type="uniformFixedValue", uniformValue=uniform_value,
        )


def u_wall_patches(moving_value) -> dict:
    """Returns default `U` conditions per wall role.

    Args:
        moving_value: Wall velocity for `MOVING_WALL`/`MOVING_SLIP`.

    Returns:
        `PatchRole` -> condition, for `BoundaryField(..., roles=)`.
    """
    return {
        PatchRole.WALL: NoSlip(),
        PatchRole.MOVING_WALL: MovingWallVelocity(moving_value),
        PatchRole.SLIP: Slip(),
        PatchRole.MOVING_SLIP: MovingWallSlipVelocity(moving_value),
    }


def pressure_wall_patches(
    value=0, *, moving_wall_pressure: str = "fixedFluxPressure"
) -> dict:
    """Returns default `p`/`p_rgh` conditions per wall role.

    `fixedFluxPressure` on all of them; unlike `U`, pressure does not
    depend on wall motion or slip.

    Args:
        moving_wall_pressure: `"fixedFluxPressure"` or
            `"fixedFluxExtrapolatedPressure"` (less stable) for
            `MOVING_WALL`.

    Returns:
        `PatchRole` -> condition, for `BoundaryField(..., roles=)`.

    Raises:
        ValueError: If `moving_wall_pressure` is neither.
    """
    if moving_wall_pressure == "fixedFluxPressure":
        moving = FixedFluxPressure(value)
    elif moving_wall_pressure == "fixedFluxExtrapolatedPressure":
        moving = FixedFluxExtrapolatedPressure(value)
    else:
        raise ValueError(
            f"unknown moving_wall_pressure {moving_wall_pressure!r}, "
            "expected 'fixedFluxPressure' or "
            "'fixedFluxExtrapolatedPressure'"
        )
    wall = FixedFluxPressure(value)
    return {
        PatchRole.WALL: wall,
        PatchRole.MOVING_WALL: moving,
        PatchRole.SLIP: wall,
        PatchRole.MOVING_SLIP: wall,
    }


WALL_ROLES = (
    PatchRole.WALL, PatchRole.MOVING_WALL, PatchRole.SLIP,
    PatchRole.MOVING_SLIP,
)


def scalar_wall_patches(inlet_value=None) -> dict:
    """Returns default conditions for a passive scalar per wall role.

    `zeroGradient` on all of them (species, `T`, `XiFluid`'s `egr`/`fu`/
    `ft`, ...).

    Args:
        inlet_value: If given, adds `INLET`
            (`UniformInletOutlet(inlet_value)`); `None` omits it.

    Returns:
        `PatchRole` -> condition, for `BoundaryField(..., roles=)`.
    """
    zero_gradient = ZeroGradient()
    patches = {role: zero_gradient for role in WALL_ROLES}
    if inlet_value is not None:
        patches[PatchRole.INLET] = UniformInletOutlet(inlet_value)
    return patches


def alphat_wall_patches(value=0, prt=0.85) -> dict:
    """Returns default `alphat` conditions per wall role plus `INLET`.

    `AlphatWallFunction` on `WALL`/`MOVING_WALL`, `calculated` on
    `SLIP`/`MOVING_SLIP`/`INLET`.

    Args:
        prt: Turbulent Prandtl number, see `AlphatWallFunction`.

    Returns:
        `PatchRole` -> condition, for `BoundaryField(..., roles=)`.
    """
    wall_bc = AlphatWallFunction(value, prt=prt)
    calculated = Calculated(value)
    return {
        PatchRole.WALL: wall_bc,
        PatchRole.MOVING_WALL: wall_bc,
        PatchRole.SLIP: calculated,
        PatchRole.MOVING_SLIP: calculated,
        PatchRole.INLET: calculated,
    }


class BoundaryField(Dictionary):
    """A `boundaryField` dict including `caseDicts/setConstraintTypes`.

    A `Dictionary` subclass so `.set()` recognizes it. OpenFOAM matches a
    patch by constraint group (`empty`, `wedge`, `symmetry`, `cyclic`,
    `processor`, ...) as well as by name, so that include covers all
    constraint patches; `patches` only needs the real BC patches
    (`tutorials/incompressibleFluid/ballValve/0/p`).

    Args:
        patches: Patch name or `PatchRole` -> `BoundaryCondition`; each
            condition is copied in.
        roles: `PatchRole` -> patch or group names, required for
            `PatchRole` keys; see `patches_by_name()`.
        mesh: Optional `PolyMesh`; each key is resolved against its patches
            (by name or `inGroups`) and checked for an `Applicability`
            mismatch (a `WALL`-only BC on a non-wall patch or vice versa).

    Raises:
        TypeError: If a patch value is not a `BoundaryCondition`, or a
            key is a `PatchRole` and `roles` is `None`.
        ValueError: If `mesh` is given and a mismatch is found.
    """

    def __init__(self, patches: dict, mesh=None, roles=None) -> None:
        if roles is not None:
            patches = patches_by_name(patches, roles)
        elif any(isinstance(key, PatchRole) for key in patches):
            raise TypeError("BoundaryField() needs roles= for PatchRole keys")
        bad = [
            f"{key!r} ({type(value).__name__})"
            for key, value in patches.items()
            if not isinstance(value, BoundaryCondition)
        ]
        if bad:
            raise TypeError(
                "BoundaryField() patch values must be "
                "BoundaryCondition instances: " + ", ".join(bad)
            )
        super().__init__(patches)
        self.insert(0, Directive(
            "#includeEtc", args=String("caseDicts/setConstraintTypes"),
        ))
        if mesh is not None:
            self._check_against_mesh(patches, mesh)

    def set(self, key, value):
        # One condition object often serves several patches.
        if isinstance(value, BoundaryCondition):
            value = copy.deepcopy(value)
        return super().set(key, value)

    @staticmethod
    def _check_against_mesh(patches: dict, mesh) -> None:
        mismatches = []
        for key, bc in patches.items():
            if bc.applicability is Applicability.ANY:
                continue
            for name, mesh_patch in mesh.patches.items():
                if name != key and key not in mesh_patch.groups:
                    continue
                patch_type = mesh_patch.type
                is_wall = patch_type == "wall"
                if bc.applicability is Applicability.WALL and not is_wall:
                    mismatches.append(
                        f"{key!r} -> mesh patch {name!r} (type "
                        f"{patch_type!r}): {type(bc).__name__} is "
                        "WALL-only"
                    )
                elif (
                    bc.applicability is Applicability.INLET_OUTLET
                    and is_wall
                ):
                    mismatches.append(
                        f"{key!r} -> mesh patch {name!r} (type "
                        f"{patch_type!r}): {type(bc).__name__} is "
                        "non-wall-only"
                    )
        if mismatches:
            raise ValueError(
                "BoundaryField() patch/mesh applicability mismatch: "
                + "; ".join(mismatches)
            )
