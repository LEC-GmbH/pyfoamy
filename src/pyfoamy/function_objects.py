"""Builds entries for a case's `functions` dict."""

from dataclasses import dataclass, field
from numbers import Number
from typing import Dict, List, Optional, Tuple

from pyfoamy.enums import (
    SurfaceFieldValueOperation, TimeControlMode, VolFieldValueOperation,
)
from pyfoamy.include_entries import (
    IncludeEntryKind, TemplateEntry, entry_dict,
)
from pyfoamy.parsing.grammar import Code, Compound, String

_INTERVAL_REQUIRED = {
    TimeControlMode.ADJUSTABLE_RUN_TIME,
    TimeControlMode.RUN_TIME,
    TimeControlMode.CLOCK_TIME,
    TimeControlMode.CPU_TIME,
}
_INTERVAL_OPTIONAL = {
    TimeControlMode.TIME_STEP,
    TimeControlMode.WRITE_TIME,
}


@dataclass
class TimeControl:
    """An `executeControl`/`writeControl` choice for function objects.

    Which of `interval`/`times`/`frequencies`/`repeat` apply depends on
    `mode`; a wrong combination raises at construction:

    - `"timeStep"`/`"writeTime"`: `interval` optional.
    - `"adjustableRunTime"`/`"runTime"`/`"clockTime"`/`"cpuTime"`:
      `interval` required.
    - `"runTimes"`: exactly one of `times`/`frequencies`; `repeat` (loop
      `frequencies` past the end time) only with `frequencies`.
    - `"none"`: none of them.

    `start`/`end`/`begin` are written without prefix: a function object
    has one such window for both its controls.

    Args:
        interval: Period between executes/writes, in the unit of `mode`.
        times: Explicit run times, for `"runTimes"`.
        frequencies: `(time, frequency)` pairs, for `"runTimes"`.
        start: No execute/write before this time (`startTime`).
        end: No execute/write after this time (`endTime`).
        begin: Time the intervals are counted from (`beginTime`,
            OpenFOAM default: the case's begin time).

    Raises:
        ValueError: If `mode` is unsupported or the fields don't match it.
    """
    mode: TimeControlMode
    interval: Optional[float] = None
    times: Optional[List[float]] = None
    frequencies: Optional[List[Tuple[float, float]]] = None
    repeat: Optional[bool] = None
    start: Optional[float] = None
    end: Optional[float] = None
    begin: Optional[float] = None

    def __post_init__(self):
        self.mode = TimeControlMode.coerce(self.mode)

        if self.mode in _INTERVAL_REQUIRED and self.interval is None:
            raise ValueError(
                f"TimeControl({self.mode!r}, ...) requires interval"
            )
        if (
            self.mode not in _INTERVAL_REQUIRED
            and self.mode not in _INTERVAL_OPTIONAL
            and self.interval is not None
        ):
            raise ValueError(
                f"TimeControl({self.mode!r}, ...) doesn't use interval"
            )

        if self.mode == TimeControlMode.RUN_TIMES:
            if (self.times is None) == (self.frequencies is None):
                raise ValueError(
                    "TimeControl('runTimes', ...) requires exactly "
                    "one of times/frequencies"
                )
        else:
            if self.times is not None:
                raise ValueError(
                    f"TimeControl({self.mode!r}, ...) doesn't use times"
                )
            if self.frequencies is not None:
                raise ValueError(
                    f"TimeControl({self.mode!r}, ...) doesn't use "
                    "frequencies"
                )

        if self.repeat is not None and self.frequencies is None:
            raise ValueError(
                "TimeControl(..., repeat=...) requires frequencies"
            )

    def to_dict(self, prefix: str) -> dict:
        """Returns the entries under `prefix` (`"execute"` or `"write"`)."""
        result = {f"{prefix}Control": self.mode}
        if self.interval is not None:
            result[f"{prefix}Interval"] = self.interval
        if self.times is not None:
            result[f"{prefix}Times"] = self.times
        if self.frequencies is not None:
            result[f"{prefix}Frequencies"] = [
                list(pair) for pair in self.frequencies
            ]
        if self.repeat is not None:
            result[f"{prefix}Repeat"] = self.repeat
        if self.start is not None:
            result["startTime"] = self.start
        if self.begin is not None:
            result["beginTime"] = self.begin
        if self.end is not None:
            result["endTime"] = self.end
        return result


def periodic_times(values, period, end) -> list:
    """Repeats run times every `period` until `end` is covered.

    Args:
        values: Times or `(time, frequency)` pairs within one period, as
            `TimeControl`'s `times`/`frequencies` take them.
        period: `None` leaves `values` as they are.

    Returns:
        The sorted times or pairs.
    """
    pairs = bool(values) and not isinstance(values[0], Number)
    first = sorted(tuple(v) if pairs else v for v in values)
    if period is None or not first:
        return first
    result = list(first)
    shift = 0
    while (result[-1][0] if pairs else result[-1]) < end:
        shift += period
        result += [
            (v[0] + shift, v[1]) if pairs else v + shift for v in first
        ]
    return result


FUNCTION_KIND = IncludeEntryKind(
    directive="#includeFunc",
    template_root="caseDicts/functions",
    case_relative_dir="system",
    label="function object",
    file="functions",
    always_known=frozenset({
        "name", "funcName", "region",
        "executeControl", "executeInterval", "executeTimes",
        "executeFrequencies", "executeRepeat", "executeAtStart",
        "writeControl", "writeInterval", "writeTimes",
        "writeFrequencies", "writeRepeat",
        "startTime", "endTime", "beginTime",
    }),
    regional=False,
)


def merge_time_control(
    settings: Dict[str, object],
    execute_control: Optional["TimeControl"] = None,
    write_control: Optional["TimeControl"] = None,
    execute_at_start: Optional[bool] = None,
) -> Dict[str, object]:
    """Returns `settings` with the execute/write control keys merged in."""
    merged = dict(settings)
    if execute_control is not None:
        merged.update(execute_control.to_dict("execute"))
    if write_control is not None:
        merged.update(write_control.to_dict("write"))
    if execute_at_start is not None:
        merged["executeAtStart"] = execute_at_start
    return merged


class TemplateFunctionObject(TemplateEntry):
    """A not yet rendered function object from an OpenFOAM template."""
    kind = FUNCTION_KIND


def function_object_dict(
    type_name: str,
    libs,
    settings: Optional[dict] = None,
    *,
    execute_control: Optional[TimeControl] = None,
    write_control: Optional[TimeControl] = None,
    execute_at_start: Optional[bool] = None,
) -> dict:
    """Builds a function-object dict directly, without template.

    Args:
        libs: Shared libraries it needs; none writes no `libs`.
        settings: Type-specific settings, merged as-is.
        execute_control: Merged via `.to_dict("execute")`.
        write_control: Merged via `.to_dict("write")`.
        execute_at_start: Sets `executeAtStart`.
    """
    return merge_time_control(
        entry_dict(type_name, libs, settings),
        execute_control, write_control, execute_at_start,
    )


# surfaceFieldValue's patch/faceZone/triSurface selections have
# #includeFunc templates (etc/caseDicts/functions/surfaceFieldValue/);
# sampledSurface has none, hence these two helpers.

def cut_plane(
    point=None, normal=None, *, points=None,
    interpolate: Optional[bool] = None, cell_zone: Optional[str] = None,
) -> dict:
    """A `sampledSurface` cutPlane sub-dict for `surfaceFieldValue`.

    The plane is given by `point` and `normal` (`planeType
    pointAndNormal`) or by `points` (`embeddedPoints`); build the dict by
    hand for `planeEquation`.

    Args:
        point: A point in the plane [m], `(x, y, z)`.
        normal: The plane normal, `(x, y, z)`.
        points: Three points in the plane [m].
        interpolate: Interpolate to the surface points (OpenFOAM default:
            no).
        cell_zone: Restrict to this cell zone (OpenFOAM default: `all`).

    Raises:
        ValueError: Unless exactly one of the two forms is given.
    """
    if (points is None) == (point is None) or (point is None) != (
        normal is None
    ):
        raise ValueError("cut_plane() takes point and normal, or points")
    result = {"type": "cutPlane"}
    if points is None:
        result.update({
            "planeType": "pointAndNormal",
            "point": tuple(point), "normal": tuple(normal),
        })
    else:
        point1, point2, point3 = points
        result.update({
            "planeType": "embeddedPoints", "point1": tuple(point1),
            "point2": tuple(point2), "point3": tuple(point3),
        })
    if interpolate is not None:
        result["interpolate"] = interpolate
    if cell_zone is not None:
        result["cellZone"] = cell_zone
    return result


def iso_surface(
    iso_field: str, iso_value: float, *, interpolate: Optional[bool] = None,
    cell_zone: Optional[str] = None,
) -> dict:
    """A `sampledSurface` isoSurface sub-dict, see `cut_plane()`.

    Args:
        iso_field: The field to take the iso-surface of.
        iso_value: The iso-surface value.
        interpolate: Interpolate to the surface points (OpenFOAM default:
            no).
        cell_zone: Restrict to this cell zone (OpenFOAM default: `all`).
    """
    result = {
        "type": "isoSurface", "isoField": iso_field, "isoValue": iso_value,
    }
    if interpolate is not None:
        result["interpolate"] = interpolate
    if cell_zone is not None:
        result["cellZone"] = cell_zone
    return result


def patch_surface(
    patches, *, triangulate: Optional[bool] = None,
    interpolate: Optional[bool] = None,
) -> dict:
    """A `sampledSurface` sub-dict of boundary patches, see `cut_plane()`.

    Args:
        patches: Patch names or groups.
        triangulate: Split the faces into triangles (OpenFOAM default: no).
    """
    result = {"type": "patch", "patches": list(patches)}
    if triangulate is not None:
        result["triangulate"] = triangulate
    if interpolate is not None:
        result["interpolate"] = interpolate
    return result


# sampledSet sub-dicts, for `sets()` and for `streamlines()`' seeds; no
# `#includeFunc` template takes one.

def line_set(
    start, end, *, axis: Optional[str] = None, cell: bool = False,
    face: bool = False, n_points: Optional[int] = None,
) -> dict:
    """A `sampledSet` along a line.

    `n_points` gives `lineUniform`; else `cell`/`face` give `lineCell`,
    `lineFace` or `lineCellFace`.

    Args:
        start: Start point [m].
        end: End point [m].
        axis: How positions are written: `"distance"`, `"x"`, `"y"`,
            `"z"` or `"xyz"`.
        cell: Sample in the cells the line crosses.
        face: Sample at the faces the line crosses.

    Raises:
        ValueError: If none of `n_points`, `cell`, `face` is given.
    """
    if n_points is None and not (cell or face):
        raise ValueError("line_set() needs n_points, cell or face")
    kind = "Uniform" if n_points is not None else (
        ("Cell" if cell else "") + ("Face" if face else "")
    )
    result = {"type": f"line{kind}"}
    if axis is not None:
        result["axis"] = axis
    result.update({"start": tuple(start), "end": tuple(end)})
    if n_points is not None:
        result["nPoints"] = n_points
    return result


def arc_uniform_set(
    centre, normal, radial, start_angle, end_angle, n_points: int,
) -> dict:
    """A `sampledSet` of evenly spaced points on a circular arc.

    Args:
        radial: Vector from `centre` to the point at angle 0.
        start_angle: [deg].
        end_angle: [deg].
    """
    return {
        "type": "arcUniform", "centre": tuple(centre),
        "normal": tuple(normal), "radial": tuple(radial),
        "startAngle": start_angle, "endAngle": end_angle,
        "nPoints": n_points,
    }


def box_uniform_set(box_min, box_max, n_points) -> dict:
    """A `sampledSet` on a regular grid in a box.

    Args:
        n_points: Number of points in x, y and z.
    """
    return {
        "type": "boxUniform",
        "box": Compound([tuple(box_min), tuple(box_max)]),
        "nPoints": tuple(n_points),
    }


def circle_random_set(centre, normal, radius, n_points: int) -> dict:
    """A `sampledSet` of random points in a circle."""
    return {
        "type": "circleRandom", "centre": tuple(centre),
        "normal": tuple(normal), "radius": radius, "nPoints": n_points,
    }


def sphere_random_set(centre, radius, n_points: int) -> dict:
    """A `sampledSet` of random points in a sphere."""
    return {
        "type": "sphereRandom", "centre": tuple(centre), "radius": radius,
        "nPoints": n_points,
    }


def boundary_random_set(patches, n_points: int) -> dict:
    """A `sampledSet` of random points on boundary patches."""
    return {
        "type": "boundaryRandom", "patches": list(patches),
        "nPoints": n_points,
    }


def points_set(points, *, ordered: bool = False) -> dict:
    """A `sampledSet` of given points.

    Args:
        ordered: The points form a path, written with its distance.
    """
    return {
        "type": "points", "ordered": ordered,
        "points": [tuple(point) for point in points],
    }


# Function objects that either have no `#includeFunc` template or whose
# template fixes what a caller needs to choose; each returns the dict
# `function_object_dict()` builds and passes `**control` on to it.

_UTILITY = (String("libutilityFunctionObjects.so"),)
_FIELD = (String("libfieldFunctionObjects.so"),)
_SAMPLING = (String("libsampling.so"),)
_LAGRANGIAN = (String("liblagrangianFunctionObjects.so"),)


def surfaces(
    surfaces: dict, fields, *, surface_format: str = "vtk",
    write_format: Optional[str] = None,
    interpolation_scheme: str = "cellPoint",
    write_empty: Optional[bool] = None, **control,
) -> dict:
    """A `surfaces` function object: fields sampled on surfaces.

    Args:
        surfaces: `sampledSurface` sub-dicts by name, e.g. `cut_plane()`.
        write_format: `"ascii"` or `"binary"` (OpenFOAM default: the
            case's).
        write_empty: Also write surfaces without faces (OpenFOAM default:
            no).
    """
    settings = {"surfaceFormat": surface_format}
    if write_format is not None:
        settings["writeFormat"] = write_format
    settings.update({
        "fields": list(fields), "interpolationScheme": interpolation_scheme,
    })
    if write_empty is not None:
        settings["writeEmpty"] = write_empty
    settings["surfaces"] = surfaces
    return function_object_dict("surfaces", _SAMPLING, settings, **control)


def sets(
    sets: dict, fields, *, set_format: str = "raw",
    interpolation_scheme: str = "cellPoint", **control,
) -> dict:
    """A `sets` function object: fields sampled on point sets.

    Args:
        sets: `sampledSet` sub-dicts by name, e.g. `line_set()`.
    """
    return function_object_dict("sets", _SAMPLING, {
        "setFormat": set_format, "fields": list(fields),
        "interpolationScheme": interpolation_scheme, "sets": sets,
    }, **control)


def streamlines(
    seed: dict, fields, *, direction: str = "forward",
    life_time: int = 10000, n_sub_cycle: int = 5,
    set_format: str = "vtk", cloud_name: Optional[str] = None, **control,
) -> dict:
    """A `streamlines` function object.

    Args:
        seed: `sampledSet` sub-dict of the start points.
        fields: Fields sampled along the lines.
        direction: `"forward"`, `"backward"` or `"both"`.
        life_time: Maximum number of steps of a line.
        n_sub_cycle: Steps per cell.
        cloud_name: Name of the tracking cloud (OpenFOAM default: the
            function object's name).
    """
    settings = {
        "fields": list(fields), "lifeTime": life_time,
        "nSubCycle": n_sub_cycle, "direction": direction,
        "setFormat": set_format, "writeTime": True,
    }
    if cloud_name is not None:
        settings["cloudName"] = cloud_name
    settings["seedSampleSet"] = seed
    return function_object_dict("streamlines", _FIELD, settings, **control)


def write_objects(
    objects, *, write_option: str = "autoWrite", **control,
) -> dict:
    """A `writeObjects` function object.

    Args:
        objects: Names or regular expressions of registered objects.
        write_option: `"autoWrite"`, `"noWrite"` or `"anyWrite"`.
    """
    return function_object_dict("writeObjects", _UTILITY, {
        "objects": list(objects), "writeOption": write_option,
    }, **control)


_SOLVER = (String("libsolverFunctionObjects.so"),)


def derived_field(
    type_name: str, *, field: Optional[str] = None,
    result: Optional[str] = None, phase: Optional[str] = None,
    velocity: Optional[str] = None, flux: Optional[str] = None,
    **control,
) -> dict:
    """A field function object without settings of its own, e.g.
    `CourantNo`, `MachNo`, `vorticity`, `Q`, `mag`.

    Args:
        field: The field it reads, for those that take one (`mag`).
        result: Name of the field it stores.
        phase: Phase whose thermo and fields it reads.
        velocity: Name of the velocity field (`U`).
        flux: Name of the flux field (`phi`).
    """
    settings = {}
    for key, value in (
        ("field", field), ("result", result), ("phase", phase),
        ("U", velocity), ("phi", flux),
    ):
        if value is not None:
            settings[key] = value
    return function_object_dict(type_name, _FIELD, settings, **control)


def divide(fields, result: str, **control) -> dict:
    """A `divide` function object: the first field by the others."""
    return function_object_dict("divide", _FIELD, {
        "fields": list(fields), "result": result,
    }, **control)


def turbulence_fields(fields, **control) -> dict:
    """A `turbulenceFields` function object.

    Args:
        fields: Of `k`, `epsilon`, `omega`, `nut`, `nuEff`, `kappaEff`, ...
    """
    return function_object_dict("turbulenceFields", _FIELD, {
        "fields": list(fields),
    }, **control)


def scalar_transport(
    field: str, *, schemes_field: Optional[str] = None,
    solver_field: Optional[str] = None, diffusivity: str = "viscosity",
    alphal=1, alphat=1, **control,
) -> dict:
    """A `scalarTransport` function object: a passive scalar.

    Args:
        schemes_field: Field whose `fvSchemes` entries it uses.
        solver_field: Field whose `fvSolution` entry it uses.
        diffusivity: `"none"`, `"constant"` or `"viscosity"`.
        alphal: Laminar and
        alphat: turbulent viscosity factor of the diffusivity.
    """
    settings = {
        "field": field, "diffusivity": diffusivity, "alphal": alphal,
        "alphat": alphat,
    }
    if schemes_field is not None:
        settings["schemesField"] = schemes_field
    if solver_field is not None:
        settings["solverField"] = solver_field
    return function_object_dict("scalarTransport", _SOLVER, settings,
                                **control)


def fluid_max_delta_t(max_co, max_delta_t, **control) -> dict:
    """A `fluidMaxDeltaT` function object.

    Args:
        max_co: Maximum Courant number, a number or `Function1` of time.
        max_delta_t: Maximum time step, likewise.
    """
    return function_object_dict("fluidMaxDeltaT", None, {
        "maxCo": max_co, "maxDeltaT": max_delta_t,
    }, **control)


def auto_write_field(name: str, **control) -> dict:
    """A `coded` function object making the registered scalar field
    `name` written at every write time."""
    return function_object_dict("coded", _UTILITY, {
        "codeRead": Code(
            f' if (mesh().foundObject<volScalarField>("{name}")) {{'
            f' volScalarField& field ='
            f' mesh().lookupObjectRef<volScalarField>("{name}");'
            " field.writeOpt() = IOobject::AUTO_WRITE; } "
        ),
        "codeFields": Code(f' fields.append("{name}"); '),
    }, **control)


def read_field(name: str, **control) -> dict:
    """A `coded` function object reading the scalar field `name` from the
    time directory and keeping it registered and written."""
    return function_object_dict("coded", _UTILITY, {
        "codeRead": Code(
            f" volScalarField* {name} = new volScalarField ( IOobject"
            f' ( "{name}", mesh().time().name(), mesh(),'
            " IOobject::MUST_READ, IOobject::AUTO_WRITE ), mesh() );"
            f" mesh().store({name}); "
        ),
    }, **control)


def cloud_info(clouds, **control) -> dict:
    """A `cloudInfo` function object: parcel number and mass of the named
    clouds."""
    return function_object_dict("cloudInfo", _LAGRANGIAN, {
        "clouds": list(clouds),
    }, **control)


def write_now(**control) -> dict:
    """A `coded` function object that writes the time it executes at.

    With `execute_control=TimeControl("runTimes", times=...)` it adds
    write times to those of `controlDict`.
    """
    return function_object_dict("coded", _UTILITY, {
        "codeExecute": Code(
            " const_cast<Time&>(mesh().time()).writeNow(); "
        ),
    }, **control)


def set_write_interval(write_interval, **control) -> dict:
    """A `setWriteInterval` function object.

    Args:
        write_interval: `Function1` of the write interval over time.
    """
    return function_object_dict("setWriteInterval", _UTILITY, {
        "writeInterval": write_interval,
    }, **control)


def time_step(*, user_time: bool = False, **control) -> dict:
    """A `timeStep` function object, logging the time step.

    Args:
        user_time: Log it in user time (`userTimeStep`).
    """
    return function_object_dict(
        "userTimeStep" if user_time else "timeStep", _UTILITY, **control,
    )


def run_time(
    *, per_time_step: bool = True, forecast: bool = True, **control,
) -> dict:
    """A `time` function object, logging CPU and clock time.

    Args:
        per_time_step: Also log the times per time step.
        forecast: Also log the estimated remaining run time.
    """
    return function_object_dict("time", _UTILITY, {
        "perTimeStep": per_time_step, "forecast": forecast,
    }, **control)


def stop_at_time_step(
    min_delta_t, *, action: str = "writeNow", **control,
) -> dict:
    """A `stopAtTimeStep` function object.

    Args:
        min_delta_t: The run stops when the time step falls below it [s].
        action: `"noWriteNow"`, `"writeNow"` or `"nextWrite"`.
    """
    return function_object_dict("stopAtTimeStep", _UTILITY, {
        "minDeltaT": min_delta_t, "action": action,
    }, **control)


def check_mesh(
    *, write_sets: bool = False, write_surfaces: bool = False,
    all_topology: bool = False, all_geometry: bool = False,
    set_format: str = "vtk", **control,
) -> dict:
    """A `checkMesh` function object.

    Args:
        write_sets: Write the sets of failing cells, faces and points.
        write_surfaces: Write the failing faces as surfaces.
    """
    return function_object_dict("checkMesh", _UTILITY, {
        "writeSets": write_sets, "writeSurfaces": write_surfaces,
        "allTopology": all_topology, "allGeometry": all_geometry,
        "setFormat": set_format,
    }, **control)


def wall_field(
    type_name: str, *, patches=None, write_field: bool = True,
    phase: Optional[str] = None, **control,
) -> dict:
    """A `wallHeatFlux`, `yPlus` or `wallShearStress` function object.

    Args:
        patches: Wall patches (OpenFOAM default: all walls).
        write_field: `False` only logs and writes the per-patch minimum,
            maximum and integral.
    """
    settings = {}
    if patches:
        settings["patches"] = list(patches)
    if not write_field:
        settings["objects"] = []
    if phase is not None:
        settings["phase"] = phase
    return function_object_dict(type_name, _FIELD, settings, **control)


def surface_interpolate(field: str, result: str, **control) -> dict:
    """A `surfaceInterpolate` function object.

    Args:
        result: Name of the face field; the default `<field>f` would
            collide with the mesh's `Cf`/`Sf` for species `C` and `S`.
    """
    return function_object_dict("surfaceInterpolate", _FIELD, {
        "field": field, "result": result,
    }, **control)


def _field_value(
    type_name, fields, operation, selection, weight_fields, log,
    write_fields, control,
):
    settings = {
        "log": log, "writeFields": write_fields, "operation": operation,
        **selection, "fields": list(fields),
    }
    if weight_fields:
        settings["weightFields"] = list(weight_fields)
    return function_object_dict(type_name, _FIELD, settings, **control)


def vol_field_value(
    fields, operation, *, cell_zone: str = "all", weight_fields=(),
    write_location: Optional[bool] = None, log: bool = False,
    write_fields: bool = False, **control,
) -> dict:
    """A `volFieldValue` function object: one value per field of a zone.

    Args:
        operation: A `VolFieldValueOperation`.
        weight_fields: Fields whose product weights the operation.
        write_location: Also write where a minimum or maximum is.
        write_fields: Also write the fields' values in the zone.
    """
    selection = {"cellZone": cell_zone}
    if write_location is not None:
        selection["writeLocation"] = write_location
    return _field_value(
        "volFieldValue", fields, VolFieldValueOperation.coerce(operation),
        selection, weight_fields, log, write_fields, control,
    )


def surface_field_value(
    fields, operation, *, face_zone: Optional[str] = None,
    patch: Optional[str] = None, sampled_surface: Optional[dict] = None,
    weight_fields=(), log: bool = False, write_fields: bool = False,
    **control,
) -> dict:
    """A `surfaceFieldValue` function object, see `vol_field_value()`.

    A face zone takes face fields, see `surface_interpolate()`.

    Args:
        operation: A `SurfaceFieldValueOperation`.
        sampled_surface: A `sampledSurface` sub-dict, e.g. `cut_plane()`.

    Raises:
        ValueError: Unless exactly one of `face_zone`, `patch`,
            `sampled_surface` is given.
    """
    selection = {
        key: value for key, value in (
            ("faceZone", face_zone), ("patch", patch),
            ("sampledSurface", sampled_surface),
        ) if value is not None
    }
    if len(selection) != 1:
        raise ValueError(
            "surface_field_value() takes exactly one of face_zone, patch, "
            "sampled_surface"
        )
    return _field_value(
        "surfaceFieldValue", fields,
        SurfaceFieldValueOperation.coerce(operation), selection,
        weight_fields, log, write_fields, control,
    )
