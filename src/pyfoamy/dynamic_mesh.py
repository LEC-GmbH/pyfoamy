"""Builds entries of a case's `dynamicMeshDict` (point mesh movers of
OpenFOAM-dev)."""

from typing import Mapping, Optional, Sequence

from pyfoamy.parsing.grammar import String

_MOTION_SOLVERS = (String("libfvMotionSolvers.so"),)
_MESH_TO_MESH = (String("libmeshToMeshTopoChanger.so"),)


def rotating_motion(origin, axis, *, omega=None, rpm=None) -> dict:
    """A `rotatingMotion` solid-body motion function.

    Args:
        origin: A point on the axis.
        axis: Direction of the axis.
        omega: Angular speed in rad/s, a number or a Function1.
        rpm: The speed in revolutions per minute instead.

    Raises:
        ValueError: Unless exactly one of `omega` and `rpm` is given.
    """
    if (omega is None) == (rpm is None):
        raise ValueError("rotating_motion() takes either omega or rpm")
    speed = {"omega": omega} if rpm is None else {"rpm": rpm}
    return {"type": "rotatingMotion", "axis": axis, "origin": origin, **speed}


def solid_body(cell_zone: str, motion: Mapping) -> dict:
    """A `solidBody` mover: the cell zone follows the motion function."""
    return {
        "type": "solidBody", "cellZone": cell_zone,
        "solidBodyMotionFunction": dict(motion),
    }


def mover_list(movers: Mapping[str, Mapping]) -> dict:
    """A `list` mover: the named movers' displacements are added up.

    Raises:
        ValueError: If there is no mover.
    """
    if not movers:
        raise ValueError("mover_list() needs at least one mover")
    return {"type": "list", "libs": _MOTION_SOLVERS, **movers}


def mesh_to_mesh(
    times: Sequence, *, time_delta, begin=None,
    repeat: Optional[float] = None,
) -> dict:
    """A `meshToMesh` topology changer: at every time the solution is
    mapped onto the mesh stored for it in `constant/meshes`.

    Args:
        times: The mapping times, rising.
        time_delta: Tolerance in matching a time.
        begin: Start of the period, OpenFOAM's default is the case's
            begin time.
        repeat: Period after which the times repeat.

    Raises:
        ValueError: If there is no time or they do not rise.
    """
    times = list(times)
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"mesh_to_mesh() needs rising times: {times}")
    entry = {"type": "meshToMesh", "libs": _MESH_TO_MESH, "times": times}
    if begin is not None:
        entry["begin"] = begin
    entry["timeDelta"] = time_delta
    if repeat is not None:
        entry["repeat"] = repeat
    return entry
