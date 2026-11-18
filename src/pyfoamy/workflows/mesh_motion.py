"""Mesh motion without a flow solution."""

from typing import Sequence

from pyfoamy import function_objects as fo
from pyfoamy.runner import Runner

_CONTROL = "system/controlDict"
_FUNCTIONS = "system/functions"


def move_mesh(
    runner: Runner, start, end, delta_t, *, max_delta_t=None,
    write_interval=99999, write_times: Sequence = (),
    write_objects: Sequence[str] = (), check_mesh: bool = True,
    fatal_sets: Sequence[str] = (), function_objects: bool = True,
    suffix: str = "",
) -> bool:
    """Moves the mesh from `start` to `end` with the `movingMesh` solver.

    Edits `system/controlDict` and `system/functions` of the decomposed
    case on disk (the time entries stay as set, the added function
    objects are removed again) and runs `foamRun` in parallel.

    Args:
        delta_t: Time step.
        max_delta_t: Default `delta_t`.
        write_interval: Of the `controlDict`; the default writes nothing.
        write_times: Times at which the mesh is written and, with
            `check_mesh`, checked with every test and its failing faces
            written as surfaces.
        write_objects: Names or regular expressions of registered
            objects written at `write_times` beside the points.
        check_mesh: Run `checkMesh` every time step.
        fatal_sets: Names of `checkMesh` sets (`"zeroVolumeCells"`); the
            mesh is invalid if one is written at a write time.
        function_objects: `False` runs with `-noFunctionObjects`.
        suffix: Of the logs.

    Returns:
        `False` if a fatal set was written; always `True` when recorded.
    """
    for entry, value in (
        ("startTime", start), ("endTime", end), ("deltaT", delta_t),
        ("maxDeltaT", delta_t if max_delta_t is None else max_delta_t),
        ("writeInterval", write_interval),
    ):
        runner.set_entry(_CONTROL, entry, value)

    at_write_times = fo.TimeControl("runTimes", times=list(write_times))
    functions = {}
    if write_times:
        functions["writeMovedMesh"] = fo.write_objects(
            ["points", *write_objects], write_option="anyWrite",
            write_control=at_write_times,
        )
    if check_mesh:
        functions["checkMesh"] = fo.check_mesh()
        if write_times:
            functions["checkMeshWrite"] = fo.check_mesh(
                write_surfaces=True, all_topology=True, all_geometry=True,
                execute_control=at_write_times,
            )
    for name, function in functions.items():
        runner.set_entry(_FUNCTIONS, name, function)

    checked = [
        f"postProcessing/checkMesh/{time}" for time in write_times
    ] if check_mesh else []
    for directory in checked:
        runner.run("rm", args=["-rvf", directory], suffix=suffix)

    args = ["-solver", "movingMesh"]
    if not function_objects:
        args.append("-noFunctionObjects")
    runner.run("foamRun", args=args, parallel=True, suffix=suffix)

    for name in functions:
        runner.remove_entry(_FUNCTIONS, name)

    if runner.mode == "record":
        return True
    written = {
        path.stem
        for directory in checked if (runner.root / directory).is_dir()
        for path in (runner.root / directory).iterdir()
    }
    return not written & set(fatal_sets)
