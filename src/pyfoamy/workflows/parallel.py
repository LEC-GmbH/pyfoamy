"""Decomposition, reconstruction and time directories of a case."""

import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from pyfoamy.parsing.timedirs import resolve_time_instance
from pyfoamy.runner import Runner

_MESH_SIZE = re.compile(r"n(Points|Cells|Faces):\s*(\d+)")


def time_selection(times: Sequence = (), start=None, stop=None) -> str:
    """The `-time` argument selecting `times` and the range `start:stop`.

    Returns:
        The selection; one that matches no time if nothing is given.
    """
    # "-time N" alone selects the time nearest to N, "N:N" only N itself.
    if len(times) == 1:
        parts = [f"{times[0]}:{times[0]}"]
    else:
        parts = [str(time) for time in times]
    if start is not None or stop is not None:
        parts.append(
            f"{'' if start is None else start}:"
            f"{'' if stop is None else stop}"
        )
    return ",".join(parts) or "1:0"


def remove_times(
    runner: Runner, times: Sequence = (), start=None, stop=None, *,
    constant: bool = False, processor: bool = False, suffix: str = "",
) -> None:
    """Removes time directories with `foamListTimes -rm`.

    Args:
        constant: Also `constant`'s mesh.
        processor: In the processor directories instead of the case's.
    """
    args = ["-rm", "-time", time_selection(times, start, stop)]
    if processor:
        args.append("-processor")
    if constant:
        args.append("-constant")
    runner.run("foamListTimes", args=args, suffix=suffix)


def _transfer(
    runner, application, times, start, stop, fields, constant, extra,
    remove, processor, suffix,
):
    if remove:
        remove_times(
            runner, times, start, stop, constant=constant,
            processor=processor,
            suffix=f"{application}.{suffix}" if suffix else application,
        )
    selected = bool(times) or start is not None or stop is not None
    if not selected and not constant:
        return
    args = []
    if selected:
        args += ["-time", time_selection(times, start, stop)]
    if constant:
        args.append("-constant")
    if fields is False:
        args.append("-noFields")
    elif fields is not True:
        args += ["-fields", f"({' '.join(fields)})"]
    runner.run(application, args=args + extra, suffix=suffix)


def decompose(
    runner: Runner, times: Sequence = (), start=None, stop=None, *,
    fields: bool = True, constant: bool = False, force: bool = False,
    remove: bool = False, suffix: str = "",
) -> None:
    """Decomposes the selected times with `decomposePar`.

    Nothing runs if neither a time nor `constant` is selected.

    Args:
        fields: `False` decomposes the mesh only.
        constant: Include `constant`.
        force: Replace existing processor directories.
        remove: First remove the selected times from the processor
            directories.
    """
    _transfer(
        runner, "decomposePar", times, start, stop, fields, constant,
        ["-force"] if force else [], remove, True, suffix,
    )


def reconstruct(
    runner: Runner, times: Sequence = (), start=None, stop=None, *,
    fields: Union[bool, Sequence[str]] = True, constant: bool = False,
    remove: bool = False, suffix: str = "",
) -> None:
    """Reconstructs the selected times with `reconstructPar`.

    Nothing runs if neither a time nor `constant` is selected.

    Args:
        fields: `False` reconstructs the mesh only; names restrict the
            fields.
        constant: Include `constant`.
        remove: First remove the selected times from the case.
    """
    _transfer(
        runner, "reconstructPar", times, start, stop, fields, constant,
        [], remove, False, suffix,
    )


def mesh_statistics(
    root, time_value=None, region: Optional[str] = None,
) -> Dict[str, int]:
    """Numbers of points, cells and faces of a mesh, from its header.

    Reads the `note` of `polyMesh/owner`, not the mesh.

    Args:
        time_value: Time of the mesh; `constant` if `None`.

    Returns:
        `{"points": ..., "cells": ..., "faces": ...}`.

    Raises:
        FileNotFoundError: If there is no such time or mesh.
        ValueError: If the header carries no sizes.
    """
    root = Path(root)
    instance = (
        "constant" if time_value is None
        else resolve_time_instance(root, time_value)
    )
    owner = root / instance / (region or "") / "polyMesh" / "owner"
    with open(owner, errors="replace") as f:
        header = f.read(2000)
    sizes = {
        name.lower(): int(count)
        for name, count in _MESH_SIZE.findall(header)
    }
    if set(sizes) != {"points", "cells", "faces"}:
        raise ValueError(f"no mesh sizes in the header of {owner}")
    return sizes
