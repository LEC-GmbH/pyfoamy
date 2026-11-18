"""High-level container for an OpenFOAM polyMesh."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from pyfoamy.parsing.grammar import (
    AnonymousListEntry, Dictionary, KeywordEntry,
)
from pyfoamy.parsing.listfile import (
    CompactLabelListList,
    _read_bytes,
    read_face_list,
    read_header,
    read_label_list,
    read_vector_list,
)
from pyfoamy.parsing.timedirs import find_instance

_N_CELLS = re.compile(r"nCells:\s*(\d+)")


@dataclass(frozen=True)
class MeshPatch:
    """One `boundary` entry."""

    type: str
    start_face: int
    n_faces: int
    groups: Tuple[str, ...] = ()

    def faces(self) -> slice:
        """The patch's faces in the mesh's face numbering."""
        return slice(self.start_face, self.start_face + self.n_faces)


@dataclass
class PolyMesh:
    """The arrays of a `polyMesh` directory.

    Args:
        points: `(n_points, 3)` coordinates.
        faces: Point labels per face.
        owner: Owner cell per face.
        neighbour: Neighbour cell per internal face.
        patches: Patch name -> `MeshPatch`, in file order.
        n_cells: Number of cells.
    """

    points: np.ndarray
    faces: CompactLabelListList
    owner: np.ndarray
    neighbour: np.ndarray
    patches: Dict[str, MeshPatch]
    n_cells: int

    @classmethod
    def read(cls, directory) -> "PolyMesh":
        """Reads a `polyMesh` directory."""
        directory = Path(directory)
        return cls._read(directory / "points", directory)

    @classmethod
    def read_at(cls, case_dir, time_value, region=None) -> "PolyMesh":
        """Reads a case's mesh as OpenFOAM resolves it at a time.

        `points` and `faces` are searched backward independently;
        `owner`/`neighbour`/`boundary` come from where `faces` was
        found, as in OpenFOAM's `polyMesh` constructor.
        """
        points = find_instance(case_dir, time_value, "points", region=region)
        faces = find_instance(case_dir, time_value, "faces", region=region)
        return cls._read(points, faces.parent)

    @classmethod
    def _read(cls, points_path: Path, topology_dir: Path) -> "PolyMesh":
        owner = read_label_list(topology_dir / "owner")
        neighbour = read_label_list(topology_dir / "neighbour")
        return cls(
            points=read_vector_list(points_path),
            faces=read_face_list(topology_dir / "faces"),
            owner=owner,
            neighbour=neighbour,
            patches=read_boundary(topology_dir / "boundary"),
            n_cells=_n_cells(topology_dir / "owner", owner, neighbour),
        )

    @property
    def n_internal_faces(self) -> int:
        """Number of internal faces."""
        return len(self.neighbour)

    def patch_points(self, name: str) -> np.ndarray:
        """A patch's points, in OpenFOAM's order (first use by its faces)."""
        patch = self.patches[name]
        offsets = self.faces.offsets
        labels = self.faces.values[
            offsets[patch.start_face]:offsets[patch.start_face + patch.n_faces]
        ]
        _, first = np.unique(labels, return_index=True)
        return labels[np.sort(first)]


def read_boundary(path) -> Dict[str, MeshPatch]:
    """Reads a `boundary` file: patch name -> `MeshPatch`."""
    boundary = Dictionary(_read_bytes(path).decode("utf-8"))
    entries = next(n for n in boundary if isinstance(n, AnonymousListEntry))
    patches = {}
    for entry in entries.body:
        if not isinstance(entry, KeywordEntry):
            continue
        patch = entry.value
        groups = patch["inGroups"] if "inGroups" in patch else ()
        patches[str(entry.key)] = MeshPatch(
            type=str(patch["type"]),
            start_face=int(patch["startFace"]),
            n_faces=int(patch["nFaces"]),
            groups=tuple(str(group) for group in groups),
        )
    return patches


def _n_cells(owner_path, owner, neighbour) -> int:
    header, _, _ = read_header(_read_bytes(owner_path))
    foam_file = header["FoamFile"]
    if "note" in foam_file:
        match = _N_CELLS.search(str(foam_file["note"]))
        if match:
            return int(match.group(1))
    if not len(owner):
        return 0
    return int(max(owner.max(), neighbour.max(initial=-1))) + 1
