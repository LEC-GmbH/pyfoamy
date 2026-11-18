"""Reads a decomposed case's meshes and fields as the complete case."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from pyfoamy.parsing.field import FieldValues, PatchValues, _internal_size
from pyfoamy.parsing.grammar import DimensionSet
from pyfoamy.parsing.listfile import read_label_list
from pyfoamy.parsing.polymesh import PolyMesh
from pyfoamy.parsing.timedirs import (
    find_instance,
    processor_directories,
    resolve_time_instance,
)

# dimVolumetricFlux and dimMassFlux: OpenFOAM's isFlux() for surfaceScalarFields.
_FLUX_DIMENSIONS = {(0, 3, -1, 0, 0, 0, 0), (1, 0, -1, 0, 0, 0, 0)}


@dataclass
class _Processor:
    directory: Path
    mesh: PolyMesh
    cells: np.ndarray
    faces: np.ndarray
    points: np.ndarray


def _instance(case_dir: Path, path: Path) -> str:
    return path.relative_to(case_dir).parts[0]


class DecomposedCase:
    """The `processorN` directories of a case, read as one complete case.

    Topology comes from the complete case's mesh at the same instance;
    moved points and fields are assembled from the processors through
    their `*ProcAddressing`.

    Args:
        root: Path to the case directory.
        region: Mesh region, or None.
    """

    def __init__(self, root, region: Optional[str] = None):
        self.root = Path(root)
        self.region = region
        self._key: Optional[Tuple] = None
        self._complete: Optional[PolyMesh] = None
        self._processors: List[_Processor] = []

    def _load(self, time_value) -> None:
        directories = processor_directories(self.root)
        if not directories:
            raise FileNotFoundError(f"no processor directories in {self.root}")
        faces = find_instance(
            self.root, time_value, "faces", region=self.region
        )
        complete = _instance(self.root, faces)
        key = [complete]
        for directory in directories:
            # A solver may rewrite unchanged topology without addressing.
            for name in ("faces", "points", "cellProcAddressing"):
                path = find_instance(
                    directory, time_value, name, region=self.region
                )
                key.append(_instance(directory, path))
            if key[-1] != complete:
                raise ValueError(
                    f"{directory} has its addressing at {key[-1]}, the "
                    f"complete mesh its topology at {complete}"
                )
        if tuple(key) == self._key:
            return
        self._complete = PolyMesh.read_at(
            self.root, time_value, region=self.region
        )
        self._processors = []
        for directory in directories:
            addressing = find_instance(
                directory, time_value, "cellProcAddressing",
                region=self.region,
            ).parent
            processor = _Processor(
                directory,
                PolyMesh.read_at(directory, time_value, region=self.region),
                read_label_list(addressing / "cellProcAddressing"),
                read_label_list(addressing / "faceProcAddressing"),
                read_label_list(addressing / "pointProcAddressing"),
            )
            mesh = processor.mesh
            if (len(processor.cells), len(processor.faces),
                    len(processor.points)) != (
                    mesh.n_cells, len(mesh.faces), len(mesh.points)):
                raise ValueError(
                    f"{directory}: mesh at {time_value} doesn't match its "
                    f"addressing at {_instance(directory, addressing)}"
                )
            self._processors.append(processor)
        self._key = tuple(key)

    def read_mesh_at(self, time_value) -> PolyMesh:
        """The complete mesh at a time, with the processors' points."""
        self._load(time_value)
        complete = self._complete
        points = np.empty_like(complete.points)
        for processor in self._processors:
            points[processor.points] = processor.mesh.points
        return PolyMesh(
            points, complete.faces, complete.owner, complete.neighbour,
            complete.patches, complete.n_cells,
        )

    def read_field_at(self, time_value, field_name: str) -> FieldValues:
        """The complete field at a time.

        Patch values are None where a processor has none, and on
        non-conformal patches, whose faces the polyMesh doesn't hold.
        A face on a `processorCyclic` patch is left NaN.
        """
        self._load(time_value)
        complete = self._complete
        parts = []
        for processor in self._processors:
            instance = resolve_time_instance(processor.directory, time_value)
            directory = processor.directory / instance
            if self.region is not None:
                directory = directory / self.region
            parts.append(
                FieldValues.read(directory / field_name, processor.mesh)
            )
        first = parts[0]
        tail = first.internal.shape[1:]
        internal = np.full(
            (_internal_size(complete, first.kind),) + tail, np.nan
        )
        flux = (
            first.kind == "surface" and not tail
            and isinstance(first.dimensions, DimensionSet)
            and first.dimensions.as_tuple() in _FLUX_DIMENSIONS
        )
        for processor, part in zip(self._processors, parts):
            mesh = processor.mesh
            if part.kind == "vol":
                internal[processor.cells] = part.internal
            elif part.kind == "point":
                internal[processor.points] = part.internal
            else:
                n = mesh.n_internal_faces
                _set_faces(internal, processor.faces[:n], part.internal, flux)
                for name, patch in mesh.patches.items():
                    values = getattr(part.patches.get(name), "value", None)
                    if patch.type == "processor" and values is not None:
                        _set_faces(
                            internal, processor.faces[patch.faces()],
                            values, flux,
                        )
        patches = {}
        for name, patch in complete.patches.items():
            if name not in first.patches:
                continue
            value = None
            if not patch.type.startswith("nonConformal") and all(
                part.patches[name].value is not None for part in parts
            ):
                value = self._patch_array(name, first.kind, parts, tail)
            patches[name] = PatchValues(first.patches[name].type, value)
        return FieldValues(first.kind, first.dimensions, internal, patches)

    def _patch_array(self, name, kind, parts, tail) -> np.ndarray:
        complete = self._complete
        patch = complete.patches[name]
        if kind == "point":
            global_points = complete.patch_points(name)
            position = np.full(len(complete.points), -1)
            position[global_points] = np.arange(len(global_points))
            out = np.full((len(global_points),) + tail, np.nan)
            for processor, part in zip(self._processors, parts):
                local = processor.mesh.patch_points(name)
                out[position[processor.points[local]]] = part.patches[name].value
            return out
        out = np.full((patch.n_faces,) + tail, np.nan)
        for processor, part in zip(self._processors, parts):
            local = processor.mesh.patches[name]
            faces = np.abs(processor.faces[local.faces()]) - 1
            out[faces - patch.start_face] = part.patches[name].value
        return out


def _set_faces(out, addressing, values, flux: bool) -> None:
    if flux:
        values = np.where(addressing < 0, -values, values)
    out[np.abs(addressing) - 1] = values
