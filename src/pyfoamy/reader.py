"""Reads meshes and fields of a case through the python or native backend."""

from pathlib import Path
from typing import Dict, Optional

from pyfoamy import native
from pyfoamy.parsing.decomposed import DecomposedCase
from pyfoamy.parsing.field import FieldValues
from pyfoamy.parsing.polymesh import PolyMesh
from pyfoamy.parsing.timedirs import find_instance, resolve_time_instance


def _has_time(directory: Path, time_value) -> bool:
    try:
        resolve_time_instance(directory, time_value)
    except FileNotFoundError:
        return False
    return True


class CaseReader:
    """Mesh and field reads of one case; `close()` releases native handles.

    Both backends return a `PolyMesh` and `FieldValues`. A decomposed
    case is read as the complete case (python backend only).

    Args:
        root: Path to the case directory.
        default_backend: `"python"` or `"openfoam"`, used when a read's
            `backend` is `None`.
    """

    def __init__(self, root, default_backend: str = "python"):
        self.root = Path(root)
        self.default_backend = default_backend
        self._native_cases: Dict[Optional[str], native.Case] = {}
        self._decomposed_cases: Dict[Optional[str], DecomposedCase] = {}
        self._meshes: Dict[Optional[str], tuple] = {}

    def _native_case(self, region: Optional[str]) -> native.Case:
        if region not in self._native_cases:
            self._native_cases[region] = native.Case(
                self.root.parent, self.root.name, region
            )
        return self._native_cases[region]

    def _decomposed_case(self, region: Optional[str]) -> DecomposedCase:
        if region not in self._decomposed_cases:
            self._decomposed_cases[region] = DecomposedCase(self.root, region)
        return self._decomposed_cases[region]

    def _backend(self, time_value, backend, decomposed) -> str:
        backend = backend or self.default_backend
        if backend not in ("python", "openfoam"):
            raise ValueError(f"unknown backend {backend!r}")
        if decomposed is None:
            first = self.root / "processor0"
            decomposed = (
                first.is_dir() and _has_time(first, time_value)
                and not _has_time(self.root, time_value)
            )
        if decomposed and backend == "openfoam":
            raise ValueError("the openfoam backend reads no decomposed case")
        return "decomposed" if decomposed else backend

    def _mesh(self, time_value, region) -> PolyMesh:
        key = tuple(
            find_instance(self.root, time_value, name, region=region)
            for name in ("points", "faces")
        )
        cached = self._meshes.get(region)
        if cached is None or cached[0] != key:
            cached = key, PolyMesh.read_at(self.root, time_value, region)
            self._meshes[region] = cached
        return cached[1]

    def read_mesh_at(
        self, time_value, region=None, backend=None, decomposed=None
    ) -> PolyMesh:
        """Reads the mesh at a time.

        Args:
            backend: `"python"` or `"openfoam"`; None for `default_backend`.
            decomposed: Read `processor*/`; None for when the time is
                only there.

        Raises:
            ValueError: If `backend` is unknown, or `"openfoam"` with a
                decomposed read.
        """
        backend = self._backend(time_value, backend, decomposed)
        if backend == "decomposed":
            return self._decomposed_case(region).read_mesh_at(time_value)
        if backend == "openfoam":
            return self._native_case(region).read_mesh_at(time_value)
        return self._mesh(time_value, region)

    def read_field_at(
        self, time_value, field_name, region=None, backend=None,
        decomposed=None,
    ) -> FieldValues:
        """Reads a field at a time; arguments as for `read_mesh_at`."""
        backend = self._backend(time_value, backend, decomposed)
        if backend == "decomposed":
            return self._decomposed_case(region).read_field_at(
                time_value, field_name
            )
        if backend == "openfoam":
            return self._native_case(region).read_field_at(
                time_value, field_name
            )
        directory = self.root / resolve_time_instance(self.root, time_value)
        if region is not None:
            directory = directory / region
        return FieldValues.read(
            directory / field_name, self._mesh(time_value, region)
        )

    def close(self) -> None:
        """Closes every open native backend handle."""
        for case in self._native_cases.values():
            case.close()
        self._native_cases.clear()
