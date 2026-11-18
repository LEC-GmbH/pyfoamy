"""Typed accessors for a parsed field file."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from pyfoamy.parsing.fieldfile import _read_dictionary, read_field
from pyfoamy.parsing.grammar import (
    Bracket, Compound, Dictionary, DimensionSet, KeywordEntry, List,
    Variable, parse_dimensions,
)
from pyfoamy.parsing.timedirs import resolve_time_instance


@dataclass(frozen=True)
class FieldValue:
    """A `uniform` (one value) or `nonuniform` (list) field value."""

    uniform: bool
    value: object


@dataclass(frozen=True)
class FieldFunctionValue:
    """An unevaluated field-function value (`coded`, `zonal`, ...)."""

    type: str
    dict: Dictionary


def _to_plain(value):
    if isinstance(value, List):
        if value.uniform:
            return [_to_plain(value[0])] * value.count
        return [_to_plain(v) for v in value]
    return float(value)


def read_field_value(value) -> Union[FieldValue, FieldFunctionValue]:
    """Reads a parsed `internalField` or patch `value`-like entry.

    Raises:
        ValueError: If the value is neither (non)uniform nor a field
            function.
    """
    if isinstance(value, Dictionary):
        return FieldFunctionValue(str(value["type"]), value)
    if isinstance(value, Compound) and len(value) == 2:
        kind = str(value[0])
        if kind == "uniform":
            return FieldValue(True, _to_plain(value[1]))
        if kind == "nonuniform":
            return FieldValue(False, _to_plain(value[1]))
    raise ValueError(f"unrecognised field value form: {value!r}")


class BoundaryPatch:
    """One `boundaryField` patch."""

    def __init__(self, subdict: Dictionary):
        self.subdict = subdict

    @property
    def type(self) -> str:
        """The patch's boundary condition type."""
        return str(self.subdict["type"])

    def __getitem__(
        self, key: str
    ) -> Union[FieldValue, FieldFunctionValue]:
        return read_field_value(self.subdict[key])

    def __contains__(self, key: str) -> bool:
        return key in self.subdict


class Field:
    """A field file (`internalField`, `dimensions`, `boundaryField`)."""

    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary

    @classmethod
    def read(cls, path) -> "Field":
        """Reads a field file."""
        return cls(read_field(path))

    @classmethod
    def read_at(
        cls, case_dir, time_value, field_name, region=None
    ) -> "Field":
        """Reads a field from the time directory at exactly `time_value`.

        Raises:
            FileNotFoundError: If no time directory matches.
        """
        case_dir = Path(case_dir)
        instance = resolve_time_instance(case_dir, time_value)
        instance_dir = case_dir / instance
        if region is not None:
            instance_dir = instance_dir / region
        return cls.read(instance_dir / field_name)

    @property
    def dimensions(self) -> Union[DimensionSet, Bracket]:
        """The `dimensions` entry, see `parse_dimensions`."""
        return parse_dimensions(self.dictionary["dimensions"])

    @property
    def internal_field(self) -> Union[FieldValue, FieldFunctionValue]:
        """The `internalField` value."""
        return read_field_value(self.dictionary["internalField"])

    @property
    def boundary_field(self) -> Dict[str, BoundaryPatch]:
        """Patch name -> `BoundaryPatch`."""
        patches = self.dictionary["boundaryField"]
        return {
            str(entry.key): BoundaryPatch(entry.value)
            for entry in patches
            if isinstance(entry, KeywordEntry)
        }


_FIELD_CLASS = re.compile(
    r"(vol|surface|point)(Scalar|Vector|SymmTensor|Tensor|SphericalTensor)Field"
)
_COMPONENTS = {
    "Scalar": 1, "Vector": 3, "SymmTensor": 6, "Tensor": 9,
    "SphericalTensor": 1,
}


@dataclass(frozen=True)
class PatchValues:
    """One patch of `FieldValues`.

    `value` is None where the file has no plain value: none at all, a
    field function or a `$` macro.
    """

    type: str
    value: Optional[np.ndarray]


@dataclass(frozen=True)
class FieldValues:
    """A field's values as arrays of shape `(n,)` or `(n, components)`.

    `internal` has a row per cell (`kind` "vol"), internal face
    ("surface") or point ("point"). `patches` holds the mesh's patches
    that the `boundaryField` names, in mesh order; uniform values are
    repeated to the patch size.
    """

    kind: str
    dimensions: Union[DimensionSet, Bracket]
    internal: np.ndarray
    patches: Dict[str, PatchValues]

    @property
    def n_components(self) -> int:
        """1 for a scalar, 3 for a vector, ..."""
        return 1 if self.internal.ndim == 1 else self.internal.shape[1]

    @classmethod
    def read(cls, path, mesh) -> "FieldValues":
        """Reads a field file of `mesh` (a `PolyMesh`).

        Raises:
            ValueError: If the file is not a vol, surface or point field,
                or its `internalField` is not a (non)uniform value.
        """
        dictionary = _read_dictionary(path, as_array=True)
        class_name = str(dictionary["FoamFile"]["class"])
        match = _FIELD_CLASS.fullmatch(class_name)
        if match is None:
            raise ValueError(f"{path}: unsupported field class {class_name!r}")
        kind, n_components = match.group(1), _COMPONENTS[match.group(2)]
        internal = _array(
            dictionary["internalField"], _internal_size(mesh, kind),
            n_components,
        )
        if internal is None:
            raise ValueError(f"{path}: internalField is not a plain value")
        boundary = dictionary["boundaryField"]
        patches = {}
        for name, mesh_patch in mesh.patches.items():
            if name not in boundary:
                continue
            entry = boundary[name]
            value = None
            if "value" in entry:
                size = (
                    len(mesh.patch_points(name)) if kind == "point"
                    else mesh_patch.n_faces
                )
                value = _array(entry["value"], size, n_components)
            patches[name] = PatchValues(str(entry["type"]), value)
        return cls(
            kind, parse_dimensions(dictionary["dimensions"]), internal,
            patches,
        )


def _internal_size(mesh, kind: str) -> int:
    if kind == "vol":
        return mesh.n_cells
    if kind == "surface":
        return mesh.n_internal_faces
    return len(mesh.points)


def _array(value, size: int, n_components: int) -> Optional[np.ndarray]:
    if isinstance(value, (Dictionary, Variable)):
        return None
    if not (isinstance(value, Compound) and len(value) == 2):
        raise ValueError(f"unrecognised field value form: {value!r}")
    kind, data = str(value[0]), value[1]
    shape = (size,) if n_components == 1 else (size, n_components)
    if kind == "uniform":
        return np.full(shape, np.asarray(_to_plain(data), float))
    if kind != "nonuniform":
        raise ValueError(f"unrecognised field value form: {value!r}")
    if not isinstance(data, np.ndarray):
        data = np.asarray(_to_plain(data), float)
    return data.reshape((-1,) + shape[1:])
