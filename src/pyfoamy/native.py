"""Loader for pyfoamy's native (OpenFOAM-linked) mesh/field reading backend.

Nothing runs at import time. The first call (e.g. `open_case()`)
builds `openfoam/native/pfNative.C` via `wmake libso` and loads it with
`ctypes`. The build is cached under a signature of this backend's
sources, the OpenFOAM install and the linked libraries' mtime/size.

Use `open_case()` as a context manager so the handle is released:

    with native.open_case(root_path, case_name) as case:
        mesh = case.read_mesh_at(0.1)
"""

import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
from typing import List

import numpy as np

from pyfoamy._openfoam_build import source_dir, wmake_libso
from pyfoamy.parsing.field import FieldValues, PatchValues
from pyfoamy.parsing.grammar import DimensionSet
from pyfoamy.parsing.listfile import CompactLabelListList
from pyfoamy.parsing.polymesh import MeshPatch, PolyMesh

_LIB_NAME = "libPfNative.so"
_TRACKED_SOURCE_FILES = ("pfNative.C", "Make/files", "Make/options")
_LINKED_LIBS = ("libOpenFOAM.so", "libfiniteVolume.so", "libmeshTools.so")

_lib_cache = {}


class NativeError(RuntimeError):
    """Raised for an unsourced OpenFOAM, a build failure or a library error."""


class _PfMeshData(ctypes.Structure):
    # Raw memory layout: must match `struct PfMeshData` in pfNative.C.
    _fields_ = [
        ("points", ctypes.POINTER(ctypes.c_double)),
        ("nPoints", ctypes.c_long),
        ("faceOffsets", ctypes.POINTER(ctypes.c_long)),
        ("faceData", ctypes.POINTER(ctypes.c_long)),
        ("nFaces", ctypes.c_long),
        ("faceDataCount", ctypes.c_long),
        ("owner", ctypes.POINTER(ctypes.c_long)),
        ("nOwner", ctypes.c_long),
        ("neighbour", ctypes.POINTER(ctypes.c_long)),
        ("nNeighbour", ctypes.c_long),
        ("patchNames", ctypes.POINTER(ctypes.c_char_p)),
        ("patchTypes", ctypes.POINTER(ctypes.c_char_p)),
        ("patchNFaces", ctypes.POINTER(ctypes.c_long)),
        ("patchStartFaces", ctypes.POINTER(ctypes.c_long)),
        ("nPatches", ctypes.c_long),
        ("patchGroups", ctypes.POINTER(ctypes.c_char_p)),
        ("nCells", ctypes.c_long),
    ]


class _PfFieldData(ctypes.Structure):
    # Raw memory layout: must match `struct PfFieldData` in pfNative.C.
    _fields_ = [
        ("nComponents", ctypes.c_long),
        ("dimensions", ctypes.c_double * 7),
        ("internalValues", ctypes.POINTER(ctypes.c_double)),
        ("nCells", ctypes.c_long),
        ("patchNames", ctypes.POINTER(ctypes.c_char_p)),
        ("patchTypes", ctypes.POINTER(ctypes.c_char_p)),
        ("patchSizes", ctypes.POINTER(ctypes.c_long)),
        ("patchValues", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
        ("nPatches", ctypes.c_long),
    ]


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise NativeError(
            f"${name} is not set - source an OpenFOAM install's "
            "etc/bashrc before using pyfoamy's native backend"
        )
    return value


def _src_dir() -> Path:
    path = source_dir("native")
    if path is None:
        raise NativeError(
            "pyfoamy's OpenFOAM sources not found; set "
            "$PYFOAMY_OPENFOAM_SRC to a checkout's openfoam/ directory"
        )
    return path


def _lib_dir() -> Path:
    return Path(_require_env("FOAM_USER_LIBBIN"))


def _lib_path() -> Path:
    return _lib_dir() / _LIB_NAME


def _sig_path() -> Path:
    return _lib_dir() / (_LIB_NAME + ".pyfoamy-sig")


def _compute_signature(src_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in _TRACKED_SOURCE_FILES:
        digest.update((src_dir / name).read_bytes())
    digest.update(_require_env("WM_PROJECT_DIR").encode())
    digest.update(os.environ.get("WM_OPTIONS", "").encode())
    lib_bin = Path(_require_env("FOAM_LIBBIN"))
    for name in _LINKED_LIBS:
        try:
            stat = (lib_bin / name).stat()
        except FileNotFoundError:
            continue
        digest.update(
            f"{name}:{stat.st_mtime_ns}:{stat.st_size}".encode()
        )
    return digest.hexdigest()


def _up_to_date(signature: str) -> bool:
    return (
        _lib_path().exists()
        and _sig_path().exists()
        and _sig_path().read_text() == signature
    )


def _build(src_dir: Path, signature: str) -> None:
    _lib_dir().mkdir(parents=True, exist_ok=True)
    lock_path = _lib_dir() / (_LIB_NAME + ".pyfoamy-lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # Another process may have built it while we waited.
            if _up_to_date(signature):
                return
            result = wmake_libso(src_dir)
            if result.returncode != 0:
                raise NativeError(
                    "building pyfoamy's native backend "
                    f"failed:\n{result.stdout}\n{result.stderr}"
                )
            _sig_path().write_text(signature)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _ensure_built() -> Path:
    src_dir = _src_dir()
    signature = _compute_signature(src_dir)
    if not _up_to_date(signature):
        _build(src_dir, signature)
    return _lib_path()


def _load_lib() -> ctypes.CDLL:
    path = _ensure_built()
    if _lib_cache.get("path") == path:
        return _lib_cache["lib"]

    lib = ctypes.CDLL(str(path))
    lib.pf_open_case.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.pf_open_case.restype = ctypes.c_int
    lib.pf_close_case.argtypes = [ctypes.c_void_p]
    lib.pf_read_mesh_at.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.POINTER(_PfMeshData),
    ]
    lib.pf_read_mesh_at.restype = ctypes.c_int
    lib.pf_free_mesh.argtypes = [ctypes.POINTER(_PfMeshData)]
    lib.pf_read_field.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_char_p,
        ctypes.POINTER(_PfFieldData),
    ]
    lib.pf_read_field.restype = ctypes.c_int
    lib.pf_free_field.argtypes = [ctypes.POINTER(_PfFieldData)]
    lib.pf_last_error.restype = ctypes.c_char_p

    _lib_cache["path"] = path
    _lib_cache["lib"] = lib
    return lib


def _array(ptr, count: int, dtype) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype)
    # astype copies: the buffer is freed right after decoding.
    return np.ctypeslib.as_array(ptr, (count,)).astype(dtype)


def _doubles(ptr, count: int) -> np.ndarray:
    return _array(ptr, count, np.float64)


def _labels(ptr, count: int) -> np.ndarray:
    return _array(ptr, count, np.int64)


def _strings(ptr, count: int) -> List[str]:
    return [ptr[i].decode() for i in range(count)]


def _decode_mesh_data(data: "_PfMeshData") -> PolyMesh:
    n_patches = data.nPatches
    names = _strings(data.patchNames, n_patches)
    types = _strings(data.patchTypes, n_patches)
    groups = _strings(data.patchGroups, n_patches)
    n_faces = _labels(data.patchNFaces, n_patches)
    start_faces = _labels(data.patchStartFaces, n_patches)
    return PolyMesh(
        points=_doubles(data.points, 3 * data.nPoints).reshape(-1, 3),
        faces=CompactLabelListList(
            _labels(data.faceOffsets, data.nFaces + 1),
            _labels(data.faceData, data.faceDataCount),
        ),
        owner=_labels(data.owner, data.nOwner),
        neighbour=_labels(data.neighbour, data.nNeighbour),
        patches={
            names[i]: MeshPatch(
                types[i], int(start_faces[i]), int(n_faces[i]),
                tuple(groups[i].split()),
            )
            for i in range(n_patches)
        },
        n_cells=data.nCells,
    )


def _decode_field_data(data: "_PfFieldData") -> FieldValues:
    n = data.nComponents

    def shaped(ptr, count):
        flat = _doubles(ptr, count * n)
        return flat if n == 1 else flat.reshape(-1, n)

    names = _strings(data.patchNames, data.nPatches)
    types = _strings(data.patchTypes, data.nPatches)
    patches = {
        names[i]: PatchValues(
            types[i], shaped(data.patchValues[i], data.patchSizes[i])
        )
        for i in range(data.nPatches)
    }
    return FieldValues(
        "vol", DimensionSet(*data.dimensions),
        shaped(data.internalValues, data.nCells), patches,
    )


class Case:
    """A case opened via the native backend.

    Keeps an opened `Time` and `fvMesh` alive across reads. Use
    `open_case()` so the handle is released via `close()`.
    """

    def __init__(self, root_path, case_name, region=None):
        # Set first so close()/__del__ work even if construction fails.
        self._handle = ctypes.c_void_p()
        self._lib = None
        self._lib = _load_lib()
        rc = self._lib.pf_open_case(
            str(root_path).encode(),
            str(case_name).encode(),
            (region or "").encode(),
            ctypes.byref(self._handle),
        )
        if rc != 0:
            message = self._lib.pf_last_error().decode(
                errors="replace"
            )
            raise NativeError(
                f"pf_open_case failed ({rc}): {message}"
            )

    def read_mesh_at(self, time_value) -> PolyMesh:
        """Reads the mesh at a time.

        Calls `fvMesh::readUpdate()`, so a moved or topology-changed mesh is
        updated in place.

        Raises:
            NativeError: If reading fails.
        """
        data = _PfMeshData()
        rc = self._lib.pf_read_mesh_at(
            self._handle,
            ctypes.c_double(time_value),
            ctypes.byref(data),
        )
        if rc != 0:
            message = self._lib.pf_last_error().decode(
                errors="replace"
            )
            raise NativeError(
                f"pf_read_mesh_at failed ({rc}): {message}"
            )
        try:
            return _decode_mesh_data(data)
        finally:
            self._lib.pf_free_mesh(ctypes.byref(data))

    def read_field_at(self, time_value, field_name: str) -> FieldValues:
        """Reads a field at a time on the already open mesh.

        Calls `fvMesh::readUpdate()` first, like `read_mesh_at()`.

        Raises:
            NativeError: If reading fails.
        """
        data = _PfFieldData()
        rc = self._lib.pf_read_field(
            self._handle,
            ctypes.c_double(time_value),
            field_name.encode(),
            ctypes.byref(data),
        )
        if rc != 0:
            message = self._lib.pf_last_error().decode(
                errors="replace"
            )
            raise NativeError(
                f"pf_read_field failed ({rc}): {message}"
            )
        try:
            return _decode_field_data(data)
        finally:
            self._lib.pf_free_field(ctypes.byref(data))

    def close(self) -> None:
        """Closes the native handle, if still open."""
        if self._handle:
            self._lib.pf_close_case(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "Case":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        # Backstop only: GC timing must not release native OpenFOAM
        # objects (signal handlers included); use `with`/close().
        self.close()


def open_case(root_path, case_name, region=None) -> Case:
    """Opens a case for native mesh/field reading, as a context manager.

    Args:
        root_path: Directory containing the case.
        case_name: The case directory's name.
        region: Region name; `None` opens the default region.

    Raises:
        NativeError: If opening fails.
    """
    return Case(root_path, case_name, region)
