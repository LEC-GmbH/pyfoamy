"""Reads, writes and creates OpenFOAM case directories."""

import math
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from pyfoamy import custom_lib
from pyfoamy.boundary_conditions import BoundaryField, scalar_wall_patches
from pyfoamy.casefiles import CaseFiles
from pyfoamy.fields import (
    FieldFile, _FIELD_CLASS_BY_TYPE, default_field,
)
from pyfoamy.function_objects import (
    TemplateFunctionObject, TimeControl, function_object_dict,
    merge_time_control,
)
from pyfoamy.fv_constraints import TemplateFvConstraint
from pyfoamy.fv_models import TemplateFvModel
from pyfoamy.include_entries import IncludeRegistry, entry_dict
from pyfoamy.parsing.grammar import (
    Bracket, Dictionary, DimensionSet, List, String, Uniform,
)
from pyfoamy.parsing.postprocessing import read_sets, read_time_series
from pyfoamy.reader import CaseReader
from pyfoamy.runner import Runner

_ZERO_BY_TYPE = {"scalar": 0, "vector": (0,) * 3, "tensor": (0,) * 9}

# Minimal system/controlDict entries Time needs to construct at all;
# see Case.create().
_MINIMAL_CONTROL_DICT_TAIL = {
    "endTime": 1,
    "deltaT": 1,
    "writeControl": "timeStep",
    "writeInterval": 1,
}


def _format_time(time_value) -> str:
    value = float(time_value)
    if not math.isfinite(value):
        raise ValueError(f"time must be finite, got {time_value!r}")
    if value == int(value):
        return str(int(value))
    return str(value)


class _PathPrefixView:
    def __init__(self, case: "Case", prefix):
        self._case = case
        self._prefix = Path(prefix)

    def __getitem__(self, relative_path) -> Dictionary:
        return self._case[self._prefix / relative_path]

    def __setitem__(self, relative_path, value) -> None:
        self._case[self._prefix / relative_path] = value

    def add_dict(self, relative_path, content=None) -> Dictionary:
        return self._case.add_dict(self._prefix / relative_path, content)


class _Region:
    def __init__(self, case: "Case", name: str):
        self.name = name
        self._case = case
        self.system = _PathPrefixView(case, Path("system") / name)
        self.constant = _PathPrefixView(case, Path("constant") / name)

    def add_model(self, name: str, type_name: str, *args, **kwargs) -> None:
        self._case.add_model(
            name, type_name, *args, region=self.name, **kwargs
        )

    def add_model_dict(
        self, name: str, type_name: str, *args, **kwargs
    ) -> None:
        self._case.add_model_dict(
            name, type_name, *args, region=self.name, **kwargs
        )

    def add_constraint(
        self, name: str, type_name: str, *args, **kwargs
    ) -> None:
        self._case.add_constraint(
            name, type_name, *args, region=self.name, **kwargs
        )

    def add_constraint_dict(
        self, name: str, type_name: str, *args, **kwargs
    ) -> None:
        self._case.add_constraint_dict(
            name, type_name, *args, region=self.name, **kwargs
        )


class _RegionsAccessor:
    def __init__(self, case: "Case"):
        self._case = case

    def __getitem__(self, name: str) -> _Region:
        valid = self._case.region_names()
        if name not in valid:
            raise KeyError(
                f"unknown region {name!r}, declared regions are "
                f"{valid}"
            )
        return _Region(self._case, name)


class _PhaseFieldProxy:

    def __init__(self, fields: list):
        self._fields = fields

    def set_bc(self, patch: str, bc) -> None:
        for field in self._fields:
            field.set_bc(patch, bc)


class _Fields:

    def __init__(self, files: CaseFiles):
        self._files = files

    def __getitem__(self, name: str):
        exact = None
        phases = []
        for key, entry in self._files._pending.items():
            if key.name == name:
                exact = entry.dictionary
            elif "." in key.name and key.name.split(".", 1)[0] == name:
                phases.append(entry.dictionary)
        if exact is not None:
            return exact
        if phases:
            return _PhaseFieldProxy(phases)
        raise KeyError(
            f"no field registered for {name!r} (checked the exact "
            f"name and any {name}.<phase> variants)"
        )

    def add(
        self, name, *, time, type=None, dimensions=None,
        internal_value=None, path=None,
    ) -> FieldFile:
        try:
            template = default_field(name)
        except KeyError:
            template = None

        if type is not None and type not in _FIELD_CLASS_BY_TYPE:
            raise ValueError(
                f"unknown field type {type!r}, expected one of "
                f"{sorted(_FIELD_CLASS_BY_TYPE)}"
            )
        if template is not None and dimensions is None:
            if (
                type is not None
                and _FIELD_CLASS_BY_TYPE[type] != template.class_name
            ):
                raise ValueError(
                    f"type={type!r} contradicts the {name!r} template "
                    f"({template.class_name}) - pass dimensions= too to "
                    "build the field without template"
                )
            class_name = template.class_name
            content = template.to_dict(
                internal_value, self._files.patch_roles,
            )
        else:
            if type is None or dimensions is None:
                raise ValueError(
                    f"{name!r} has no default field template - pass "
                    "type= and dimensions= explicitly"
                    if template is None else
                    f"dimensions= for {name!r} needs type= as well"
                )
            class_name = _FIELD_CLASS_BY_TYPE[type]
            if isinstance(dimensions, DimensionSet):
                dims_text = " ".join(map(str, dimensions.as_tuple()))
            elif dimensions.startswith("["):
                dims_text = dimensions[1:-1]
            else:
                dims_text = dimensions
            if internal_value is None:
                internal_value = _ZERO_BY_TYPE[type]
            content = {
                "dimensions": Bracket(dims_text),
                "internalField": Uniform(internal_value),
                "boundaryField": BoundaryField(
                    scalar_wall_patches(), roles=self._files.patch_roles,
                ),
            }

        relative_path = f"{_format_time(time)}/{path or name}"
        return self._files._register(
            relative_path, FieldFile(content), class_name,
        )


class Case(CaseFiles):
    """An OpenFOAM case directory.

    Use as a context manager if any read uses `backend="openfoam"`, so
    native handles are released.

    Args:
        root: Path to the case directory.
        default_backend: `"python"` or `"openfoam"`; used by `read_mesh_at`/
            `read_field_at` when their `backend` is `None`. `"openfoam"` is
            needed for decomposed cases, moving meshes and field-function
            boundary conditions.
        runner: Used by `run()`; a live-mode `Runner` bound to `root` is
            built on first use if omitted.
        patch_roles: `PatchRole` -> patch or group names, for the default
            boundary conditions of `add_field()`; without it a new field
            gets none.

    Attributes:
        default_expand_functions: Default for `write(expand_functions=)`:
            resolve template function objects via `foamDictionary -expand`
            (`True`) or write the literal `#includeFunc` call (`False`).
        default_expand_models: Same, for `#includeModel`.
        default_expand_constraints: Same, for `#includeConstraint`.
        fields: `case.fields[name]` is the registered `FieldFile`, or a
            proxy forwarding to every `<name>.<phase>` field (e.g. `T.u`,
            `T.b`) if there is no exact match.
        reader: The `CaseReader` behind `read_mesh_at`/`read_field_at`.
        read_time_series: `parsing.read_time_series` of this case.
        read_sets: `parsing.read_sets` of this case.
    """

    def __init__(
        self,
        root,
        default_backend: str = "python",
        runner: Optional[Runner] = None,
        patch_roles: Optional[dict] = None,
    ):
        super().__init__(root)
        self.runner = runner
        self.patch_roles = patch_roles or {}
        self.reader = CaseReader(self.root, default_backend)
        self.read_mesh_at = self.reader.read_mesh_at
        self.read_field_at = self.reader.read_field_at
        self.read_time_series = partial(read_time_series, self.root)
        self.read_sets = partial(read_sets, self.root)
        self.default_expand_functions = False
        self.default_expand_models = False
        self.default_expand_constraints = False
        self.fields = _Fields(self)
        self._functions = IncludeRegistry(TemplateFunctionObject, self)
        self._models = IncludeRegistry(TemplateFvModel, self)
        self._constraints = IncludeRegistry(TemplateFvConstraint, self)

    @classmethod
    def create(
        cls,
        root,
        start_time=None,
        regions: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> "Case":
        """Creates a new case directory.

        Creates `constant/`, `system/`, `constant/geometry/`, each region's
        subdirectories and, if missing, a minimal `system/controlDict`
        (an existing one is never overwritten).

        Args:
            start_time: If given, also creates that time directory (and
                `<time>/<region>/`).
            regions: Region name -> solver module, written to `regionSolvers`.
            **kwargs: Passed to `Case.__init__`.
        """
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "constant").mkdir(exist_ok=True)
        (root / "system").mkdir(exist_ok=True)
        (root / "constant" / "geometry").mkdir(exist_ok=True)

        if regions:
            for name in regions:
                (root / "constant" / name).mkdir(exist_ok=True)
                (root / "system" / name).mkdir(exist_ok=True)

        if start_time is not None:
            time_dir = root / _format_time(start_time)
            time_dir.mkdir(exist_ok=True)
            if regions:
                for name in regions:
                    (time_dir / name).mkdir(exist_ok=True)

        case = cls(root, **kwargs)
        control_dict_path = root / "system" / "controlDict"
        if not control_dict_path.is_file():
            values = {
                "startFrom": "startTime",
                "startTime": (
                    start_time if start_time is not None else 0
                ),
            }
            values.update(_MINIMAL_CONTROL_DICT_TAIL)
            if regions:
                values["regionSolvers"] = dict(regions)
            case.write_dict("system/controlDict", values)
            case["system/controlDict"] = values
        return case

    def add_libs(self, *names: str) -> None:
        """Merges names into `controlDict`'s `libs`, as quoted strings.

        An already-present name is not duplicated.
        """
        control_dict = self["system/controlDict"]
        existing = (
            [v.data for v in control_dict["libs"]]
            if "libs" in control_dict else []
        )
        merged = existing + [n for n in names if n not in existing]
        if merged == existing:
            return
        control_dict.set("libs", List([String(name) for name in merged]))

    def require_pyfoamy_lib(self) -> Path:
        """Builds `libpyfoamy.so` if needed and adds it to `libs`.

        For a case not set up by `SolverSetup.build()`, which adds it
        itself.

        Returns:
            The path to the built library.

        Raises:
            pyfoamy.custom_lib.CustomLibError: If no OpenFOAM is sourced, its
                version is unsupported, or the build fails.
        """
        path = custom_lib.ensure_built()
        self.add_libs("libpyfoamy.so")
        return path

    def add_field(
        self,
        name: str,
        *,
        time,
        type: Optional[str] = None,
        dimensions=None,
        internal_value=None,
        path=None,
    ) -> FieldFile:
        """Registers a new in-memory field; touches no disk.

        Seeded from `default_field(name)` if `name` has a template, with a
        `boundaryField` of the template's conditions (else `zeroGradient`)
        on the patches named by `patch_roles`. Use `.set_bc()` to override
        patches.

        Args:
            name: Base name, e.g. `"U"`.
            time: Time directory to write to.
            type: `"scalar"`/`"vector"`/`"tensor"`; with `dimensions`,
                builds the field without template.
            dimensions: Dimension string, e.g. `"[]"`, or a `DimensionSet`;
                see `type`.
            internal_value: Overrides the default `internalField`; zero of
                `type` for a field without template.
            path: File name under `time` if not `name` (e.g. `name="T"`,
                `path="T.u"`).

        Raises:
            ValueError: If the path is already registered, `type`/`dimensions`
                are missing for an untemplated name, `type` is unknown or
                contradicts the template.
        """
        return self.fields.add(
            name, time=time, type=type, dimensions=dimensions,
            internal_value=internal_value, path=path,
        )

    def add_function(
        self,
        name: str,
        type_name: str,
        *args,
        execute_control: Optional[TimeControl] = None,
        write_control: Optional[TimeControl] = None,
        execute_at_start: Optional[bool] = None,
        overwrite: bool = False,
        expand: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Registers a function object from an OpenFOAM template.

        Validated now against `etc/caseDicts/functions/**`: raises for a
        missing required setting, warns for an unrecognized one. Written as
        `#includeFunc <type_name>(...)` or expanded, see `expand`.

        There is no `region=`: `system/functions` is one case-global file.
        Pass `region=` as an ordinary keyword to scope the function itself.

        Args:
            name: Unique key, required.
            type_name: Function object type, e.g. `"probes"`.
            *args: Positional `#includeFunc` arguments.
            execute_control: Merged via `.to_dict("execute")`.
            write_control: Merged via `.to_dict("write")`.
            execute_at_start: Sets `executeAtStart`.
            overwrite: Allow replacing a function object of the same name.
            expand: Overrides `default_expand_functions`.
            **kwargs: Named `#includeFunc` arguments (e.g. `points=`).

        Raises:
            ValueError: If `name` exists and not `overwrite`, or a required
                setting is missing.
        """
        self._functions.add(
            name, type_name, args,
            merge_time_control(
                kwargs, execute_control, write_control, execute_at_start,
            ),
            overwrite=overwrite, expand=expand,
        )

    def add_function_dict(
        self,
        name: str,
        type_name: str,
        libs,
        settings: Optional[dict] = None,
        *,
        execute_control: Optional[TimeControl] = None,
        write_control: Optional[TimeControl] = None,
        execute_at_start: Optional[bool] = None,
        overwrite: bool = False,
    ) -> None:
        """Registers a function object from a plain dict, without template.

        Args:
            libs: Shared libraries it needs.
            settings: Type-specific settings, merged as-is.
            execute_control: Merged via `.to_dict("execute")`.
            write_control: Merged via `.to_dict("write")`.
            execute_at_start: Sets `executeAtStart`.
            overwrite: Allow replacing a function object of the same name.

        Raises:
            ValueError: If `name` exists and not `overwrite`.
        """
        self._functions.add_dict(
            name,
            function_object_dict(
                type_name, libs, settings,
                execute_control=execute_control,
                write_control=write_control,
                execute_at_start=execute_at_start,
            ),
            overwrite=overwrite,
        )

    def add_model(
        self,
        name: str,
        type_name: str,
        *args,
        region: Optional[str] = None,
        overwrite: bool = False,
        expand: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Registers an fvModel from an OpenFOAM template.

        Validated now against `etc/caseDicts/fvModels/**` (or a case-relative
        `constant/<type_name>`): raises for a missing required setting, warns
        for an unrecognized one. Written as `#includeModel <type_name>(...)`
        or expanded, see `expand`.

        Args:
            name: Unique key per `region`. Phase variants of one type need
                distinct names; the tutorial convention is
                `f"{type_name}.{phase}"`.
            type_name: fvModel type, e.g. `"porousZone"`.
            *args: Positional `#includeModel` arguments.
            region: Target `constant/<region>/fvModels`. `None` means
                `constant/fvModels`, valid only for a case without regions.
            overwrite: Allow replacing a model of the same name and region.
            expand: Overrides `default_expand_models`.
            **kwargs: Named `#includeModel` arguments (e.g. `cellZone=`).

        Raises:
            KeyError: If `region` is not a declared region.
            ValueError: If `region` is omitted in a case with regions, `name`
                exists and not `overwrite`, or a required setting is missing.
        """
        self._models.add(
            name, type_name, args, kwargs,
            region=region, overwrite=overwrite, expand=expand,
        )

    def add_model_dict(
        self,
        name: str,
        type_name: str,
        libs=None,
        settings: Optional[dict] = None,
        *,
        region: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Registers an fvModel from a plain dict, without template.

        Args:
            libs: Shared libraries it needs.
            settings: Type-specific settings, merged as-is.
            region: See `add_model()`.
            overwrite: Allow replacing a model of the same name and region.

        Raises:
            KeyError: If `region` is not a declared region.
            ValueError: If `region` is omitted in a case with regions, or
                `name` exists and not `overwrite`.
        """
        self._models.add_dict(
            name, entry_dict(type_name, libs, settings),
            region=region, overwrite=overwrite,
        )

    def add_constraint(
        self,
        name: str,
        type_name: str,
        *args,
        region: Optional[str] = None,
        overwrite: bool = False,
        expand: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Registers an fvConstraint from an OpenFOAM template.

        Validated now against `etc/caseDicts/fvConstraints/**` (or a
        case-relative `system/<type_name>`): raises for a missing required
        setting, warns for an unrecognized one. Written as
        `#includeConstraint <type_name>(...)` or expanded, see `expand`.

        Args:
            name: Unique key per `region`. Phase variants of one type need
                distinct names; the tutorial convention is
                `f"{type_name}.{phase}"`.
            type_name: fvConstraint type, e.g. `"limitTemperature"`.
            *args: Positional `#includeConstraint` arguments.
            region: Target `system/<region>/fvConstraints`; see `add_model()`.
            overwrite: Allow replacing a constraint of the same name and
                region.
            expand: Overrides `default_expand_constraints`.
            **kwargs: Named `#includeConstraint` arguments (e.g. `min=`).

        Raises:
            KeyError: If `region` is not a declared region.
            ValueError: If `region` is omitted in a case with regions, `name`
                exists and not `overwrite`, or a required setting is missing.
        """
        self._constraints.add(
            name, type_name, args, kwargs,
            region=region, overwrite=overwrite, expand=expand,
        )

    def add_constraint_dict(
        self,
        name: str,
        type_name: str,
        libs=None,
        settings: Optional[dict] = None,
        *,
        region: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Registers an fvConstraint from a plain dict, without template.

        Args:
            libs: Shared libraries it needs.
            settings: Type-specific settings, merged as-is.
            region: See `add_constraint()`.
            overwrite: Allow replacing a constraint of the same name and
                region.

        Raises:
            KeyError: If `region` is not a declared region.
            ValueError: If `region` is omitted in a case with regions, or
                `name` exists and not `overwrite`.
        """
        self._constraints.add_dict(
            name, entry_dict(type_name, libs, settings),
            region=region, overwrite=overwrite,
        )

    def write(
        self,
        orig: Union[None, bool, Iterable[str]] = None,
        expand_functions: Optional[bool] = None,
        expand_models: Optional[bool] = None,
        expand_constraints: Optional[bool] = None,
    ) -> None:
        """Writes every registered dictionary to disk, via `write_dict`.

        Writes unconditionally, each with its registered `class_name`.
        Function objects, fvModels and fvConstraints are rendered into
        `system/functions`, `constant/fvModels` and `system/fvConstraints`.

        Args:
            orig: `None` writes each file at its registered path. `True`
                appends `.orig` to every file name; a collection of file names
                (not paths) appends it only to those.
            expand_functions: Overrides `default_expand_functions`.
            expand_models: Overrides `default_expand_models`.
            expand_constraints: Overrides `default_expand_constraints`.
        """
        for registry, default, override in (
            (self._functions, self.default_expand_functions,
             expand_functions),
            (self._models, self.default_expand_models, expand_models),
            (self._constraints, self.default_expand_constraints,
             expand_constraints),
        ):
            registry.flush(default if override is None else override)
        super().write(orig)

    @property
    def system(self) -> _PathPrefixView:
        """A view of `case[...]` prefixed with `"system"`."""
        return _PathPrefixView(self, "system")

    @property
    def constant(self) -> _PathPrefixView:
        """A view of `case[...]` prefixed with `"constant"`."""
        return _PathPrefixView(self, "constant")

    @property
    def regions(self) -> _RegionsAccessor:
        """Per-region `.system`/`.constant` views; bad region: `KeyError`."""
        return _RegionsAccessor(self)

    def run(
        self,
        command: str,
        args: Optional[list] = None,
        options: Optional[Dict[str, object]] = None,
        parallel: bool = False,
        suffix: str = "",
        overwrite: bool = False,
        append: bool = False,
        np: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """Runs (or records) an application via `self.runner`.

        See `Runner.run` for the argument semantics.

        Args:
            np: Number of MPI ranks; default `numberOfSubdomains` from
                `system/decomposeParDict`.
        """
        if self.runner is None:
            self.runner = Runner(self.root)
        return self.runner.run(
            command,
            args=args,
            options=options,
            parallel=parallel,
            suffix=suffix,
            overwrite=overwrite,
            append=append,
            np=np,
            env=env,
        )

    def close(self) -> None:
        """Closes every open native backend handle."""
        self.reader.close()

    def __enter__(self) -> "Case":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        # Backstop only - close() should be called explicitly (e.g.
        # via `with`), not relied on via garbage-collection timing.
        self.close()
