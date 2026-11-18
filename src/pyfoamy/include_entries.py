"""Shared template resolution, validation and expansion for `#include*`.

`#includeFunc`/`#includeModel`/`#includeConstraint` share one C++ base
(`includeFuncEntry`) and one substitution mechanism (`dictionaryIO.C`:
`readConfigFile`/`addArgEntry`/`unsetConfigEntries`).
"""

import os
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, FrozenSet, Iterator, Optional, Set, Tuple

from pyfoamy.parsing.grammar import (
    Arguments, Compound, Directive, Dictionary, KeywordEntry, List, String,
    Word, to_grammar_value,
)

# Filled by `readConfigFile`'s own field/fields/objects handling, so
# never "required" despite an unresolved placeholder.
FIELD_LIST_KEYS = frozenset({"fields", "objects", "field"})

# The bracket text is only a hint, sometimes a literal example such as
# "<200>" (etc/caseDicts/fvConstraints/limitTemperature), so any
# content is accepted.
_PLACEHOLDER_RE = re.compile(r"^<[^<>]+>$")


@dataclass(frozen=True)
class IncludeEntryKind:
    """One of the three `#include<X>` directive families.

    Args:
        directive: The directive, e.g. `"#includeFunc"`.
        template_root: Template search path under `etc/`, e.g.
            `"caseDicts/functions"`.
        case_relative_dir: Case directory `readConfigFile` checks before
            `etc/` (`"system"` or `"constant"`); also holds `file`.
        label: Name for error messages.
        file: The case file the entries are written to, e.g. `"fvModels"`.
        always_known: Named arguments never flagged as unrecognized.
        regional: Whether each region has its own `file`.
    """
    directive: str
    template_root: str
    case_relative_dir: str
    label: str
    file: str
    always_known: FrozenSet[str] = frozenset()
    regional: bool = True


def _wm_project_dir() -> Path:
    value = os.environ.get("WM_PROJECT_DIR")
    if not value:
        raise RuntimeError(
            "no OpenFOAM install sourced ($WM_PROJECT_DIR unset) - "
            "templates can't be located"
        )
    return Path(value)


def find_template(
    kind: IncludeEntryKind,
    type_name: str,
    case_root: Optional[Path] = None,
) -> Path:
    """Locates the OpenFOAM template file for an `#include<X>` call.

    Args:
        case_root: If given, `<case_root>/<kind.case_relative_dir>/
            <type_name>` is checked first, as `readConfigFile` does (e.g.
            a tutorial's `constant/heatSource`).

    Raises:
        RuntimeError: If no OpenFOAM is sourced.
        ValueError: If no template named `type_name` is found.
    """
    if case_root is not None:
        case_relative = Path(case_root) / kind.case_relative_dir / type_name
        if case_relative.is_file():
            return case_relative

    root = _wm_project_dir() / "etc" / kind.template_root
    matches = [
        p for p in root.rglob(type_name)
        if p.is_file() and p.name == type_name
    ]
    if not matches:
        location = f"{root}" + (
            "" if case_root is None
            else f" or {Path(case_root) / kind.case_relative_dir}"
        )
        raise ValueError(
            f"no OpenFOAM {kind.label} template named {type_name!r} "
            f"found under {location}"
        )
    return matches[0]


def _is_placeholder(value) -> bool:
    return (
        isinstance(value, (Word, String))
        and bool(_PLACEHOLDER_RE.match(value.data))
    )


def _iter_leaf_values(value) -> Iterator[object]:
    if isinstance(value, (Compound, List)):
        for item in value:
            yield from _iter_leaf_values(item)
    else:
        yield value


def _iter_placeholders(
    d: Dictionary, prefix: Tuple[str, ...] = ()
) -> Iterator[Tuple[Tuple[str, ...], str]]:
    # `path` identifies the required argument (OpenFOAM's error names
    # it by slash-joined path, e.g. "porosityForce/cellZone"); `hint`
    # is just the bracket text.
    for entry in d:
        if not isinstance(entry, KeywordEntry):
            continue
        key = entry.key.data
        if key in FIELD_LIST_KEYS:
            continue
        if entry.is_dict:
            yield from _iter_placeholders(entry.value, prefix + (key,))
            continue
        for leaf in _iter_leaf_values(entry.value):
            if _is_placeholder(leaf):
                yield prefix + (key,), leaf.data[1:-1]


def _required_args(template_path: Path) -> Set[str]:
    d = Dictionary.from_file(template_path)
    return {"/".join(path) for path, _ in _iter_placeholders(d)}


def _known_keys(
    template_path: Path, _seen: Optional[Set[Path]] = None
) -> Set[str]:
    if _seen is None:
        _seen = set()
    template_path = template_path.resolve()
    if template_path in _seen:
        return set()
    _seen.add(template_path)

    d = Dictionary.from_file(template_path)
    known = {
        entry.key.data for entry in d if isinstance(entry, KeywordEntry)
    }
    for path, hint in _iter_placeholders(d):
        known.add("/".join(path))
        known.add(hint)
    for node in d:
        if not isinstance(node, Directive):
            continue
        if node.name != "#includeEtc" or node.args is None:
            continue
        chained = _wm_project_dir() / "etc" / node.args.data
        if chained.is_file():
            known |= _known_keys(chained, _seen)
    return known


def validate_call(
    kind: IncludeEntryKind,
    type_name: str,
    kwargs: Dict[str, object],
    case_root: Optional[Path] = None,
) -> None:
    """Checks an `#include<X>` call against its template, before OpenFOAM.

    Args:
        kwargs: The named arguments about to be passed.
        case_root: Passed to `find_template`.

    Raises:
        ValueError: If a required template setting has no kwarg.
    """
    template_path = find_template(kind, type_name, case_root)
    missing = _required_args(template_path) - set(kwargs)
    if missing:
        raise ValueError(
            f"{type_name!r} requires {sorted(missing)}, not given"
        )
    known = _known_keys(template_path) | kind.always_known
    unrecognized = set(kwargs) - known
    if unrecognized:
        warnings.warn(
            f"{type_name!r} doesn't declare {sorted(unrecognized)} - "
            "likely a typo (OpenFOAM accepts, and silently inserts, "
            "any named argument whether or not anything reads it)"
        )


def build_call_args(args: Tuple, kwargs: Dict[str, object]) -> Arguments:
    """Builds the `Arguments` for an `#include<X>` call's `(...)` part."""
    return Arguments(
        positional=[to_grammar_value(a) for a in args],
        named={key: to_grammar_value(v) for key, v in kwargs.items()},
    )


def expand_via_foam_dictionary(
    kind: IncludeEntryKind,
    call_text: str,
    case_root: Optional[Path] = None,
) -> Dictionary:
    """Resolves an `#include<X>` call via `foamDictionary -expand`.

    Args:
        call_text: The call's `type(args)` text.
        case_root: If the call uses a case-relative template, only that
            file is copied into the throwaway case, so the expanded file is
            not clobbered by a real one there.

    Raises:
        RuntimeError: If `foamDictionary` fails.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = Path(tmp)
        dict_path = f"{kind.case_relative_dir}/{kind.file}"
        target = case_dir / dict_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "FoamFile\n{\n"
            "    version 2.0;\n    format ascii;\n"
            "    class dictionary;\n"
            f'    location "{kind.case_relative_dir}";\n'
            f"    object {kind.file};\n}}\n"
            f"{kind.directive} {call_text}\n"
        )
        if case_root is not None:
            type_name = call_text.split("(", 1)[0]
            case_relative = (
                Path(case_root) / kind.case_relative_dir / type_name
            )
            if case_relative.is_file():
                copy = case_dir / kind.case_relative_dir / type_name
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_text(case_relative.read_text())
        result = subprocess.run(
            ["foamDictionary", "-expand", dict_path],
            capture_output=True, text=True, cwd=case_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"foamDictionary failed to expand {call_text!r}:\n"
                f"{result.stderr}"
            )
        return Dictionary(result.stdout)


@dataclass
class TemplateEntry:
    """A not yet rendered `#include<X>` call; subclasses set `kind`.

    Args:
        args: Positional arguments.
        kwargs: Named arguments; includes `name`.
        expand: `True` resolves via `foamDictionary -expand`, `False`
            writes the literal call, `None` uses the case default.
    """
    kind: ClassVar[IncludeEntryKind]
    type_name: str
    args: Tuple = ()
    kwargs: Dict[str, object] = field(default_factory=dict)
    expand: Optional[bool] = None

    def call_text(self) -> str:
        """Returns this call's `type(args)` text."""
        return f"{self.type_name}({build_call_args(self.args, self.kwargs)})"

    def render(self, expand: bool, case_root: Optional[Path] = None):
        """Renders a `Directive` (literal) or `KeywordEntry` (expanded).

        Args:
            expand: Used unless `self.expand` overrides it.
            case_root: Passed to `foamDictionary -expand` for a case-relative
                template.
        """
        if self.expand if self.expand is not None else expand:
            resolved = expand_via_foam_dictionary(
                self.kind, self.call_text(), case_root
            )
            name = self.kwargs["name"]
            return KeywordEntry(Word(name), resolved[name])
        return Directive(
            name=self.kind.directive,
            args=Word(self.type_name),
            call_args=build_call_args(self.args, self.kwargs),
        )


def entry_dict(
    type_name: str, libs=None, settings: Optional[dict] = None
) -> dict:
    """Builds a `type`/`libs`/settings entry directly, without template.

    Args:
        libs: Shared libraries it needs; none writes no `libs`.
        settings: Type-specific settings, merged as-is.
    """
    result = {"type": type_name}
    if libs:
        result["libs"] = list(libs)
    result.update(settings or {})
    return result


class IncludeRegistry:
    """One case's entries of an `#include<X>` family, rendered on write.

    Args:
        template_cls: The family's `TemplateEntry` subclass.
        files: The case's `CaseFiles`.
    """

    def __init__(self, template_cls, files):
        self.template_cls = template_cls
        self.kind = template_cls.kind
        self._files = files
        self._entries: Dict[Tuple[Optional[str], str], object] = {}

    def _key(self, name: str, region: Optional[str], overwrite: bool):
        if self.kind.regional:
            self._files.check_region(region, self.kind.file)
        key = (region, name)
        if key in self._entries and not overwrite:
            where = "" if region is None else f" for region {region!r}"
            raise ValueError(
                f"{self.kind.label} {name!r} is already registered{where}"
                " - pass overwrite=True to replace it"
            )
        return key

    def add(
        self, name: str, type_name: str, args: Tuple = (),
        kwargs: Optional[Dict[str, object]] = None, *,
        region: Optional[str] = None, overwrite: bool = False,
        expand: Optional[bool] = None,
    ) -> None:
        """Registers a call of an OpenFOAM template, validated against it.

        Raises:
            KeyError: If `region` is not a declared region.
            ValueError: If `region` is omitted in a case with regions, `name`
                exists and not `overwrite`, or a required setting is missing.
        """
        key = self._key(name, region, overwrite)
        kwargs = dict(kwargs or {}, name=name)
        validate_call(self.kind, type_name, kwargs, self._files.root)
        self._entries[key] = self.template_cls(
            type_name, tuple(args), kwargs, expand=expand,
        )

    def add_dict(
        self, name: str, entry: dict, *,
        region: Optional[str] = None, overwrite: bool = False,
    ) -> None:
        """Registers a plain dict entry; raises as `add()`."""
        self._entries[self._key(name, region, overwrite)] = entry

    def flush(self, expand: bool) -> None:
        """Renders the entries into the family's file(s) of the case."""
        by_region: Dict[Optional[str], Dictionary] = {}
        for (region, name), entry in self._entries.items():
            entries = by_region.setdefault(region, Dictionary())
            if isinstance(entry, TemplateEntry):
                entries.append(entry.render(expand, self._files.root))
            else:
                entries.set(name, entry)
        for region, entries in by_region.items():
            self._files[
                Path(self.kind.case_relative_dir, region or "", self.kind.file)
            ] = entries
