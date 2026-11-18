"""Registry of a case's dictionary files: read, edit in memory, write."""

import shutil
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, NamedTuple, Optional, Set, Union

from pyfoamy.parsing.grammar import Dictionary, String
from pyfoamy.parsing.regions import Region, control_dict_regions

_FOAM_FILE_VERSION = Decimal("2.0")
_FOAM_FILE_FORMAT = "ascii"

# Dicts with a hardcoded, non-overridable name in every case -
# always registered, never auto-vivified on first access.
_STANDARD_DICTS = (
    "system/controlDict", "system/fvSchemes", "system/fvSolution",
)


class _PendingEntry(NamedTuple):
    dictionary: Dictionary
    class_name: str
    write_header: bool = True


def _orig_names(
    orig: Union[None, bool, Iterable[str]]
) -> Optional[Set[str]]:
    if orig is True or not orig:
        return None
    return set(orig)


def _coerce_to_dictionary(value: Union[Dictionary, dict]) -> Dictionary:
    if isinstance(value, Dictionary):
        return value
    if isinstance(value, dict):
        return Dictionary(value)
    raise TypeError(
        f"expected a Dictionary or dict, got {type(value)!r}"
    )


def foam_file_header(relative_path, class_name: str) -> dict:
    """Returns the `FoamFile` header entries for a case-relative path."""
    path = Path(relative_path)
    return {
        "version": _FOAM_FILE_VERSION,
        "format": _FOAM_FILE_FORMAT,
        "class": class_name,
        "location": String(str(path.parent)),
        "object": path.name,
    }


class CaseFiles:
    """The dictionary files of a case directory, held in memory until
    `write()`.

    Args:
        root: Path to the case directory.
    """

    def __init__(self, root):
        self.root = Path(root)
        self._pending: Dict[Path, _PendingEntry] = {}
        # Parsed on first access, so one construct the grammar lacks
        # does not make the whole case unopenable.
        self._unparsed: Set[Path] = set()
        for relative_path in _STANDARD_DICTS:
            if (self.root / relative_path).is_file():
                self._unparsed.add(Path(relative_path))
            else:
                self.add_dict(relative_path)

    def read_dict(self, relative_path) -> Dictionary:
        """Reads a dictionary at an exact path relative to the case root."""
        return Dictionary.from_file(self.root / relative_path)

    def write_dict(
        self,
        relative_path,
        value: Union[Dictionary, dict],
        class_name: str = "dictionary",
        header: bool = True,
    ) -> None:
        """Writes a dictionary to a path relative to the case root.

        Creates parent directories; adds a `FoamFile` header if `value` has
        none and `header` is `True`.

        Args:
            value: A `Dictionary` or plain `dict`; never mutated.
            class_name: `FoamFile` `class` for an added header.
            header: If `False`, never add a header (for `#include`-only
                fragment files).

        Raises:
            TypeError: If `value` is neither a `Dictionary` nor a `dict`.
        """
        dictionary = _coerce_to_dictionary(value)

        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not header or "FoamFile" in dictionary:
            text = dictionary.as_text + "\n"
        else:
            header_text = Dictionary({
                "FoamFile": foam_file_header(relative_path, class_name),
            }).as_text
            text = header_text + "\n" + dictionary.as_text + "\n"
        path.write_text(text)

    def list_regions(self) -> list[Region]:
        """Returns the regions declared in `regionSolvers`.

        Reads the registered `system/controlDict`, so unwritten edits
        count.

        Raises:
            KeyError: If the case has no `regionSolvers` entry.
        """
        return control_dict_regions(self["system/controlDict"])

    def __getitem__(self, relative_path) -> Dictionary:
        key = Path(relative_path)
        if key in self._unparsed:
            self._pending[key] = _PendingEntry(
                self.read_dict(key), "dictionary",
            )
            self._unparsed.discard(key)
        if key not in self._pending:
            raise KeyError(
                f"no dictionary registered at {str(key)!r} - call "
                "add_dict()/add_field() to create one first"
            )
        return self._pending[key].dictionary

    def __setitem__(self, relative_path, value: Union[Dictionary, dict]
                    ) -> None:
        key = Path(relative_path)
        dictionary = _coerce_to_dictionary(value)
        self._unparsed.discard(key)
        self._pending[key] = _PendingEntry(dictionary, "dictionary")

    def add_dict(
        self, relative_path, content=None, header: bool = True,
    ) -> Dictionary:
        """Registers a new in-memory dictionary; touches no disk.

        Args:
            content: A `Dictionary` or plain `dict`; default empty.
            header: If `False`, never add a `FoamFile` header on write (for
                `#include`-only fragment files).

        Raises:
            ValueError: If `relative_path` is already registered.
            TypeError: If `content` is neither a `Dictionary` nor a `dict`.
        """
        return self._register(
            relative_path, content, "dictionary", write_header=header,
        )

    def _register(
        self, relative_path, content, class_name: str,
        write_header: bool = True,
    ) -> Dictionary:
        key = Path(relative_path)
        if key in self._pending or key in self._unparsed:
            raise ValueError(
                f"{str(key)!r} is already registered - use "
                f"case[{str(key)!r}] to read/mutate it, or "
                f"case[{str(key)!r}] = ... to replace it wholesale"
            )
        dictionary = (
            Dictionary() if content is None
            else _coerce_to_dictionary(content)
        )
        self._pending[key] = _PendingEntry(
            dictionary, class_name, write_header,
        )
        return dictionary

    def region_names(self) -> list[str]:
        try:
            return [region.name for region in self.list_regions()]
        except KeyError:
            return []

    def check_region(self, region: Optional[str], label: str) -> None:
        valid = self.region_names()
        if region is None:
            if valid:
                raise ValueError(
                    f"this case has regions {valid} - region= is "
                    f"required for {label}, since a case-root "
                    f"{label} file would never be constructed (every "
                    "mesh a multi-region solver builds is one of its "
                    "named regions, never an implicit unnamed one)"
                )
            return
        if region not in valid:
            raise KeyError(
                f"unknown region {region!r}, declared regions are "
                f"{valid}"
            )

    def write(
        self,
        orig: Union[None, bool, Iterable[str]] = None,
    ) -> None:
        """Writes every registered dictionary to disk, via `write_dict`.

        Args:
            orig: `None` writes each file at its registered path. `True`
                appends `.orig` to every file name; a collection of file names
                (not paths) appends it only to those.
        """
        orig_names = _orig_names(orig)

        def target(key):
            wants_orig = (
                orig is True
                or (orig_names is not None and key.name in orig_names)
            )
            return key.with_name(f"{key.name}.orig") if wants_orig else key

        for key, entry in self._pending.items():
            self.write_dict(
                target(key), entry.dictionary, class_name=entry.class_name,
                header=entry.write_header,
            )
        # Never accessed, so unchanged: copied verbatim, comments and all.
        for key in self._unparsed:
            if target(key) != key:
                shutil.copyfile(self.root / key, self.root / target(key))
