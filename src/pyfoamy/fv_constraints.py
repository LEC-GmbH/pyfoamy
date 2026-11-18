"""Builds entries for a case's `fvConstraints` dict."""

from pyfoamy.include_entries import (
    IncludeEntryKind, TemplateEntry, entry_dict,
)

CONSTRAINT_KIND = IncludeEntryKind(
    directive="#includeConstraint",
    template_root="caseDicts/fvConstraints",
    case_relative_dir="system",
    label="fvConstraint",
    file="fvConstraints",
    # `phase`: read by some C++ classes (limitTemperature, fixedTemperature)
    # via `lookupOrDefault` without any template naming it.
    always_known=frozenset({"name", "funcName", "region", "phase"}),
)


class TemplateFvConstraint(TemplateEntry):
    """A not yet rendered fvConstraint from an OpenFOAM template."""
    kind = CONSTRAINT_KIND


fv_constraint_dict = entry_dict
