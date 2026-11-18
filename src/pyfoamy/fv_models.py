"""Builds entries for a case's `fvModels` dict."""

from pyfoamy.include_entries import (
    IncludeEntryKind, TemplateEntry, entry_dict,
)

MODEL_KIND = IncludeEntryKind(
    directive="#includeModel",
    template_root="caseDicts/fvModels",
    case_relative_dir="constant",
    label="fvModel",
    file="fvModels",
    # `phase`: read by some C++ classes (viscousHeating, buoyancyForce, ...)
    # via `lookupOrDefault` without any template naming it.
    always_known=frozenset({"name", "funcName", "region", "phase"}),
)


class TemplateFvModel(TemplateEntry):
    """A not yet rendered fvModel from an OpenFOAM template."""
    kind = MODEL_KIND


fv_model_dict = entry_dict
