"""Lazy access to `Case`, `Runner`, field templates, boundary conditions
and the OpenFOAM version.

Importing a package submodule runs this `__init__`, so loading the
core stack here would burden even `pyfoamy.cantera`/`pyfoamy.pyjac`,
which are meant to stay lightweight.
"""

__version__ = "0.4.1"

__all__ = [
    "Case",
    "Runner",
    "FieldTemplate",
    "AlphatJayatillekeWallFunction",
    "AlphatWallFunction",
    "Applicability",
    "BoundaryCondition",
    "BoundaryField",
    "Calculated",
    "default_field",
    "EntrainmentPressure",
    "EpsilonWallFunction",
    "ExternalTemperature",
    "FixedFluxExtrapolatedPressure",
    "FixedFluxPressure",
    "FixedValue",
    "Freestream",
    "FreestreamPressure",
    "FreestreamVelocity",
    "FlowRateInletVelocity",
    "GenericBoundaryCondition",
    "InletOutlet",
    "KqRWallFunction",
    "MovingWallSlipVelocity",
    "MovingWallVelocity",
    "NoSlip",
    "NutUSpaldingWallFunction",
    "NutkWallFunction",
    "OmegaWallFunction",
    "PatchRole",
    "PressureInletOutletVelocity",
    "RotatingWallVelocity",
    "Slip",
    "TurbulentEpsilon",
    "TurbulentInlet",
    "TurbulentKineticEnergy",
    "TurbulentOmega",
    "UniformFixedValue",
    "UniformInletOutlet",
    "UniformTotalPressure",
    "WaveTransmissive",
    "ZeroGradient",
    "alphat_wall_patches",
    "default_wall_boundary_field",
    "patches_by_name",
    "pressure_wall_patches",
    "scalar_wall_patches",
    "u_wall_patches",
    "openfoam_version",
    "use_openfoam_version",
]

_MODULE_OF = {
    "Case": "pyfoamy.case",
    "Runner": "pyfoamy.runner",
    "FieldTemplate": "pyfoamy.fields",
    "default_field": "pyfoamy.fields",
    "default_wall_boundary_field": "pyfoamy.fields",
    "openfoam_version": "pyfoamy.openfoam_version",
    "use_openfoam_version": "pyfoamy.openfoam_version",
}
#: Every other name in `__all__` comes from `boundary_conditions`.
_MODULE_OF.update({
    name: "pyfoamy.boundary_conditions"
    for name in __all__ if name not in _MODULE_OF
})


def __getattr__(name: str):
    module_name = _MODULE_OF.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache - only pay the import cost once
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
