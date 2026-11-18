"""Builds a `Case` for OpenFOAM-dev's `fluid` solver module."""

from typing import Optional

from pyfoamy.models.compressible_fluid import (
    CompressibleFluidSetup, append_include,
)
from pyfoamy.models.thermo import ThermoTypeSpec
from pyfoamy.models.turbulence import TurbulenceSpec


class FluidSetup(CompressibleFluidSetup):
    """A single-mixture compressible flow setup.

    `physicalProperties` beyond `thermoType` is one `mixture {...}`
    sub-dict; adds a `div(phi,(p|rho))` scheme specific to this module.

    Args:
        thermo_type: `thermoType` structure.
        mixture_data: `mixture` sub-dict content, passed through.
        turbulence: Turbulence model choice; default laminar.
        buoyant: Solve `p_rgh` and add `constant/g`.
        transonic: Add the transonic pressure scheme.
        mixture_file: File under `constant/` that `mixture_data` is written
            to and `physicalProperties` `#include`s, like
            `MulticomponentFluidSetup`'s `species_file`; `None` embeds it
            inline.
    """

    solver_name = "fluid"

    def __init__(
        self,
        thermo_type: ThermoTypeSpec,
        mixture_data: dict,
        turbulence: Optional[TurbulenceSpec] = None,
        buoyant: bool = False,
        transonic: bool = False,
        mixture_file: Optional[str] = "mixture.dat",
    ):
        super().__init__(thermo_type, turbulence, buoyant, transonic)
        self.mixture_data = mixture_data
        self.mixture_file = mixture_file

    def _physical_properties_extra(self) -> dict:
        if self.mixture_file is not None:
            return {}
        return {"mixture": self.mixture_data}

    def _split_physical_properties(self, case) -> None:
        if self.mixture_file is None:
            return
        case.add_dict(
            f"constant/{self.mixture_file}",
            {"mixture": self.mixture_data},
            header=False,
        )
        append_include(
            case["constant/physicalProperties"], self.mixture_file,
        )

    def _fixed_div_terms(self) -> dict:
        terms = dict(super()._fixed_div_terms())
        terms["phi,(p|rho)"] = "Gauss linear"
        return terms
