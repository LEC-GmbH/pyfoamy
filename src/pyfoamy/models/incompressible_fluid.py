"""Builds a `Case` for OpenFOAM-dev's `incompressibleFluid` module."""

from typing import Optional

from pyfoamy.case import Case
from pyfoamy.models.solver_setup import SolverSetup
from pyfoamy.models.turbulence import TurbulenceSpec


class IncompressibleFluidSetup(SolverSetup):
    """A single-phase, constant-viscosity incompressible flow setup.

    No `alphat` or buoyancy: `incompressibleFluid` has neither a
    thermophysical transport dependency nor buoyancy code. `pressure_field`
    is always `"p"`.

    Args:
        nu: Kinematic viscosity.
        turbulence: Turbulence model choice; default laminar.
    """

    solver_name = "incompressibleFluid"

    def __init__(
        self,
        nu,
        turbulence: Optional[TurbulenceSpec] = None,
    ):
        self.nu = nu
        self.turbulence = turbulence or TurbulenceSpec("laminar")

    def required_fields(self):
        """Returns the pressure field and `U` plus the turbulence fields."""
        return super().required_fields() + self.turbulence.required_fields()

    def _constant_dicts(self, case: Case, gravity) -> None:
        case["constant/physicalProperties"] = {
            "viscosityModel": "constant",
            "nu": self.nu,
        }
        case["constant/momentumTransport"] = (
            self.turbulence.momentum_transport_dict()
        )

    def _momentum_group_fields(self):
        return ["U"] + [
            field
            for field in self.turbulence.required_fields()
            if field != "nut"
        ]

    def _div_order_categories(self) -> dict:
        categories = {"phi,U": "defaultV"}
        for field, category in (
            self.turbulence.required_div_categories().items()
        ):
            categories[f"phi,{field}"] = category
        return categories
