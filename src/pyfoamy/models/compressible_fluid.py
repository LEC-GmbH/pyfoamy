"""Shared base for the `fluid`/`multicomponentFluid`/`XiFluid` setups."""

from typing import Optional

from pyfoamy.case import Case
from pyfoamy.models.solver_setup import BuoyantSolverSetup
from pyfoamy.models.thermo import ThermoTypeSpec
from pyfoamy.models.turbulence import TurbulenceSpec
from pyfoamy.parsing.grammar import Directive, String


def append_include(dictionary, relative_path: str) -> None:
    """Appends a bare `#include "$FOAM_CASE/constant/<relative_path>"`.

    Mutates the live `Dictionary`, since a plain dict cannot express a
    directive entry; call it on a registered dict (`case[path]` or what
    `add_dict()` returns).

    Args:
        dictionary: The registered `Dictionary` to append to.
        relative_path: Path relative to `constant/`, e.g. `"species.dat"`.
    """
    dictionary.data.append(
        Directive(
            "#include", args=String(f"$FOAM_CASE/constant/{relative_path}"),
        )
    )


class CompressibleFluidSetup(BuoyantSolverSetup):
    """Shared setup for single-phase compressible solver modules.

    Buoyancy replaces the solved `p` with `p_rgh` and adds `constant/g`;
    transonic adds a `div(phid,p)` scheme. `rho`'s `fvSolution` entry is
    `{"solver": "diagonal"}`: it must exist but its content has no effect.
    Subclasses override `_physical_properties_extra()` for the
    `physicalProperties` body beyond `thermoType`.

    Args:
        thermo_type: `thermoType` structure.
        turbulence: Turbulence model choice; default laminar.
        buoyant: Solve `p_rgh` and add `constant/g`.
        transonic: Add the transonic pressure scheme.
    """

    solver_name: str = ""

    def __init__(
        self,
        thermo_type: ThermoTypeSpec,
        turbulence: Optional[TurbulenceSpec] = None,
        buoyant: bool = False,
        transonic: bool = False,
    ):
        super().__init__(buoyant, transonic)
        self.thermo_type = thermo_type
        self.turbulence = turbulence or TurbulenceSpec("laminar")

    def _physical_properties_extra(self) -> dict:
        raise NotImplementedError

    def _split_physical_properties(self, case: Case) -> None:
        # Hook for subclasses that move part of `physicalProperties` into
        # an `#include`d file, right after it is set; no-op by default.
        pass

    def required_fields(self):
        """Returns the pressure field, `U`, `T`, turbulence, `alphat` fields.

        The energy field (`h`/`e`) and `rho` are excluded: derived, with no
        IC/BC data. `alphat` is included for non-laminar turbulence.
        """
        fields = super().required_fields() + ["T"]
        fields += self.turbulence.required_fields()
        if self.turbulence.simulation_type != "laminar":
            fields.append("alphat")
        return fields

    def _momentum_group_fields(self):
        fields = ["U", self.thermo_type.energy_field]
        fields += [
            field
            for field in self.turbulence.required_fields()
            if field != "nut"
        ]
        return fields

    def _div_order_categories(self) -> dict:
        categories = {
            "phi,U": "defaultV",
            f"phi,{self.thermo_type.energy_field}": "default",
        }
        for field, category in (
            self.turbulence.required_div_categories().items()
        ):
            categories[f"phi,{field}"] = category
        categories.update(self._transonic_div_categories())
        return categories

    def _fixed_div_terms(self) -> dict:
        return {
            "phi,K": "Gauss linear",
            "((rho*nuEff)*dev2(T(grad(U))))": "Gauss linear",
        }

    def _constant_dicts(self, case: Case, gravity) -> None:
        case["constant/physicalProperties"] = {
            "thermoType": self.thermo_type.to_dict(),
            **self._physical_properties_extra(),
        }
        self._split_physical_properties(case)
        case["constant/momentumTransport"] = (
            self.turbulence.momentum_transport_dict()
        )
        gravity_entry = self._gravity_entry(gravity)
        if gravity_entry is not None:
            case["constant/g"] = gravity_entry
