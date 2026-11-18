"""The `thermoType` sub-dict of a compressible `physicalProperties`."""

from dataclasses import dataclass


@dataclass
class ThermoTypeSpec:
    """The structural choice of `thermoType` type names.

    Per-mixture data (`specie`/`thermodynamics`/`transport`/
    `equationOfState`) is supplied separately by the caller.

    Args:
        type: e.g. `"hePsiThermo"`.
        mixture: e.g. `"pureMixture"`.
        transport: e.g. `"const"`.
        thermo: e.g. `"hConst"`.
        equation_of_state: e.g. `"perfectGas"`.
        energy: e.g. `"sensibleEnthalpy"`.
        specie: Default `"specie"`.
    """
    type: str
    mixture: str
    transport: str
    thermo: str
    equation_of_state: str
    energy: str
    specie: str = "specie"

    @property
    def energy_field(self) -> str:
        """The energy field name, `"h"` or `"e"`, derived from `energy`.

        Raises:
            ValueError: If `energy` names neither enthalpy nor internal energy.
        """
        if "Enthalpy" in self.energy:
            return "h"
        if "InternalEnergy" in self.energy:
            return "e"
        raise ValueError(
            f"cannot derive the transported energy field name from "
            f"energy={self.energy!r} - only confirmed for "
            "'...Enthalpy' -> 'h' and '...InternalEnergy' -> 'e'"
        )

    def to_dict(self) -> dict:
        """Returns the `thermoType` sub-dict content."""
        return {
            "type": self.type,
            "mixture": self.mixture,
            "transport": self.transport,
            "thermo": self.thermo,
            "equationOfState": self.equation_of_state,
            "specie": self.specie,
            "energy": self.energy,
        }
