"""Builds a `Case` for OpenFOAM-dev's `XiFluid` (b-Xi combustion) module."""

from dataclasses import dataclass
from typing import Optional

from pyfoamy.case import Case
from pyfoamy.enums import (
    MixtureType,
    SuModel,
    XiEqModel,
    XiGModel,
    XiModel,
    XiProfile,
)
from pyfoamy.models.coefficients import coefficients_entry
from pyfoamy.models.compressible_fluid import append_include
from pyfoamy.models.multicomponent_fluid import DLB_PYJAC, ReactionSpec
from pyfoamy.models.solver_setup import BuoyantSolverSetup
from pyfoamy.models.thermo import ThermoTypeSpec
from pyfoamy.models.turbulence import TurbulenceSpec

# Supported (u_mixture, b_mixture) combinations, plus each phase's
# extra transported species fields. The full registered set also
# includes uMulticomponent/bMulticomponent combos, not supported here.
_MIXTURE_TYPES = {
    MixtureType.HOMOGENEOUS: {
        "u_mixture": "uHomogeneous",
        "b_mixture": "bHomogeneous",
        "u_species": (),
        "b_species": (),
    },
    MixtureType.INHOMOGENEOUS: {
        "u_mixture": "uInhomogeneous",
        "b_mixture": "bInhomogeneous",
        "u_species": ("fu",),
        "b_species": ("ft",),
    },
    MixtureType.INHOMOGENEOUS_EGR: {
        "u_mixture": "uInhomogeneousEGR",
        "b_mixture": "bInhomogeneous",
        "u_species": ("fu", "egr"),
        "b_species": ("ft",),
    },
}


def xi_fluid_physical_properties_extra(
    stoichiometric_air_fuel_ratio: float, fuel: dict, oxidant: dict,
    products: dict,
) -> dict:
    """Builds the `physicalProperties.{u,b}` body beyond `thermoType`.

    For the `inhomogeneous`/`inhomogeneousEGR` mixture types, from fitted
    species dicts (e.g. `ThermoDatabase.lump()`); needs no Cantera.

    Args:
        stoichiometric_air_fuel_ratio: `stoichiometricAirFuelMassRatio`.
        fuel: `fuel` sub-dict (`specie`/`thermodynamics`/`transport`).
        oxidant: `oxidant` sub-dict, same shape.
        products: `products` sub-dict, same shape.

    Returns:
        A dict for `XiFluidSetup`'s `u_`/`b_physical_properties_extra`.
    """
    return {
        "stoichiometricAirFuelMassRatio": stoichiometric_air_fuel_ratio,
        "fuel": fuel,
        "oxidant": oxidant,
        "products": products,
    }


@dataclass
class SuModelSpec:
    """A laminar flame speed model choice for `combustionProperties`.

    Args:
        model: `"linearEquilibrium"`, `"transport"`, `"uniformConstant"`
            or `"unstrained"`.
        coefficients: Written to a bare `<model>` sub-dict
            (`pyfoamy.models.coefficients`), e.g. the nested
            `unstrainedLaminarFlameSpeed` for `"unstrained"`.

    Raises:
        ValueError: If `model` is unsupported.
    """
    model: SuModel
    coefficients: Optional[dict] = None

    def __post_init__(self):
        self.model = SuModel.coerce(self.model)

    def to_dict(self) -> dict:
        """Returns the `laminarFlameSpeed` content."""
        return {
            "model": self.model,
            **coefficients_entry(self.model, self.coefficients),
        }


@dataclass
class XiEqModelSpec:
    """A flame-wrinkling equilibrium sub-model (`flameWrinkling.eq`).

    Required for `XiModelSpec`'s `"equilibrium"` and `"transport"`.

    Args:
        model: `"constant"`, `"Gulder"` or `"instability"`.
        coefficients: Written to a bare `<model>` sub-dict.

    Raises:
        ValueError: If `model` is unsupported.
    """
    model: XiEqModel
    coefficients: Optional[dict] = None

    def __post_init__(self):
        self.model = XiEqModel.coerce(self.model)

    def to_dict(self) -> dict:
        """Returns the `eq` content."""
        return {
            "model": self.model,
            **coefficients_entry(self.model, self.coefficients),
        }


@dataclass
class XiGModelSpec:
    """A flame-wrinkling generation-rate sub-model (`generationRate`).

    Required only for `XiModelSpec`'s `"transport"`.

    Args:
        model: `"instability"` or `"KTS"`.
        coefficients: Written to a bare `<model>` sub-dict.

    Raises:
        ValueError: If `model` is unsupported.
    """
    model: XiGModel
    coefficients: Optional[dict] = None

    def __post_init__(self):
        self.model = XiGModel.coerce(self.model)

    def to_dict(self) -> dict:
        """Returns the `generationRate` content."""
        return {
            "model": self.model,
            **coefficients_entry(self.model, self.coefficients),
        }


@dataclass
class XiProfileSpec:
    """A flame-wrinkling profile sub-model (`flameWrinkling.profile`).

    Required for `XiModelSpec`'s `"equilibrium"` and `"transport"`.

    Args:
        model: `"cubic"`, `"linear"` or `"uniform"`.
        coefficients: Written to a bare `<model>` sub-dict.

    Raises:
        ValueError: If `model` is unsupported.
    """
    model: XiProfile
    coefficients: Optional[dict] = None

    def __post_init__(self):
        self.model = XiProfile.coerce(self.model)

    def to_dict(self) -> dict:
        """Returns the `profile` content."""
        return {
            "model": self.model,
            **coefficients_entry(self.model, self.coefficients),
        }


@dataclass
class XiModelSpec:
    """A flame-wrinkling model choice for `combustionProperties`.

    Required sub-models depend on `model`: `"uniformConstant"` needs
    `xi_value`; `"equilibrium"`/`"transport"` need `eq` and `profile`;
    `"transport"` also needs `generation_rate`. Other combinations raise.

    Args:
        model: `"equilibrium"`, `"transport"` or `"uniformConstant"`.
        eq: `XiEqModelSpec`, see above.
        profile: `XiProfileSpec`, see above.
        generation_rate: `XiGModelSpec`, see above.
        xi_value: Literal `Xi`, for `"uniformConstant"` only.
        coefficients: Further entries, e.g. `strainReduction`, written
            beside `model` and not to a `<model>` sub-dict: the model
            looks its sub-models up in the dict of its coefficients.

    Raises:
        ValueError: If `model` is unsupported or the sub-models/`xi_value`
            don't match it.
    """
    model: XiModel
    eq: Optional[XiEqModelSpec] = None
    profile: Optional[XiProfileSpec] = None
    generation_rate: Optional[XiGModelSpec] = None
    xi_value: Optional[float] = None
    coefficients: Optional[dict] = None

    def __post_init__(self):
        self.model = XiModel.coerce(self.model)
        if self.model == XiModel.UNIFORM_CONSTANT:
            if self.xi_value is None:
                raise ValueError(
                    "XiModelSpec('uniformConstant', ...) requires "
                    "xi_value"
                )
            if (
                self.eq is not None or self.profile is not None
                or self.generation_rate is not None
            ):
                raise ValueError(
                    "XiModelSpec('uniformConstant', ...) doesn't use "
                    "eq/profile/generation_rate"
                )
            return

        if self.xi_value is not None:
            raise ValueError(
                f"XiModelSpec({self.model!r}, ...) doesn't use "
                "xi_value"
            )
        if self.eq is None or self.profile is None:
            raise ValueError(
                f"XiModelSpec({self.model!r}, ...) requires both eq "
                "and profile"
            )
        if (
            self.model == XiModel.TRANSPORT
            and self.generation_rate is None
        ):
            raise ValueError(
                "XiModelSpec('transport', ...) requires generation_rate"
            )
        if (
            self.model == XiModel.EQUILIBRIUM
            and self.generation_rate is not None
        ):
            raise ValueError(
                "XiModelSpec('equilibrium', ...) doesn't use "
                "generation_rate"
            )

    def to_dict(self) -> dict:
        """Returns the `flameWrinkling` content."""
        result = {"model": self.model}
        if self.xi_value is not None:
            result["Xi"] = self.xi_value
        if self.eq is not None:
            result["eq"] = self.eq.to_dict()
        if self.profile is not None:
            result["profile"] = self.profile.to_dict()
        if self.generation_rate is not None:
            result["generationRate"] = self.generation_rate.to_dict()
        if self.coefficients:
            result.update(self.coefficients)
        return result


class XiFluidSetup(BuoyantSolverSetup):
    """A two-phase (unburnt/burnt) premixed combustion setup.

    Not a `CompressibleFluidSetup`: `XiFluid` has per-phase
    `physicalProperties.u`/`.b`, `momentumTransport.u`/`.b`, `T` and
    `alphat`, which does not fit that single-phase `build()`. The `.u`/`.b`
    suffix is a plain file name convention. As there, `rho`'s `fvSolution`
    entry is `{"solver": "diagonal"}`. Buoyancy and transonic come from
    `BuoyantSolverSetup`, inherited from `isothermalFluid` as in `fluid`.
    The `uMulticomponent`/`bMulticomponent` mixtures are not supported.

    Args:
        thermo_type: `transport`/`thermo`/`equationOfState`/`energy`/
            `specie` for both phases; its `type`/`mixture` are ignored, as
            each phase's come from `mixture_type`.
        mixture_type: `"homogeneous"`, `"inhomogeneous"` or
            `"inhomogeneousEGR"`.
        u_physical_properties_extra: `physicalProperties.u` beyond
            `thermoType`, e.g. `stoichiometricAirFuelMassRatio` + `fuel`/
            `oxidant`/`products` (inhomogeneous) or `Phi` + `reactants`
            (homogeneous); passed through.
        b_physical_properties_extra: Same for `.b`; may differ from the
            unburnt one, see `species_file`.
        su_model: Laminar flame speed model.
        xi_model: Flame wrinkling model.
        turbulence: Turbulence choice for both phases; default laminar.
        u_reaction: Unburnt-side reaction model; default none.
        b_reaction: Burnt-side reaction model; default none.
        combustion_extra: Extra top-level `combustionProperties` entries.
        buoyant: Solve `p_rgh` and add `constant/g`.
        transonic: Add the transonic pressure scheme.
        species_file: The union of both `*_physical_properties_extra` is
            written once to `constant/<species_file>` and `#include`d by
            both phases (each mixture class reads only its own keys);
            `None` embeds each phase's content inline.

    Raises:
        ValueError: If `mixture_type` is unsupported; if `species_file` is
            set and the two extras share a key with different values; or
            if both reactions select `"loadBalanced_pyJac"`, since the
            unprefixed global pyJac C symbols (`eval_jacob`, `PYJAC_NSP`,
            ...) allow only one compiled mechanism per process.
    """

    solver_name = "XiFluid"

    def __init__(
        self,
        thermo_type: ThermoTypeSpec,
        mixture_type: MixtureType,
        u_physical_properties_extra: dict,
        b_physical_properties_extra: dict,
        su_model: SuModelSpec,
        xi_model: XiModelSpec,
        turbulence: Optional[TurbulenceSpec] = None,
        u_reaction: Optional[ReactionSpec] = None,
        b_reaction: Optional[ReactionSpec] = None,
        combustion_extra: Optional[dict] = None,
        buoyant: bool = False,
        transonic: bool = False,
        species_file: Optional[str] = "species.dat",
    ):
        mixture_type = MixtureType.coerce(mixture_type, "mixture_type")
        if species_file is not None:
            for key, u_value in u_physical_properties_extra.items():
                b_value = b_physical_properties_extra.get(key, u_value)
                if b_value != u_value:
                    raise ValueError(
                        f"XiFluidSetup(..., species_file=...) can't "
                        f"merge {key!r} - u_physical_properties_extra "
                        "and b_physical_properties_extra disagree on "
                        "its value"
                    )
        if (
            u_reaction is not None
            and b_reaction is not None
            and u_reaction.chemistry_type == DLB_PYJAC
            and b_reaction.chemistry_type == DLB_PYJAC
        ):
            raise ValueError(
                "u_reaction and b_reaction can't both use "
                "\"loadBalanced_pyJac\" - pyJac's generated C symbols "
                "(eval_jacob, PYJAC_NSP, ...) are global and "
                "unprefixed, so two differently-compiled mechanisms "
                "can't coexist in one process"
            )
        super().__init__(buoyant, transonic)
        self.thermo_type = thermo_type
        self.mixture_type = mixture_type
        self.u_physical_properties_extra = u_physical_properties_extra
        self.b_physical_properties_extra = b_physical_properties_extra
        self.su_model = su_model
        self.xi_model = xi_model
        self.turbulence = turbulence or TurbulenceSpec("laminar")
        self.u_reaction = u_reaction or ReactionSpec("noReaction")
        self.b_reaction = b_reaction or ReactionSpec("noReaction")
        self.combustion_extra = combustion_extra
        self.species_file = species_file

    def _mixture(self) -> dict:
        return _MIXTURE_TYPES[self.mixture_type]

    def _species_content(self) -> dict:
        return {
            **self.u_physical_properties_extra,
            **self.b_physical_properties_extra,
        }

    def required_fields(self):
        """Returns the pressure field, `U`, `b`, `Xi`, `T.u`/`T.b`, turbulence
        fields, `alphat.u`/`alphat.b` if non-laminar, and the species fields
        of the mixture type.
        """
        mix = self._mixture()
        fields = super().required_fields() + ["b", "Xi", "T.u", "T.b"]
        if self.turbulence.simulation_type != "laminar":
            fields += ["alphat.u", "alphat.b"]
        fields += self.turbulence.required_fields()
        fields += [f"{specie}.u" for specie in mix["u_species"]]
        fields += [f"{specie}.b" for specie in mix["b_species"]]
        return fields

    def _momentum_group_fields(self):
        mix = self._mixture()
        fields = ["U", "b"]
        fields += [
            f"{self.thermo_type.energy_field}.{phase}"
            for phase in ("u", "b")
        ]
        fields += [
            field
            for field in self.turbulence.required_fields()
            if field != "nut"
        ]
        fields += [f"{specie}.u" for specie in mix["u_species"]]
        fields += [f"{specie}.b" for specie in mix["b_species"]]
        if self.xi_model.model != "uniformConstant":
            fields += ["Xi", "Su"]
        return fields

    def _div_order_categories(self) -> dict:
        mix = self._mixture()
        categories = {
            "phi,U": "defaultV",
            "phi,b": "01",
            "alphaPhi,K": "secondOrderBounded",
            f"alphaPhi,{self.thermo_type.energy_field}":
                "secondOrderBounded",
        }
        if self.xi_model.model != "uniformConstant":
            categories["phiXi,Xi"] = "secondOrderBounded"
        for specie in mix["u_species"] + mix["b_species"]:
            categories[f"alphaPhi,{specie}"] = "01"
        for field, category in (
            self.turbulence.required_div_categories().items()
        ):
            categories[f"phi,{field}"] = category
        categories.update(self._transonic_div_categories())
        return categories

    def _physical_properties(self, phase: str) -> dict:
        mix = self._mixture()
        thermo_type = {
            "type": f"{phase}RhoThermo",
            "mixture": mix[f"{phase}_mixture"],
            "transport": self.thermo_type.transport,
            "thermo": self.thermo_type.thermo,
            "equationOfState": self.thermo_type.equation_of_state,
            "specie": self.thermo_type.specie,
            "energy": self.thermo_type.energy,
        }
        if self.species_file is not None:
            return {"thermoType": thermo_type}
        extra = (
            self.u_physical_properties_extra if phase == "u"
            else self.b_physical_properties_extra
        )
        return {"thermoType": thermo_type, **extra}

    def _combustion_properties(self) -> dict:
        result = {
            "laminarFlameSpeed": self.su_model.to_dict(),
            "flameWrinkling": self.xi_model.to_dict(),
        }
        if self.combustion_extra:
            result.update(self.combustion_extra)
        return result

    def _fixed_div_terms(self) -> dict:
        return {"((rho*nuEff)*dev2(T(grad(U))))": "Gauss linear"}

    def _constant_dicts(self, case: Case, gravity) -> None:
        # Also writes `reactionProperties.{u,b}`/`chemistryProperties.{u,b}`
        # for each reaction that is not `"noReaction"`.
        for phase in ("u", "b"):
            case[f"constant/physicalProperties.{phase}"] = (
                self._physical_properties(phase)
            )
        if self.species_file is not None:
            case.add_dict(
                f"constant/{self.species_file}",
                self._species_content(),
                header=False,
            )
            for phase in ("u", "b"):
                append_include(
                    case[f"constant/physicalProperties.{phase}"],
                    self.species_file,
                )
        case["constant/momentumTransport"] = (
            self.turbulence.momentum_transport_dict()
        )
        for phase in ("u", "b"):
            case[f"constant/momentumTransport.{phase}"] = {
                "simulationType": self.turbulence.simulation_type,
            }
        gravity_entry = self._gravity_entry(gravity)
        if gravity_entry is not None:
            case["constant/g"] = gravity_entry
        case["constant/combustionProperties"] = (
            self._combustion_properties()
        )
        for phase, reaction in (
            ("u", self.u_reaction), ("b", self.b_reaction)
        ):
            reaction_properties = reaction.reaction_properties_dict()
            if reaction_properties is not None:
                case[f"constant/reactionProperties.{phase}"] = (
                    reaction_properties
                )
            chemistry_properties = reaction.chemistry_properties_dict()
            if chemistry_properties is not None:
                case[f"constant/chemistryProperties.{phase}"] = (
                    chemistry_properties
                )
                case.add_libs(*reaction.required_libs())
                if (
                    reaction.reactions is not None
                    and reaction.reactions_file is not None
                ):
                    case.add_dict(
                        f"constant/{phase}.{reaction.reactions_file}",
                        {"reactions": reaction.reactions},
                        header=False,
                    )
                    append_include(
                        case[f"constant/chemistryProperties.{phase}"],
                        f"{phase}.{reaction.reactions_file}",
                    )
