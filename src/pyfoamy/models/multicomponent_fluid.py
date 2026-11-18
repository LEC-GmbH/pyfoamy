"""Builds a `Case` for OpenFOAM-dev's `multicomponentFluid` module."""

from dataclasses import dataclass
from typing import List, Optional

from pyfoamy.enums import (
    ChemistryType, CustomName, EDCVersion, ReactionModel,
)
from pyfoamy.models.coefficients import coefficients_entry
from pyfoamy.models.compressible_fluid import (
    CompressibleFluidSetup, append_include,
)
from pyfoamy.models.thermo import ThermoTypeSpec
from pyfoamy.models.turbulence import TurbulenceSpec

#: DLBFoam's chemistry types (third-party), opt-in `chemistry_type` values.
DLB = CustomName("loadBalanced")
DLB_PYJAC = CustomName("loadBalanced_pyJac")
_DLB_LIBS = ["libchemistryModel_DLB.so", "libODE_DLB.so"]


@dataclass
class ReactionSpec:
    """A reaction model choice for `constant/reactionProperties`.

    Args:
        model: `"noReaction"`, `"laminar"`, `"PaSR"` or `"EDC"`, or a
            third-party model as a `pyfoamy.enums.CustomName`.
        coefficients: Model coefficients passed through, e.g.
            `{"Cmix": 1.0}` for `PaSR`; an `EDC` `version` is validated
            against `EDCVersion`.
        chemistry_type: `chemistryProperties` `type`: `"standard"`, or a
            third-party type as a `CustomName`, such as DLBFoam's `DLB`/
            `DLB_PYJAC` of this module. `DLB_PYJAC` expects a mechanism
            compiled via
            `pyfoamy.pyjac.generate.generate_pyjac_mechanism()`; this is
            not checked.
        chemistry_coefficients: Extra `chemistryProperties` entries (e.g.
            DLBFoam's `skipSpecies`), passed through.
        initial_chemical_time_step: `initialChemicalTimeStep`; omitted if
            `None`.
        ode: `ode` sub-dict; omitted if `None`.
        chemistry: The `chemistry` switch; omitted (on) if `None`.
        reduction: `reduction` sub-dict, e.g. `{"method": "DRG", ...}`;
            omitted (none) if `None`.
        tabulation: `tabulation` sub-dict, e.g. `{"method": "ISAT",
            ...}`; omitted (none) if `None`.
        skip_species: DLBFoam's `skipSpecies`; written only if true.
        skip_threshold: DLBFoam's `skipThreshold`; omitted if `None`.
        ref_mapping: DLBFoam's `refmapping` sub-dict; omitted if `None`.
        cell_zone: `cellZone` restricting the chemistry; omitted if
            `None`.
        loadbalancing: DLBFoam `loadbalancing` sub-dict (`active`/`log`);
            omitted if `None`; applies to either DLBFoam type.
        reactions: `reactions` sub-dict, e.g. from
            `ReactionConverter.reactions_dict()`; omitted if `None`.
        reactions_file: If `reactions` is given, written to
            `constant/<reactions_file>` and `#include`d, as real
            mechanisms can have thousands of reactions; `None` embeds
            inline. Unused when `reactions` is `None`.

    Raises:
        ValueError: If `model` is unsupported, `reactions` is given with
            `"noReaction"`, or `chemistry_coefficients` sets `"type"`
            (use `chemistry_type`).
    """
    model: ReactionModel
    coefficients: Optional[dict] = None
    chemistry_type: ChemistryType = ChemistryType.STANDARD
    chemistry_coefficients: Optional[dict] = None
    initial_chemical_time_step: Optional[float] = None
    ode: Optional[dict] = None
    chemistry: Optional[bool] = None
    reduction: Optional[dict] = None
    tabulation: Optional[dict] = None
    skip_species: bool = False
    skip_threshold: Optional[float] = None
    ref_mapping: Optional[dict] = None
    cell_zone: Optional[object] = None
    loadbalancing: Optional[dict] = None
    reactions: Optional[dict] = None
    reactions_file: Optional[str] = "reactions.dat"

    def __post_init__(self):
        self.model = ReactionModel.coerce(self.model, "reaction model")
        self.chemistry_type = ChemistryType.coerce(
            self.chemistry_type, "chemistry type"
        )
        if (
            self.model == ReactionModel.EDC
            and self.coefficients
            and "version" in self.coefficients
        ):
            self.coefficients = {
                **self.coefficients,
                "version": EDCVersion.coerce(
                    self.coefficients["version"], "EDC version"
                ),
            }
        if self.model == ReactionModel.NO_REACTION and self.reactions:
            raise ValueError(
                "ReactionSpec('noReaction', ...) doesn't use reactions - "
                "no chemistryProperties file is written at all"
            )
        if self.chemistry_coefficients and "type" in (
            self.chemistry_coefficients
        ):
            raise ValueError(
                "chemistry_coefficients can't set 'type' - use "
                "chemistry_type instead"
            )

    def required_libs(self) -> List[str]:
        """Returns the `controlDict` `libs` that `chemistry_type` needs.

        Empty for `"standard"` and for a `CustomName` other than `DLB`/
        `DLB_PYJAC`, whose library is the caller's. `libc_pyjac.so` is not
        included: it is case-local and `generate_pyjac_mechanism()`'s job.
        """
        if self.chemistry_type in (DLB, DLB_PYJAC):
            return list(_DLB_LIBS)
        return []

    def reaction_properties_dict(self) -> Optional[dict]:
        """Returns `reactionProperties` content, `None` for `"noReaction"`."""
        if self.model == ReactionModel.NO_REACTION:
            return None
        return {
            "reactionModel": self.model,
            **coefficients_entry(self.model, self.coefficients),
        }

    def chemistry_properties_dict(self) -> Optional[dict]:
        """Returns `chemistryProperties` content, `None` for `"noReaction"`."""
        if self.model == ReactionModel.NO_REACTION:
            return None
        result = {"type": self.chemistry_type}
        if self.chemistry is not None:
            result["chemistry"] = self.chemistry
        if self.initial_chemical_time_step is not None:
            result["initialChemicalTimeStep"] = (
                self.initial_chemical_time_step
            )
        if self.skip_species:
            result["skipSpecies"] = True
        if self.skip_threshold is not None:
            result["skipThreshold"] = self.skip_threshold
        for key, value in (
            ("ode", self.ode),
            ("reduction", self.reduction),
            ("tabulation", self.tabulation),
            ("loadbalancing", self.loadbalancing),
            ("refmapping", self.ref_mapping),
            ("cellZone", self.cell_zone),
        ):
            if value is not None:
                result[key] = value
        if self.reactions is not None and self.reactions_file is None:
            result["reactions"] = self.reactions
        if self.chemistry_coefficients:
            result.update(self.chemistry_coefficients)
        return result


class MulticomponentFluidSetup(CompressibleFluidSetup):
    """A multi-specie compressible flow setup.

    Species are caller-named: every specie, including `default_specie`
    (solved by subtraction), gets a `0/<name>` field.
    `physicalProperties` beyond `thermoType` is `defaultSpecie` plus one
    entry per specie. Species and enthalpy transport share one
    `div(phi,Yi_h)` scheme.

    Args:
        thermo_type: `thermoType` structure.
        default_specie: Key in `species`.
        species: Specie name -> `specie`/`thermodynamics`/`transport`
            data, passed through.
        reaction: Reaction model choice; default no reaction.
        turbulence: Turbulence model choice; default laminar.
        buoyant: Solve `p_rgh` and add `constant/g`.
        transonic: Add the transonic pressure scheme.
        species_file: File under `constant/` that `species` is written to
            and `physicalProperties` `#include`s, since a real mechanism
            is large; `None` embeds inline. `defaultSpecie` stays inline.
        pyjac_species_order: For `"loadBalanced_pyJac"`, the species order
            the compiled mechanism expects (the return value of
            `generate_pyjac_mechanism()`), checked against `species`.
            DLBFoam only compares the species count at run time, so a
            mismatch would silently evaluate the Jacobian for the wrong
            species.

    Raises:
        ValueError: If `default_specie` is not in `species`, or
            `pyjac_species_order` differs from `species`' order.
    """

    solver_name = "multicomponentFluid"

    def __init__(
        self,
        thermo_type: ThermoTypeSpec,
        default_specie: str,
        species: dict,
        reaction: Optional[ReactionSpec] = None,
        turbulence: Optional[TurbulenceSpec] = None,
        buoyant: bool = False,
        transonic: bool = False,
        species_file: Optional[str] = "species.dat",
        pyjac_species_order: Optional[List[str]] = None,
    ):
        if default_specie not in species:
            raise ValueError(
                f"default_specie {default_specie!r} must be a key in "
                "species"
            )
        if pyjac_species_order is not None and (
            list(species) != pyjac_species_order
        ):
            raise ValueError(
                "species order doesn't match the compiled pyJac "
                f"mechanism's own order - species={list(species)!r}, "
                f"pyjac_species_order={pyjac_species_order!r}"
            )
        super().__init__(thermo_type, turbulence, buoyant, transonic)
        self.default_specie = default_specie
        self.species = species
        self.reaction = reaction or ReactionSpec("noReaction")
        self.species_file = species_file

    def required_fields(self):
        """Returns the compressible fields plus every specie."""
        return super().required_fields() + list(self.species)

    def _species_content(self) -> dict:
        result = {"species": list(self.species)}
        result.update(self.species)
        return result

    def _physical_properties_extra(self) -> dict:
        result = {"defaultSpecie": self.default_specie}
        if self.species_file is None:
            result.update(self._species_content())
        return result

    def _split_physical_properties(self, case) -> None:
        if self.species_file is None:
            return
        case.add_dict(
            f"constant/{self.species_file}", self._species_content(),
            header=False,
        )
        append_include(
            case["constant/physicalProperties"], self.species_file,
        )

    def _momentum_group_fields(self):
        return (
            super()._momentum_group_fields()
            + ["Yi"]
            + list(self.species)
        )

    def _div_order_categories(self) -> dict:
        categories = dict(super()._div_order_categories())
        del categories[f"phi,{self.thermo_type.energy_field}"]
        categories["phi,Yi_h"] = "default"
        return categories

    def _fixed_div_terms(self) -> dict:
        terms = dict(super()._fixed_div_terms())
        terms["U"] = "Gauss linear"
        return terms

    def _constant_dicts(self, case, gravity) -> None:
        # Adds `reactionProperties`/`chemistryProperties` (and their `libs`)
        # unless `"noReaction"`.
        super()._constant_dicts(case, gravity)
        reaction_properties = self.reaction.reaction_properties_dict()
        if reaction_properties is not None:
            case["constant/reactionProperties"] = reaction_properties
        chemistry_properties = (
            self.reaction.chemistry_properties_dict()
        )
        if chemistry_properties is not None:
            case["constant/chemistryProperties"] = chemistry_properties
            case.add_libs(*self.reaction.required_libs())
            if (
                self.reaction.reactions is not None
                and self.reaction.reactions_file is not None
            ):
                case.add_dict(
                    f"constant/{self.reaction.reactions_file}",
                    {"reactions": self.reaction.reactions},
                    header=False,
                )
                append_include(
                    case["constant/chemistryProperties"],
                    self.reaction.reactions_file,
                )
