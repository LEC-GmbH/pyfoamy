"""Converts a `Mechanism` to the `records` data (no other Cantera use)."""

import math
from typing import List, Tuple

from pyfoamy.cantera.mechanism import Mechanism, require_cantera
from pyfoamy.pyjac.records import (
    ArrheniusParams, ChebyshevRate, GAS_CONSTANT_SI, PlogPoint, Reaction,
    STANDARD_PRESSURE, Species,
)


def _arrhenius_params(rate) -> ArrheniusParams:
    return (
        rate.pre_exponential_factor,
        rate.temperature_exponent,
        rate.activation_energy / GAS_CONSTANT_SI,
    )


def _third_body_kwargs(third_body, name_to_index) -> dict:
    # An explicit collision partner (e.g. gri30's "H + O2 + O2 <=> HO2 +
    # O2") is a Cantera `default_efficiency` of 0.0 plus that species at
    # 1.0; `pdep_species` represents it, also for non-falloff reactions.
    if third_body.name != "M":
        return {"pdep_species": name_to_index[third_body.name]}
    default = third_body.default_efficiency
    if default == 1.0:
        named = third_body.efficiencies.items()
    else:
        # The generated code adds (eff - 1) * C per listed species, so
        # every species needs its effective value.
        named = [
            (name, third_body.efficiencies.get(name, default))
            for name in name_to_index
        ]
    efficiencies = [
        (name_to_index[name], eff) for name, eff in named if eff != 1.0
    ]
    return {"third_body_efficiencies": efficiencies} if efficiencies else {}


def _convert_falloff(reaction, name_to_index) -> dict:
    rate = reaction.rate
    # Cantera's `low_rate`/`high_rate` are resolved whatever LOW/HIGH
    # keyword the CHEMKIN file used; high goes to A/b/E, low to
    # `low_rate_params` (the falloff formula is symmetric in this).
    A, b, E = _arrhenius_params(rate.high_rate)
    kwargs = {
        "A": A, "b": b, "E": E,
        "pressure_dependent": True,
        "low_rate_params": _arrhenius_params(rate.low_rate),
    }
    kwargs.update(_third_body_kwargs(reaction.third_body, name_to_index))
    if reaction.reaction_type == "falloff-Troe":
        kwargs["troe_params"] = tuple(float(x) for x in rate.falloff_coeffs)
    elif reaction.reaction_type == "falloff-SRI":
        kwargs["sri_params"] = tuple(float(x) for x in rate.falloff_coeffs)
    return kwargs


def _convert_reaction(reaction, name_to_index) -> Reaction:
    if reaction.orders:
        raise ValueError(
            f"reaction {reaction.equation!r} has non-default reaction "
            "orders - not supported (pyJac's own rate/Jacobian "
            "generation always assumes rate order == stoichiometric "
            "coefficient)"
        )

    reactants = [name_to_index[n] for n in reaction.reactants]
    reactant_nu = [float(nu) for nu in reaction.reactants.values()]
    products = [name_to_index[n] for n in reaction.products]
    product_nu = [float(nu) for nu in reaction.products.values()]

    kwargs = dict(
        reactants=reactants, reactant_nu=reactant_nu,
        products=products, product_nu=product_nu,
        reversible=reaction.reversible,
    )

    reaction_type = reaction.reaction_type
    if reaction_type == "Arrhenius":
        A, b, E = _arrhenius_params(reaction.rate)
        kwargs.update(A=A, b=b, E=E)
    elif reaction_type == "three-body-Arrhenius":
        A, b, E = _arrhenius_params(reaction.rate)
        kwargs.update(A=A, b=b, E=E, third_body=True)
        kwargs.update(
            _third_body_kwargs(reaction.third_body, name_to_index),
        )
    elif reaction_type in (
        "falloff-Lindemann", "falloff-Troe", "falloff-SRI",
    ):
        # Not `third_body`: pyJac's thd_body and pdep are mutually
        # exclusive for a falloff reaction.
        kwargs.update(_convert_falloff(reaction, name_to_index))
    elif reaction_type == "pressure-dependent-Arrhenius":
        plog: List[PlogPoint] = sorted(
            (
                (float(pressure), *_arrhenius_params(arrhenius))
                for pressure, arrhenius in reaction.rate.rates
            ),
            key=lambda point: point[0],
        )
        kwargs.update(A=0.0, b=0.0, E=0.0, plog=plog)
    elif reaction_type == "Chebyshev":
        rate = reaction.rate
        kwargs.update(
            A=0.0, b=0.0, E=0.0,
            chebyshev=ChebyshevRate(
                t_range=tuple(rate.temperature_range),
                p_range=tuple(rate.pressure_range),
                coefficients=[list(row) for row in rate.data],
            ),
        )
    else:
        raise ValueError(
            f"reaction {reaction.equation!r} has unsupported rate "
            f"type {reaction_type!r}"
        )

    # Cantera has no explicit-reverse (CHEMKIN REV) rate; ck2yaml splits
    # such reactions into two irreversible ones, so `reverse_params`
    # stays unset.
    return Reaction(**kwargs)


def _convert_species(sp) -> Species:
    import cantera as ct

    if not isinstance(sp.thermo, ct.NasaPoly2):
        model = sp.thermo.input_data.get("model", type(sp.thermo).__name__)
        raise ValueError(
            f"species {sp.name!r} has {model} thermo, pyJac needs NASA7"
        )
    coeffs = sp.thermo.coeffs  # [Tcommon, high a1..a7, low a1..a7]
    # The generated Kc assumes STANDARD_PRESSURE; rebasing the entropy
    # constant a7 leaves Cp and H unchanged.
    shift = math.log(sp.thermo.reference_pressure / STANDARD_PRESSURE)
    high = [float(x) for x in coeffs[1:8]]
    low = [float(x) for x in coeffs[8:15]]
    high[6] += shift
    low[6] += shift
    return Species(
        name=sp.name, mw=sp.molecular_weight,
        low=tuple(low), high=tuple(high), t_mid=float(coeffs[0]),
    )


def extract(
    mechanism: Mechanism, last_spec: str,
) -> Tuple[List[Species], List[Reaction]]:
    """Extracts species and reactions from `mechanism` as plain records.

    `last_spec` moves to the end (pyJac's dependent-species convention),
    the others keep their order.

    Args:
        last_spec: Typically an inert bath gas (N2, Ar, He); resolved
            case-insensitively, unknown names fail immediately.

    Raises:
        ValueError: If a species has no NASA7 thermo, or a reaction
            cannot be represented.
    """
    require_cantera()
    mechanism = Mechanism.coerce(mechanism)
    gas = mechanism.gas
    resolved_last_spec = mechanism.find_species(last_spec)

    ordered_names = [
        name for name in gas.species_names if name != resolved_last_spec
    ] + [resolved_last_spec]
    name_to_index = {name: i for i, name in enumerate(ordered_names)}

    species = [_convert_species(gas.species(name)) for name in ordered_names]

    reactions = [
        _convert_reaction(reaction, name_to_index)
        for reaction in gas.reactions()
    ]

    return species, reactions
