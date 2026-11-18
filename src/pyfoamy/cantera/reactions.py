"""Converts a `Mechanism`'s Cantera reactions to OpenFOAM dicts."""

from dataclasses import dataclass
from typing import Any, Dict, List

from pyfoamy.cantera.mechanism import Mechanism, ct, require_cantera
from pyfoamy.openfoam_version import openfoam_version
from pyfoamy.parsing.grammar import String

# Cantera rate type -> OpenFOAM rate name: 14 appends it to "reversible"
# (makeReaction.H), dev wraps it, "Reversible<...>" (makeRateReaction.H).
_REACTION_TYPE_SUFFIXES = {
    "Arrhenius": ("Arrhenius", "Arrhenius"),
    "three-body-Arrhenius": ("ThirdBodyArrhenius", "ThirdBody<Arrhenius>"),
    "pressure-dependent-Arrhenius": ("PLog", "pLog"),
    "falloff-Lindemann": (
        "ArrheniusLindemannFallOff", "FallOff<Arrhenius,Lindemann>",
    ),
    "falloff-Troe": ("ArrheniusTroeFallOff", "FallOff<Arrhenius,Troe>"),
    "falloff-SRI": ("ArrheniusSRIFallOff", "FallOff<Arrhenius,SRI>"),
}


def _reaction_type(rtype: str, reversible: bool) -> str:
    prefix = "reversible" if reversible else "irreversible"
    v14, dev = _REACTION_TYPE_SUFFIXES[rtype]
    if openfoam_version() == "dev":
        return f"{prefix.capitalize()}<{dev}>"
    return f"{prefix}{v14}"


def _arrhenius_dict(
    rate: "ct.ArrheniusRate", rate_multiplier: float,
) -> Dict[str, float]:
    return {
        "A": rate.pre_exponential_factor*rate_multiplier,
        "beta": rate.temperature_exponent,
        "Ea": rate.activation_energy,
    }


def _plog_dict(
    rate: "ct.PlogRate", rate_multiplier: float,
) -> Dict[str, Any]:
    # pLog takes one Arrhenius set per pressure, with Ta [K] = Ea/R.
    pressures = [p for p, _ in rate.rates]
    if len(set(pressures)) != len(pressures):
        raise ValueError("PLOG rate with duplicate pressures")
    return {
        "coeffs": [
            [
                p,
                arrhenius.pre_exponential_factor*rate_multiplier,
                arrhenius.temperature_exponent,
                arrhenius.activation_energy/ct.gas_constant,
            ]
            for p, arrhenius in sorted(rate.rates, key=lambda r: r[0])
        ],
    }


def _third_body_dict(
    gas: "ct.Solution", third_body: "ct.ThirdBody",
) -> Dict[str, Any]:
    if openfoam_version() == "14":
        # Release 14 reads all species' efficiencies from coeffs, or the
        # default alone (any other entry is ignored); dev rejects coeffs.
        return {
            "coeffs": [
                [name, third_body.efficiency(name)]
                for name in gas.species_names
            ],
        }
    default = third_body.default_efficiency
    efficiencies = [
        [name, third_body.efficiency(name)] for name in gas.species_names
        if third_body.efficiency(name) != default
    ]
    result: Dict[str, Any] = {"defaultEfficiency": default}
    if efficiencies:
        result["efficiencies"] = efficiencies
    return result


def _falloff_dict(
    reaction: "ct.Reaction", rate_multiplier: float,
) -> Dict[str, Any]:
    rate = reaction.rate
    result: Dict[str, Any] = {
        "k0": _arrhenius_dict(rate.low_rate, rate_multiplier),
        "kInf": _arrhenius_dict(rate.high_rate, rate_multiplier),
        "F": {},
    }
    coeffs = rate.falloff_coeffs
    if reaction.reaction_type == "falloff-Troe":
        result["F"]["alpha"] = float(coeffs[0])
        result["F"]["Tsss"] = max(float(coeffs[1]), 1e-10)
        result["F"]["Ts"] = max(float(coeffs[2]), 1e-10)
        result["F"]["Tss"] = (
            min(float(coeffs[3]), 1e20) if len(coeffs) > 3 else 1e20
        )
    elif reaction.reaction_type == "falloff-SRI":
        for letter, value in zip("abcde", coeffs):
            result["F"][letter] = float(value)
    return result


def _reaction_equation(reaction: "ct.Reaction") -> str:
    if reaction.orders:
        raise ValueError(
            "reaction has non-default reaction orders - not supported "
            "(would need OpenFOAM's per-specie '^exponent' equation "
            "syntax)"
        )
    reactants = reaction.reactant_string
    products = reaction.product_string
    if reaction.third_body is not None:
        # The collider (M or an explicit species such as AR) is
        # carried by the third-body efficiencies alone - leaving an
        # explicit collider in the equation would count its
        # concentration twice.
        name = reaction.third_body.name
        if reaction.reaction_type.startswith("falloff-"):
            suffix = f" (+{name})"
        else:
            suffix = f" + {name}"
        if not (reactants.endswith(suffix) and products.endswith(suffix)):
            raise ValueError(
                f"could not locate third-body collider {suffix.strip()!r} "
                "at the end of both sides of the reaction equation"
            )
        reactants = reactants[:-len(suffix)]
        products = products[:-len(suffix)]
    return f"{reactants} = {products}"


@dataclass
class ReactionConverter:
    """Converts a `Mechanism`'s reactions to OpenFOAM `reactions` dicts.

    Always converts the full mechanism: a reaction network cannot be
    restricted to a species subset.

    Attributes:
        mechanism: A mechanism path/name or a `Mechanism` (reused as
            is).
    """

    mechanism: Mechanism

    def __post_init__(self) -> None:
        require_cantera()
        self.mechanism = Mechanism.coerce(self.mechanism)

    def reaction_dict(
        self, reaction: "ct.Reaction", rate_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """Converts one Cantera reaction to its OpenFOAM dict form.

        Type names and the third-body form follow `openfoam_version()`.
        OpenFOAM-14 reads PLOG (`pLog`) only with `libpyfoamy.so` loaded.

        Args:
            reaction: A reaction of `mechanism.gas.reactions()`.
            rate_multiplier: Factor on every pre-exponential factor `A`
                (PLOG and falloff `k0`/`kInf` included), scaling the rate
                by exactly this factor; a reverse rate follows through
                the equilibrium constant.

        Returns:
            A dict with `type`/`reaction` plus the rate, third-body and
            falloff keys the type needs.

        Raises:
            ValueError: If the type is unsupported (e.g. PLOG with a
                third body), orders are non-default, or a PLOG has
                duplicate pressures (not representable by `pLog`).
        """
        rtype = reaction.reaction_type
        if rtype not in _REACTION_TYPE_SUFFIXES:
            raise ValueError(
                f"reaction type {rtype!r} is not supported - supported "
                f"types are {sorted(_REACTION_TYPE_SUFFIXES)}"
            )
        # Validated before touching .reversible/gas: fail fast, and
        # messages stay reaction-independent so reactions_dict() can
        # group repeats.
        equation = _reaction_equation(reaction)

        gas = self.mechanism.gas
        result: Dict[str, Any] = {
            "type": _reaction_type(rtype, reaction.reversible),
            # Always a quoted String, as OpenFOAM writes this keyword
            # (simplifiedSiwek reactions), not left to the bare-word
            # check of value_to_tokens().
            "reaction": String(equation),
        }

        if rtype in ("Arrhenius", "three-body-Arrhenius"):
            result.update(_arrhenius_dict(reaction.rate, rate_multiplier))
        elif rtype == "pressure-dependent-Arrhenius":
            result.update(_plog_dict(reaction.rate, rate_multiplier))
        elif rtype.startswith("falloff-"):
            result.update(_falloff_dict(reaction, rate_multiplier))

        if rtype == "three-body-Arrhenius":
            result.update(_third_body_dict(gas, reaction.third_body))
        elif rtype.startswith("falloff-"):
            result["thirdBodyEfficiencies"] = _third_body_dict(
                gas, reaction.third_body,
            )

        return result

    def reactions_dict(
        self, rate_multiplier: float = 1.0,
    ) -> Dict[str, Dict[str, Any]]:
        """Converts every reaction to `{f"reaction{i}": reaction_dict}`.

        Args:
            rate_multiplier: Factor on every rate, as in `reaction_dict()`.

        Raises:
            ValueError: Naming every distinct conversion failure with
                its count, not just the first.
        """
        gas = self.mechanism.gas
        result: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, List[str]] = {}
        for i, reaction in enumerate(gas.reactions()):
            try:
                result[f"reaction{i}"] = self.reaction_dict(
                    reaction, rate_multiplier,
                )
            except ValueError as exc:
                errors.setdefault(str(exc), []).append(reaction.equation)

        if errors:
            raise ValueError(
                "could not convert every reaction:\n" + "\n".join(
                    f"{message} ({len(equations)} reaction(s), e.g. "
                    f"{equations[0]!r})"
                    for message, equations in errors.items()
                )
            )
        return result
