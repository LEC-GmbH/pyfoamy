"""Resolves abstract discretization "orders" into `fvSchemes` text."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Union

from pyfoamy.parsing.grammar import Compound, Dictionary

_DDT_SCHEMES = {
    "steady": "steadyState",
    "firstOrder": "Euler",
    "secondOrderBounded": "CrankNicolson 0.5",
    "secondOrderWeakBounded": "CrankNicolson 0.9",
    "secondOrder": "backward",
}

_SCALAR_DIV_SCHEMES = {
    "firstOrder": "upwind",
    "secondOrderBounded": "limitedLinear 1",
    "secondOrderWeakBounded": "linearUpwind GRAD",
    "secondOrder": "LUST GRAD",
    "01": "limitedLinear01 1",
}

_VECTOR_DIV_SCHEMES = {
    "firstOrder": "upwind",
    "secondOrderBounded": "limitedLinearV 1",
    "secondOrderWeakBounded": "linearUpwindV GRAD",
    "secondOrder": "LUST GRAD",
}


def tokenize_scheme(text: str) -> Compound:
    """Converts a space-separated scheme value into a `Compound`.

    Each piece is coerced on its own, since scheme text mixes words
    (`"grad(U)"`) and numbers (`"0.5"`).
    """
    return Compound(*text.split())


@dataclass
class CourantBlend:
    """A Courant-blended scheme, like OpenFOAM's `cellCoBlended`.

    `low_order` below `low_co`, `high_order` above `high_co`, blended
    between. Orders may themselves be `CourantBlend`s.

    Raises:
        ValueError: Unless `0 <= low_co < high_co`.
    """
    low_co: float
    low_order: "Order"
    high_co: float
    high_order: "Order"

    def __post_init__(self):
        if not (0 <= self.low_co < self.high_co):
            raise ValueError(
                "CourantBlend requires 0 <= low_co < high_co, got "
                f"low_co={self.low_co!r}, high_co={self.high_co!r}"
            )


@dataclass
class MultivariateSelection:
    """One scheme per field of a multivariate term such as
    `div(phi,Yi_h)`.

    Args:
        orders: Field name -> order name or literal scheme text.
    """
    orders: Dict[str, str]


# An order is either a recognized name/literal OpenFOAM text, or a
# CourantBlend of two such orders.
Order = Union[str, CourantBlend]

_FLUXES = ("phi,", "phiXi,", "alphaPhi,")


def unblended(term: str, order: Order) -> Order:
    """Returns the order `term` can take: `cellCoBlended` needs a mass or
    volumetric flux, so any other term (`phid,p`) gets the high-Courant
    order of a `CourantBlend`.

    Args:
        term: The `div(...)` argument, e.g. `"phi,U"`.
    """
    while isinstance(order, CourantBlend) and not term.startswith(_FLUXES):
        order = order.high_order
    return order


def resolve_ddt_scheme(order: str) -> Compound:
    """Resolves a `ddtSchemes` order into scheme text.

    Args:
        order: A name from `_DDT_SCHEMES`, or literal scheme text used
            verbatim.
    """
    return tokenize_scheme(_DDT_SCHEMES.get(order, order))


def resolve_div_scheme(
    field: str, order, vector: bool, gauss_prefix: bool = True,
    grad: Optional[str] = None,
) -> Compound:
    """Resolves one `div(phi,<field>)` value.

    Args:
        field: Names the gradient scheme a `linearUpwind`/`LUST` scheme
            uses, `grad(<field>)`.
        order: An order name, a `CourantBlend`, a `MultivariateSelection`
            or literal scheme text.
        vector: Resolve against the vector rather than scalar table.
        gauss_prefix: Prepend `Gauss`; `False` only for a `CourantBlend`'s
            nested schemes, since OpenFOAM has one `Gauss` per entry.
        grad: The gradient scheme's name instead of `grad(<field>)`, e.g.
            `"default"`.

    Returns:
        A `Compound`; a `CourantBlend`'s sub-schemes nest as `Compound`s,
        which render inline.
    """
    prefix = ["Gauss"] if gauss_prefix else []
    if isinstance(order, MultivariateSelection):
        return Compound(*prefix, "multivariateSelection", Dictionary({
            name: resolve_div_scheme(
                name, one, vector, gauss_prefix=False, grad=grad,
            )
            for name, one in order.orders.items()
        }))
    if isinstance(order, CourantBlend):
        low = resolve_div_scheme(
            field, order.low_order, vector, gauss_prefix=False, grad=grad,
        )
        high = resolve_div_scheme(
            field, order.high_order, vector, gauss_prefix=False, grad=grad,
        )
        return Compound(
            *prefix,
            "cellCoBlended",
            Decimal(str(order.low_co)),
            low,
            Decimal(str(order.high_co)),
            high,
        )

    table = _VECTOR_DIV_SCHEMES if vector else _SCALAR_DIV_SCHEMES
    if order in table:
        text = table[order].replace("GRAD", grad or f"grad({field})")
    else:
        text = order
    return Compound(*prefix, tokenize_scheme(text))
