import pytest

from pyfoamy.models.schemes import (
    CourantBlend,
    MultivariateSelection,
    resolve_ddt_scheme,
    resolve_div_scheme,
    tokenize_scheme,
    unblended,
)


def _words(scheme):
    # Compounds may nest (CourantBlend); str() flattens them.
    return str(scheme).split()


def test_tokenize_scheme_splits_on_whitespace():
    assert _words(tokenize_scheme("Gauss limitedLinear 1")) == [
        "Gauss", "limitedLinear", "1",
    ]


@pytest.mark.parametrize(
    "order,expected",
    [
        ("steady", ["steadyState"]),
        ("firstOrder", ["Euler"]),
        ("secondOrderBounded", ["CrankNicolson", "0.5"]),
        ("secondOrderWeakBounded", ["CrankNicolson", "0.9"]),
        ("secondOrder", ["backward"]),
    ],
)
def test_resolve_ddt_scheme_known_orders(order, expected):
    assert _words(resolve_ddt_scheme(order)) == expected


def test_resolve_ddt_scheme_passes_through_literal_text():
    assert _words(resolve_ddt_scheme("CrankNicolson 0.7")) == [
        "CrankNicolson", "0.7",
    ]


@pytest.mark.parametrize(
    "order,vector,expected",
    [
        ("firstOrder", False, ["Gauss", "upwind"]),
        ("secondOrderBounded", False, ["Gauss", "limitedLinear", "1"]),
        (
            "secondOrderWeakBounded", False,
            ["Gauss", "linearUpwind", "grad(k)"],
        ),
        ("secondOrder", False, ["Gauss", "LUST", "grad(k)"]),
        ("01", False, ["Gauss", "limitedLinear01", "1"]),
        ("firstOrder", True, ["Gauss", "upwind"]),
        (
            "secondOrderBounded", True,
            ["Gauss", "limitedLinearV", "1"],
        ),
        (
            "secondOrderWeakBounded", True,
            ["Gauss", "linearUpwindV", "grad(k)"],
        ),
    ],
)
def test_resolve_div_scheme_known_orders(order, vector, expected):
    assert _words(resolve_div_scheme("k", order, vector)) == expected


def test_resolve_div_scheme_passes_through_literal_text():
    result = resolve_div_scheme("U", "linearUpwind grad(U)", vector=True)
    assert _words(result) == ["Gauss", "linearUpwind", "grad(U)"]


def test_courant_blend_requires_low_co_less_than_high_co():
    with pytest.raises(ValueError, match="low_co"):
        CourantBlend(5, "firstOrder", 2, "secondOrder")


def test_courant_blend_requires_non_negative_low_co():
    with pytest.raises(ValueError, match="low_co"):
        CourantBlend(-1, "firstOrder", 2, "secondOrder")


def test_resolve_div_scheme_matches_real_tutorial_syntax():
    # tutorials/incompressibleVoF/planingHullW3/system/fvSchemes:
    # div(rhoPhi,U) Gauss cellCoBlended 2 linearUpwind grad(U) 5 upwind;
    blend = CourantBlend(
        2, "linearUpwind grad(U)", 5, "upwind"
    )
    result = resolve_div_scheme("U", blend, vector=True)
    assert _words(result) == [
        "Gauss", "cellCoBlended", "2", "linearUpwind", "grad(U)",
        "5", "upwind",
    ]


def test_resolve_div_scheme_courant_blend_uses_order_tables():
    blend = CourantBlend(
        2, "secondOrderWeakBounded", 5, "firstOrder"
    )
    result = resolve_div_scheme("U", blend, vector=True)
    assert _words(result) == [
        "Gauss", "cellCoBlended", "2", "linearUpwindV", "grad(U)",
        "5", "upwind",
    ]


def test_resolve_div_scheme_courant_blend_nests_recursively():
    inner = CourantBlend(5, "firstOrder", 10, "secondOrder")
    outer = CourantBlend(1, "secondOrderBounded", 5, inner)
    result = resolve_div_scheme("U", outer, vector=False)
    assert _words(result) == [
        "Gauss", "cellCoBlended", "1", "limitedLinear", "1",
        "5", "cellCoBlended", "5", "upwind", "10", "LUST", "grad(U)",
    ]


def test_multivariate_selection_resolves_each_field():
    scheme = resolve_div_scheme(
        "Yi_h",
        MultivariateSelection({"h": "secondOrderBounded", "O2": "01"}),
        vector=False,
    )
    assert str(scheme) == (
        "Gauss multivariateSelection "
        "{ h limitedLinear 1; O2 limitedLinear01 1; }"
    )


def test_grad_names_the_gradient_scheme():
    assert _words(resolve_div_scheme(
        "h", "secondOrderWeakBounded", vector=False, grad="default",
    )) == ["Gauss", "linearUpwind", "default"]


def test_only_a_flux_term_keeps_a_courant_blend():
    blend = CourantBlend(1, "secondOrderBounded", 2, "firstOrder")
    assert unblended("phi,U", blend) is blend
    assert unblended("alphaPhi,K", blend) is blend
    assert unblended("phid,p", blend) == "firstOrder"
    assert unblended("phid,p", "secondOrder") == "secondOrder"
