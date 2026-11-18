import pytest

from pyfoamy.enums import SimulationType
from pyfoamy.parsing.grammar import Dictionary


def test_str_returns_the_plain_value_not_the_member_name():
    assert str(SimulationType.RAS) == "RAS"


def test_coerce_accepts_a_plain_string():
    assert SimulationType.coerce("LES") is SimulationType.LES


def test_coerce_is_idempotent_on_an_existing_member():
    assert (
        SimulationType.coerce(SimulationType.LES) is SimulationType.LES
    )


def test_coerce_raises_with_supported_values_listed():
    with pytest.raises(ValueError, match="LES.*RAS.*laminar"):
        SimulationType.coerce("bogus", kind="simulation type")


def test_member_written_through_the_real_dictionary_writer():
    # to_grammar_value uses str(); needs OpenFoamEnum's __str__.
    d = Dictionary({"simulationType": SimulationType.RAS})
    reparsed = Dictionary(d.as_text)
    assert str(reparsed["simulationType"]) == "RAS"
