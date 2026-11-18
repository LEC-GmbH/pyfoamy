import subprocess
from decimal import Decimal
from pathlib import Path

import pyparsing as pp
import pytest

from pyfoamy.parsing.grammar import (
    AnonymousListEntry,
    Arguments,
    Bracket,
    Code,
    Compound,
    Dictionary,
    Directive,
    KeywordEntry,
    Label,
    List,
    Number,
    ParseError,
    String,
    Switch,
    Uniform,
    UnitNumber,
    Variable,
    Word,
    to_grammar_value,
    _value_unit,
)


def _foam_dictionary_expand(tmp_path, text):
    path = tmp_path / "testDict"
    header = (
        "FoamFile\n{\n"
        "    version 2.0;\n    format ascii;\n"
        "    class dictionary;\n    object testDict;\n}\n\n"
    )
    path.write_text(header + text)
    return subprocess.run(
        ["foamDictionary", "-expand", str(path)],
        capture_output=True, text=True,
    )


def _foam_dictionary(tmp_path, text):
    # No -expand: directives are only checked, not evaluated.
    path = tmp_path / "testDict"
    header = (
        "FoamFile\n{\n"
        "    version 2.0;\n    format ascii;\n"
        "    class dictionary;\n    object testDict;\n}\n\n"
    )
    path.write_text(header + text)
    return subprocess.run(
        ["foamDictionary", str(path)], capture_output=True, text=True,
    )


def parse_value(text):
    return _value_unit.parse_string(text, parse_all=True)[0]


# -- Word ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("linear", "linear"),
        ("grad(U)", "grad(U)"),
        ("div(phi,U)", "div(phi,U)"),
        ("List<scalar>", "List<scalar>"),
        ("grad(div(phi,U))", "grad(div(phi,U))"),
    ],
)
def test_word_accepts_balanced_parens(text, expected):
    assert str(Word(text)) == expected


@pytest.mark.parametrize("text", ["123asdf", "-5", ".5", "5e3", '"x"', ""])
def test_word_rejects_non_word_text(text):
    with pytest.raises(ValueError):
        Word(text)


def test_word_rejects_unbalanced_paren():
    with pytest.raises(ValueError):
        Word("(bad")


def test_word_leading_plus_is_a_word_not_a_number():
    # '+' never starts a number in OpenFOAM, only '-', '.' or a digit.
    assert str(Word("+5")) == "+5"


def test_word_list_boundary_not_swallowed_by_embedded_parens():
    # The word's ")" must not swallow the enclosing list's.
    result = _value_unit.parse_string("(grad(p) grad(U))", parse_all=True)
    assert list(result[0]) == [Word("grad(p)"), Word("grad(U)")]


def test_word_equality_and_hash_are_type_strict():
    assert Word("x") == Word("x")
    assert Word("x") != "x"
    assert Word("x") != String("x")
    assert len({Word("x"), String("x"), Word("x")}) == 2


def test_word_userstring_ergonomics():
    w = Word("linear")
    assert str(w[0:3]) == "lin"
    assert "lin" in w
    assert isinstance(w + "X", Word)


# -- Switch ------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["true", "on", "yes"])
def test_switch_true_spellings(spelling):
    assert bool(Switch(spelling)) is True
    assert str(Switch(spelling)) == spelling


@pytest.mark.parametrize("spelling", ["false", "off", "no"])
def test_switch_false_spellings(spelling):
    assert bool(Switch(spelling)) is False
    assert str(Switch(spelling)) == spelling


def test_switch_from_bool_round_trips_as_true_false():
    assert str(Switch(True)) == "true"
    assert str(Switch(False)) == "false"


def test_switch_equality_ignores_spelling():
    assert Switch("on") == Switch("yes")
    assert Switch("on") == True  # noqa: E712
    assert Switch("off") != Switch("on")


def test_switch_rejects_unrecognized_spelling():
    with pytest.raises(ValueError):
        Switch("none")
    with pytest.raises(TypeError):
        Switch(1)


def test_word_as_switch():
    d = Dictionary("adjustTimeStep no;\n")
    assert d["adjustTimeStep"].as_switch() == Switch("no")
    assert bool(d["adjustTimeStep"].as_switch()) is False


def test_word_as_switch_rejects_non_switch_word():
    with pytest.raises(ValueError):
        Word("none").as_switch()


def test_parsing_never_reinterprets_switch_words_automatically():
    # Text cannot tell a boolean from an identifier ("type none;").
    d = Dictionary("adjustTimeStep no;\n")
    assert isinstance(d["adjustTimeStep"], Word)
    assert not isinstance(d["adjustTimeStep"], Switch)


# -- String ----------------------------------------------------------------


def test_string_renders_quoted():
    assert str(String("foo bar")) == '"foo bar"'


def test_string_coerces_via_str():
    assert str(String(5)) == '"5"'


def test_string_allows_empty():
    # Valid as a value (`dataFile "";` in decomposeParDict), not as a
    # key; see test_dictionary_rejects_empty_key.
    assert str(String("")) == '""'


def test_string_backslash_newline_continues_the_string():
    d = Dictionary('a "foo\\\nbar";\n')
    assert d["a"] == String("foo\nbar")
    assert d.as_text.endswith('"foo\\\nbar";')


def test_string_bare_newline_is_rejected():
    with pytest.raises(ParseError):
        Dictionary('a "foo\nbar";\n')


# -- Label / Number --------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_type,expected",
    [
        ("5", Label, 5),
        ("-5", Label, -5),
        ("5.0", Number, Decimal("5")),
        ("5e3", Number, Decimal("5000")),
        (".5", Number, Decimal("0.5")),
        ("-1.5E-10", Number, Decimal("-1.5E-10")),
    ],
)
def test_number_dispatch(text, expected_type, expected):
    result = parse_value(text)
    assert isinstance(result, expected_type)
    assert result == expected


@pytest.mark.parametrize("text", ["inf", "-inf", "nan", "-nan"])
def test_number_non_finite_round_trips(text):
    result = parse_value(text)
    assert isinstance(result, Number)
    assert str(result) == text


def test_word_starting_with_inf_or_nan_stays_a_word():
    assert parse_value("info") == Word("info")
    assert parse_value("nanometer") == Word("nanometer")


def test_decimal_prefix_does_not_win_as_label():
    # Label must not match the "150" of "150.0".
    result = parse_value("150.0")
    assert isinstance(result, Number)
    assert result == Decimal("150")


@pytest.mark.parametrize("cls", [Label, Number])
def test_leading_plus_rejected(cls):
    with pytest.raises(ValueError):
        cls("+5")


@pytest.mark.parametrize(
    "text,expected",
    [("160.0", "160"), ("160.", "160"), ("160.000", "160"),
     ("1.30", "1.3"), ("1.50", "1.5"), ("0.0", "0"), ("-5.0", "-5")],
)
def test_number_strips_trailing_zeros(text, expected):
    assert str(Number(text)) == expected


def test_number_preserves_non_whole_scientific_form():
    assert str(Number("-1.5E-10")) == "-1.5E-10"


def test_number_huge_value_falls_back_to_normalize():
    assert str(Number("1E+21")) == "1E+21"


def test_label_is_a_real_int():
    assert Label(5) + 1 == 6
    assert isinstance(Label(5) + 1, int)
    assert Label(5) == 5


def test_number_is_a_real_decimal():
    assert Number("1.30") == Decimal("1.30")


def test_label_rejects_decimal_text():
    with pytest.raises(ValueError):
        Label("150.0")


# -- Label/Number word-continuation ambiguity (split=) ----------------------


@pytest.mark.parametrize("text", ["123dfasdf", "1.5dfasdf"])
def test_digit_word_adjacency_rejected_by_default(text):
    with pytest.raises(pp.ParseException):
        _value_unit.parse_string(text, parse_all=True)


def test_digit_word_adjacency_splits_with_split_true():
    # foamDictionary reads "123dfasdf" as two tokens; split=True does
    # the same, the default is stricter.
    d = Dictionary("b 123dfasdf;", split=True)
    assert list(d["b"]) == [Label(123), Word("dfasdf")]


def test_digit_next_to_paren_never_needs_split():
    result = parse_value("(4(1 2 3 4))")
    assert list(result) == [
        Label(4), List([Label(1), Label(2), Label(3), Label(4)]),
    ]


# -- Variable --------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["$internalField", "$foo(bar)", "${../foo}", "${a${b}c}"],
)
def test_variable_accepts_real_forms(text):
    assert str(Variable(text)) == text


@pytest.mark.parametrize("text", ["$", "$;"])
def test_variable_rejects_bare_dollar(text):
    # Fatal in OpenFOAM (ISstream::readVariable).
    with pytest.raises(ValueError):
        Variable(text)


def test_variable_embedded_in_list_keeps_boundary():
    result = _value_unit.parse_string("(x $foo(bar) y)", parse_all=True)[0]
    assert list(result) == [Word("x"), Variable("$foo(bar)"), Word("y")]


# -- Code --------------------------------------------------------------


def test_code_round_trips():
    assert str(Code("some code;")) == "#{some code;#}"


def test_code_allows_empty():
    assert str(Code("")) == "#{#}"


def test_code_tolerates_lone_hash():
    text = "#{\n// a comment with # in it\n#}"
    result = Code.grammar.parse_string(text, parse_all=True)
    assert "# in it" in result[0]


# -- Bracket -------------------------------------------------------------


def test_bracket_round_trips():
    assert str(Bracket("bar")) == "[bar]"


@pytest.mark.parametrize("text", [
    "units [mm];\n",
    "dimensions [0 1 -1 0 0 0 0];\n",
    "dimensions [pressure];\n",
    "x [Pa s];\n",
    "y [kg/m^3];\n",
    "z [length*velocity];\n",
])
def test_bracket_standalone_shapes(text):
    key = text.split()[0]
    d = Dictionary(text)
    assert isinstance(d[key], Bracket)


def test_bracket_in_a_list():
    d = Dictionary("units ([mm] [g/CAD]);\n")
    assert list(d["units"]) == [Bracket("mm"), Bracket("g/CAD")]


def test_bracket_after_a_number():
    # "pressure 5 [bar];"
    d = Dictionary("pressure 5 [bar];\n")
    assert d["pressure"] == UnitNumber(5, "bar")


def test_bracket_fixes_slash_in_unit_content():
    # "/" is a word delimiter.
    d = Dictionary("supplyMassFlowRate 0.1 [g/s];\n")
    assert d["supplyMassFlowRate"] == UnitNumber("0.1", "g/s")


def test_bracket_round_trip_is_stable():
    text = "units ([mm] [g/CAD]);\npressure 5 [bar];\n"
    first = Dictionary(text).as_text
    second = Dictionary(first).as_text
    assert first == second


def test_bracket_in_a_list_stays_a_bare_bracket():
    result = parse_value("([mm] [g/CAD])")
    assert list(result) == [Bracket("mm"), Bracket("g/CAD")]


# -- UnitNumber ----------------------------------------------------------


def test_unit_number_round_trips():
    assert str(UnitNumber(5, "bar")) == "5 [bar]"
    assert str(UnitNumber("0.1", "g/s")) == "0.1 [g/s]"


def test_unit_number_add_and_sub_same_unit():
    a = UnitNumber(5, "bar")
    assert a + UnitNumber(2, "bar") == UnitNumber(7, "bar")
    assert a - UnitNumber(2, "bar") == UnitNumber(3, "bar")


def test_unit_number_arithmetic_with_a_bare_number_keeps_the_unit():
    a = UnitNumber(5, "bar")
    assert a + 1 == UnitNumber(6, "bar")
    assert 1 + a == UnitNumber(6, "bar")
    assert a - 1 == UnitNumber(4, "bar")
    assert 10 - a == UnitNumber(5, "bar")


def test_unit_number_scales_by_a_bare_number():
    a = UnitNumber(5, "bar")
    assert a * 2 == UnitNumber(10, "bar")
    assert 2 * a == UnitNumber(10, "bar")
    assert a / 2 == UnitNumber(Decimal("2.5"), "bar")


def test_unit_number_rejects_mismatched_units():
    with pytest.raises(ValueError):
        UnitNumber(5, "bar") + UnitNumber(1, "Pa")


def test_unit_number_rejects_multiplying_two_unit_numbers():
    with pytest.raises(TypeError):
        UnitNumber(5, "bar") * UnitNumber(2, "bar")


def test_number_format_matches_str_not_raw_decimal():
    # Decimal.__format__ bypasses the __str__ override.
    assert f"{Number('160.0')}" == "160"
    assert f"{UnitNumber('160.0', 'bar')}" == "160 [bar]"


# -- List --------------------------------------------------------------


def test_list_plain():
    result = parse_value("(1 2 3)")
    assert list(result) == [Label(1), Label(2), Label(3)]
    assert str(result) == "(1 2 3)"


def test_list_typed_header_count_is_not_stored_but_type_name_is():
    # The count is not stored, the type name is (field files need it).
    result = parse_value("List<scalar> 4(1 2 3 4)")
    assert list(result) == [Label(1), Label(2), Label(3), Label(4)]
    assert result.count is None
    assert result.type_name == "scalar"
    assert str(result) == "List<scalar> 4(1 2 3 4)"


def test_list_typed_header_without_count_round_trips():
    result = parse_value("List<scalar>(1 2 3 4)")
    assert result.type_name == "scalar"
    assert str(result) == "List<scalar> 4(1 2 3 4)"


def test_list_without_type_name_never_gains_a_header():
    result = parse_value("(1 2 3 4)")
    assert result.type_name is None
    assert str(result) == "(1 2 3 4)"
    assert result.__str__(leading_count=True) == "4(1 2 3 4)"


def test_bare_leading_count_is_never_collapsed_without_a_header():
    # A bare count is ambiguous with a separate value; the tutorials
    # always pair it with a List<Type> header.
    result = parse_value("(4(1 2 3 4))")
    assert list(result) == [
        Label(4), List([Label(1), Label(2), Label(3), Label(4)]),
    ]


def test_list_uniform_shorthand_requires_count():
    result = parse_value("4{0}")
    assert result.uniform and result.count == 4
    assert str(result) == "4{0}"
    with pytest.raises(ValueError):
        List([0], uniform=True)
    with pytest.raises(ValueError):
        List([0], count=4)


def test_list_uniform_shorthand_with_typed_header_round_trips():
    result = parse_value("List<scalar> 4{0}")
    assert result.uniform and result.count == 4 and result.type_name == "scalar"
    assert str(result) == "List<scalar> 4{0}"


def test_list_nested():
    result = parse_value("(1 (2 3) 4)")
    assert list(result) == [Label(1), List([Label(2), Label(3)]), Label(4)]


def test_leading_scalar_before_nested_list_is_not_swallowed_as_count():
    # "1 (2 3)": the count mismatch rejects the counted-list reading.
    result = parse_value("(1 (2 3))")
    assert list(result) == [Label(1), List([Label(2), Label(3)])]

    matching = parse_value("(2(1 2))")
    assert list(matching) == [Label(2), List([Label(1), Label(2)])]


def test_bare_counts_never_collapse_sublists_within_a_list():
    d = Dictionary("a ( 2 ( 1 2 ) 4 ( 1 asdf 4 3) );")
    assert list(d["a"]) == [
        Label(2),
        List([Label(1), Label(2)]),
        Label(4),
        List([Label(1), Word("asdf"), Label(4), Label(3)]),
    ]


def test_bare_dict_as_list_item():
    result = parse_value("(1 { a 2; } 3)")
    assert result[0] == Label(1)
    assert result[1]["a"] == Label(2)
    assert result[2] == Label(3)


def test_name_subdict_pair_in_list_is_two_items_not_one():
    # In a value, "name {...}" is two items, not an entry.
    result = parse_value("(asdf { a b; })")
    assert len(result) == 2
    assert result[0] == Word("asdf")
    assert result[1]["a"] == Word("b")


def test_list_from_native_and_text_mix():
    result = List(["a", -1, "{ c -1; }", "ds", 1.5e2])
    assert result[0] == Word("a")
    assert result[1] == Label(-1)
    assert result[2]["c"] == Label(-1)
    assert result[3] == Word("ds")
    assert result[4] == Number("150")


def test_list_varargs_and_list_arg_construction_agree():
    assert List(1, 2, 3) == List([1, 2, 3])


def test_list_missing_semicolon_before_close_brace_is_fatal():
    # Fatal for a top-level entry in OpenFOAM too.
    with pytest.raises(Exception):
        parse_value("({ a 1 })")


# -- Compound ------------------------------------------------------------


def test_compound_renders_space_joined_no_brackets():
    from pyfoamy.parsing.grammar import Compound
    c = Compound("Gauss", "limitedLinear01", 1)
    assert str(c) == "Gauss limitedLinear01 1"
    assert list(c) == [Word("Gauss"), Word("limitedLinear01"), Label(1)]


def test_compound_has_no_uniform_shorthand():
    from pyfoamy.parsing.grammar import Compound
    assert not hasattr(Compound, "grammar")


# -- Uniform ---------------------------------------------------------------


def test_uniform_renders_scalar():
    assert str(Uniform(0)) == "uniform 0"


def test_uniform_renders_sequence():
    assert str(Uniform([0, 1, 0])) == "uniform (0 1 0)"


def test_uniform_is_construction_side_only():
    # Parsing gives a plain Compound; Uniform is for writing.
    d = Dictionary("a uniform 0;")
    assert not isinstance(d["a"], Uniform)
    assert d["a"] == Compound(Word("uniform"), Label(0))


def test_uniform_value_is_mutable_when_shape_matches():
    u = Uniform(1)
    u.value = 5
    assert str(u) == "uniform 5"

    v = Uniform([1, 2, 3])
    v.value = [4, 5, 6]
    assert str(v) == "uniform (4 5 6)"


def test_uniform_value_assignment_rejects_scalar_to_sequence():
    u = Uniform(1)
    with pytest.raises(TypeError):
        u.value = [1, 2, 3]


def test_uniform_value_assignment_rejects_sequence_to_scalar():
    u = Uniform([1, 2, 3])
    with pytest.raises(TypeError):
        u.value = 1


def test_uniform_value_assignment_rejects_length_change():
    u = Uniform([1, 2, 3])
    with pytest.raises(ValueError):
        u.value = [1, 2]


def test_uniform_equality_compares_value():
    assert Uniform(0) == Uniform(0)
    assert Uniform(0) != Uniform(1)
    assert Uniform([0, 1, 0]) == Uniform([0, 1, 0])
    assert Uniform(0) != 0


def test_uniform_is_a_compound_and_equals_the_bare_parsed_form():
    assert isinstance(Uniform(0), Compound)
    assert Uniform(0) == Compound(Word("uniform"), Label(0))
    d = Dictionary("a uniform 0;")
    assert d["a"] == Uniform(0)


def test_uniform_value_assignment_checks_shape_after_coercion():
    # The shape check is on the parsed value, not the Python type.
    u = Uniform(0)
    with pytest.raises(TypeError):
        u.value = "(0 1 0)"


def test_uniform_is_unhashable():
    with pytest.raises(TypeError):
        hash(Uniform(0))


def test_uniform_set_on_dictionary_round_trips():
    d = Dictionary()
    d.set("internalField", Uniform(0))
    assert d.as_text == "internalField  uniform 0;"
    d2 = Dictionary(d.as_text)
    assert d2["internalField"] == Compound(Word("uniform"), Label(0))


# -- Dictionary / KeywordEntry --------------------------------------------


def test_dictionary_basic_entries():
    d = Dictionary('a 1; b "str"; c { d 1; }')
    assert d["a"] == Label(1)
    assert d["b"] == String("str")
    assert d["c"]["d"] == Label(1)


def test_dictionary_compound_value_collapsing():
    d = Dictionary("div(U) Gauss limitedLinear 1;")
    entry = d[0]
    assert str(entry.value) == "Gauss limitedLinear 1"


def test_dictionary_bare_word_with_empty_value():
    from pyfoamy.parsing.grammar import Compound
    # "H2O;": a species name without value.
    d = Dictionary("mixture\n{\n    H2O;\n}\n")
    assert d["mixture"][0] == KeywordEntry(Word("H2O"), Compound([]))
    assert d.as_text == d.__class__(d.as_text).as_text


def test_dictionary_variable_substitution_entry_with_semicolon():
    from pyfoamy.parsing.grammar import Compound
    # "$p;": dictionary merge, common in fvSolution.
    d = Dictionary('"(U|e).*" { $p; tolerance 1e-05; }')
    inner = d["(U|e).*"]
    assert inner[0] == KeywordEntry(Variable("$p"), Compound([]))
    assert inner["tolerance"] == Number("1e-05")


def test_dictionary_bare_variable_substitution_no_semicolon():
    from pyfoamy.parsing.grammar import Compound
    # "{ $var }" without ";": used in tutorials (momentumTransfer) and
    # accepted by solvers, although plain foamDictionary rejects it.
    d = Dictionary("foo { $bar }\n")
    assert d["foo"][0] == KeywordEntry(Variable("$bar"), Compound([]))


def test_dictionary_scoped_variable_substitution():
    # "$ev/motion;": scoped reference, as in dynamicMeshDict.
    d = Dictionary("foo { $ev/motion; }\n")
    assert d["foo"][0].key == Variable("$ev/motion")


def test_dictionary_variable_substitution_round_trips():
    text = '"(U|e).*" { $p; tolerance 1e-05; }'
    first = Dictionary(text).as_text
    second = Dictionary(first).as_text
    assert first == second


def test_dictionary_allows_empty_string_value():
    # decomposeParDict: `dataFile "";`
    d = Dictionary('dataFile "";\n')
    assert d["dataFile"] == String("")


def test_dictionary_rejects_empty_string_key():
    # Fatal in OpenFOAM (keyType.C: "Empty pattern string").
    with pytest.raises(ParseError):
        Dictionary('"" { foo 1; }\n')


def test_dictionary_set_accepts_variable_key():
    d = Dictionary()
    d.set(Variable("$p"), {})
    assert d[0] == KeywordEntry(Variable("$p"), Dictionary({}))


def test_dictionary_last_key_wins():
    d = Dictionary("a 1; a 2;")
    assert d["a"] == Label(2)
    assert len(d) == 2  # both entries kept, only lookup is last-wins


def test_dictionary_set_updates_in_place():
    d = Dictionary()
    d.set("a", 1)
    d.set("b", "x")
    d.set("a", 2)
    assert len(d) == 2
    assert d["a"] == Label(2)


def test_dictionary_contains():
    d = Dictionary("a 1;")
    assert "a" in d
    assert "b" not in d


def test_dictionary_from_plain_dict_nested():
    d = Dictionary({"a": 1, "b": {"c": 2.5}, "d": [1, 2, "x"]})
    assert d["a"] == Label(1)
    assert d["b"]["c"] == Number("2.5")
    assert d["d"] == List([1, 2, "x"])


def test_dictionary_rejects_invalid_key():
    with pytest.raises(ValueError):
        Dictionary({"123asdf": -1})


def test_dictionary_rejects_bad_source_type():
    with pytest.raises(TypeError):
        Dictionary(5)


def test_dictionary_from_file_reads_and_parses(tmp_path):
    path = tmp_path / "someDict"
    path.write_text("a 1;\nb (1 2 3);\n")
    d = Dictionary.from_file(path)
    assert d["a"] == Label(1)
    assert list(d["b"]) == [Label(1), Label(2), Label(3)]


def test_dictionary_from_file_accepts_str_path(tmp_path):
    path = tmp_path / "someDict"
    path.write_text("a 1;\n")
    d = Dictionary.from_file(str(path))
    assert d["a"] == Label(1)


def test_dictionary_as_text_has_no_outer_braces():
    d = Dictionary("a 1;")
    assert str(d).startswith("{")
    assert not d.as_text.startswith("{")


def test_dictionary_inlist_is_compact():
    d = Dictionary("a 1; b 2;")
    assert d.__str__(inlist=True) == "{ a 1; b 2; }"


def test_dictionary_nested_indentation_does_not_double():
    d = Dictionary("a { b { c 1; } d 2; } e 3;")
    lines = d.as_text.splitlines()
    # "c 1;" is two levels deep -> 8 spaces.
    c_line = next(line for line in lines if "c" in line)
    assert c_line.startswith(" " * 8) and not c_line.startswith(" " * 9)


def test_original_recursive_value_parsing_example():
    text = (
        "a ( { c d; b ( -1 4 asdf { a -1; } ); }"
        "  asdf { my entry;  } );"
    )
    d = Dictionary(text)
    val = d["a"]
    assert len(val) == 3
    bare_dict, word_asdf, named_dict = val
    assert bare_dict["c"] == Word("d")
    assert word_asdf == Word("asdf")
    assert named_dict["my"] == Word("entry")
    b_list = bare_dict["b"]
    assert list(b_list[:2]) == [Label(-1), Label(4)]
    assert b_list[2] == Word("asdf")
    assert b_list[3]["a"] == Label(-1)


# -- AnonymousListEntry (bare <count>(...) list at dict-body level) --------


def test_anonymous_list_entry_bare_no_count():
    d = Dictionary("(\n a { x 1; }\n b { y 2; }\n)\n")
    assert len(d) == 1
    entry = d[0]
    assert isinstance(entry, AnonymousListEntry)
    assert entry.count is None
    assert entry.body["a"]["x"] == Label(1)
    assert entry.body["b"]["y"] == Label(2)


def test_anonymous_list_entry_with_count():
    d = Dictionary("2\n(\n a { x 1; }\n b { y 2; }\n)\n")
    entry = d[0]
    assert entry.count == Label(2)
    assert entry.body["a"]["x"] == Label(1)


def test_anonymous_list_entry_rejects_wrong_count():
    with pytest.raises(ParseError):
        Dictionary("3\n(\n a { x 1; }\n b { y 2; }\n)\n")


def test_anonymous_list_entry_round_trips():
    text = "2\n(\n a\n{\n    x              1;\n}\n b\n{\n    y              2;\n}\n)"
    d = Dictionary(text)
    d2 = Dictionary(d.as_text)
    assert d == d2


def test_anonymous_list_entry_skipped_by_string_key_lookup():
    # String lookup skips non-KeywordEntry nodes.
    d = Dictionary("foo 1;\n(\n a { x 1; }\n)\n")
    assert d["foo"] == Label(1)
    assert len(d) == 2


def test_real_boundary_file_parses_and_round_trips():
    boundary_path = (
        Path(__file__).parent.parent / "fixtures" / "cavity_ascii"
        / "constant" / "polyMesh" / "boundary"
    )
    text = boundary_path.read_text()
    d = Dictionary(text)
    anon = [n for n in d if isinstance(n, AnonymousListEntry)]
    assert len(anon) == 1
    assert anon[0].count == len(anon[0].body)
    d2 = Dictionary(d.as_text)
    assert d == d2


# -- to_grammar_value ------------------------------------------------------


def test_to_grammar_value_passthrough_preserves_caller_intent():
    s = String("5")
    assert to_grammar_value(s) is s


def test_to_grammar_value_bool():
    assert to_grammar_value(True) == Word("true")
    assert to_grammar_value(False) == Word("false")


# -- Directive --------------------------------------------------------


def _parse_body(text):
    return Dictionary(text)


def test_directive_zero_arg():
    d = _parse_body("#print\nfoo 1;\n")
    assert d[0] == Directive("#print")
    assert d["foo"] == Label(1)
    assert str(d[0]) == "#print"


def test_directive_word_rejects_leading_hash():
    with pytest.raises(ValueError):
        Word("#print")


def test_directive_name_arg_input_mode():
    d = _parse_body("#inputMode merge\nfoo 1;\n")
    assert d[0] == Directive("#inputMode", args=Word("merge"))


def test_directive_name_arg_include_func_named_args():
    d = _parse_body(
        "#includeFunc residuals(name=residuals, fields=(h G ILambda_1_0))\n"
    )
    assert d[0].name == "#includeFunc"
    assert d[0].args == Word("residuals")
    assert d[0].call_args == Arguments(named={
        "name": Word("residuals"),
        "fields": List([Word("h"), Word("G"), Word("ILambda_1_0")]),
    })


def test_directive_name_arg_include_func_mixes_positional_and_named():
    # "volAverage(name=Tmean, T)"
    d = _parse_body("#includeFunc volAverage(name=Tmean, T)\n")
    assert d[0].call_args == Arguments(
        positional=[Word("T")], named={"name": Word("Tmean")}
    )


def test_directive_call_args_supports_dict_and_list_style_editing():
    # A named List argument is a real, appendable List.
    d = _parse_body(
        "#includeFunc residuals(name=residuals, fields=(h G ILambda_1_0))\n"
    )
    d[0].call_args.named["fields"].append(Word("k"))
    assert d[0].call_args.named["fields"] == List(
        [Word("h"), Word("G"), Word("ILambda_1_0"), Word("k")]
    )


def test_directive_name_arg_include_func_bare_name():
    d = _parse_body("#includeFunc wallHeatFlux\n")
    assert d[0] == Directive("#includeFunc", args=Word("wallHeatFlux"))
    assert d[0].call_args is None


def test_directive_arg_or_list_bare_token():
    d = _parse_body("#remove foo\nk 1;\n")
    assert d[0] == Directive("#remove", args=Word("foo"))


def test_directive_arg_or_list_paren_group():
    d = _parse_body("#remove (foo bar)\nk 1;\n")
    assert d[0] == Directive(
        "#remove", args=Arguments(positional=[Word("foo"), Word("bar")])
    )


def test_directive_arg_or_list_stops_at_first_close_paren():
    # readArgOrList does not count nested parens, so "(foo (bar))"
    # leaves a stray ")" and fails.
    with pytest.raises(ParseError):
        Dictionary("#remove (foo (bar))\nk 1;\n")


def test_directive_include_like_no_args():
    d = _parse_body('#include "alphaInternalField"\nk 1;\n')
    assert d[0] == Directive("#include", args=String("alphaInternalField"))
    assert d[0].call_args is None


def test_directive_include_like_with_args():
    d = _parse_body(
        '#include "alphaInternalField"(name=volume, y0=-0.088)\nk 1;\n'
    )
    assert d[0].args == String("alphaInternalField")
    assert d[0].call_args == Arguments(named={
        "name": Word("volume"), "y0": Number("-0.088"),
    })


def test_directive_include_like_allows_space_before_paren():
    # Whitespace, even a newline, before "(...)" is accepted.
    d = Dictionary('#include "foo" (name=bar)\nk 1;\n')
    assert d[0].call_args.named == {"name": Word("bar")}


def test_directive_include_like_allows_multiline_args():
    d = Dictionary('#include "adddd"\n(param=value)\nfoo 1;\n')
    assert d[0].call_args.named == {"param": Word("value")}


def test_directive_include_like_nested_parens_in_args():
    d = _parse_body(
        "#includeFunc patchAverage(fields=(T mag(U) p))\n"
    )
    assert d[0].args == Word("patchAverage")
    assert d[0].call_args == Arguments(named={
        "fields": List([Word("T"), Word("mag(U)"), Word("p")]),
    })


def test_directive_value_position_calc():
    d = _parse_body('angle #calc "-degToRad($angleOfAttack)";\n')
    assert d["angle"] == Directive(
        "#calc", args=String("-degToRad($angleOfAttack)")
    )


def test_directive_value_position_neg():
    d = _parse_body("internalField uniform (#neg $UMean 0 0);\n")
    result = list(d["internalField"])[1]
    assert result[0] == Directive("#neg", args=Variable("$UMean"))


def test_dictionary_tolerates_stray_semicolon_after_subdict():
    # Stray ";" after "}" ("foo { ... };") is tolerated.
    d = _parse_body("foo { bar 1; };\n")
    assert d["foo"]["bar"] == Label(1)


def test_dictionary_stray_semicolon_inside_list_item_dict():
    # Same for a dictionary list item.
    d = _parse_body(
        "patches ({ name inletFuel; } { name other; });\n"
    )
    assert list(d["patches"])[0]["name"] == Word("inletFuel")


def test_dictionary_writer_does_not_add_stray_semicolon_for_dict_in_list():
    d = _parse_body("patches ({ name inletFuel; });\n")
    assert "};" not in d.as_text
    assert d.as_text == Dictionary(d.as_text).as_text


def test_directive_value_position_code_stream():
    d = _parse_body(
        "vertices #codeStream\n{\n    code #{ os << 1; #};\n};\n"
    )
    value = d["vertices"]
    assert value.name == "#codeStream"
    assert value.subdict["code"] == Code(" os << 1; ")


def test_directive_code_stream_as_bare_list_item():
    result = parse_value("(1 #codeStream { code #{ x; #}; } 2)")
    assert result[0] == Label(1)
    assert result[1].name == "#codeStream"
    assert result[2] == Label(2)


def test_structural_directive_as_bare_list_item():
    # e.g. dynamicMeshDict: `times ( #include "meshTimes" );`
    result = parse_value('(#include "foo")')
    assert result[0] == Directive("#include", args=String("foo"))


def test_directive_code_dict_requires_its_subdict():
    # #codeDict needs its subdict, like #codeStream.
    with pytest.raises(ParseError):
        Dictionary("#codeDict\nfoo 1;\n")


def test_directive_code_dict_captures_subdict_like_code_stream():
    d = Dictionary("#codeDict\n{\n    code #{ os << 1; #};\n}\nfoo 1;\n")
    assert isinstance(d[0], Directive)
    assert d[0].name == "#codeDict"
    assert d[0].subdict["code"] == Code(" os << 1; ")
    assert d["foo"] == Label(1)


@pytest.mark.parametrize("name", [
    "#codeBlock", "#endCodeBlock", "#else", "#endif",
])
def test_directive_bare_pairing_directives_are_leaf_nodes(name):
    # Pairs are not matched, as in foamDictionary without -expand.
    d = Dictionary(f"{name}\nfoo 1;\n")
    assert d[0] == Directive(name)
    assert d["foo"] == Label(1)


@pytest.mark.parametrize("name", ["#if", "#ifeq", "#elif"])
def test_directive_conditionals_are_leaf_nodes(name):
    # readArgList requires the parens ("#if 1" is fatal) but does not
    # check the pairing with #endif.
    d = Dictionary(f"{name} ($a b)\nfoo 1;\n#endif\n")
    assert d[0] == Directive(
        name, args=Arguments.from_space_separated_text("$a b")
    )
    assert d["foo"] == Label(1)
    assert d[-1] == Directive("#endif")


@pytest.mark.parametrize(
    "text",
    [
        "#if (1)\na 1;\n#else\na 2;\n#endif\n",
        "#ifeq ($x 1.5)\na 1;\n#endif\n",
        '#if (#calc "${U} < (10)")\na 1;\n#endif\n',
        "#ifeq (${FOAM_APPLICATION} foamRun)\na 1;\n#endif\n",
    ],
)
def test_directive_conditional_arguments_round_trip(text):
    d = Dictionary(text)
    assert Dictionary(d.as_text) == d


def test_directive_stray_else_and_endif_with_no_matching_if():
    # foamDictionary accepts a lone #else/#endif.
    d = Dictionary("a 1;\n#else\nb 2;\n#endif\nc 3;\n")
    assert list(d) == [
        KeywordEntry(Word("a"), Label(1)),
        Directive("#else"),
        KeywordEntry(Word("b"), Label(2)),
        Directive("#endif"),
        KeywordEntry(Word("c"), Label(3)),
    ]


def test_directive_unknown_raises():
    with pytest.raises(ParseError, match="not supported"):
        Dictionary("#somethingMadeUp\nfoo 1;\n")


def test_directive_nodes_are_skipped_by_key_lookup():
    d = Dictionary("#print\nfoo 1;\n#print\n")
    assert d["foo"] == Label(1)
    assert "foo" in d
    assert len(d) == 3
    d.set("foo", 2)
    assert d["foo"] == Label(2)
    assert len(d) == 3


def test_dictionary_print_at_end():
    d = Dictionary({"a": 1}, print_at_end=True)
    assert d[-1] == Directive("#print")
    assert d.as_text == "a              1;\n#print"


@pytest.mark.parametrize("text", [
    "#print\nfoo 1;\n",
    "#inputMode merge\nfoo 1;\n",
    "#remove (foo bar)\nk 1;\n",
    '#include "alphaInternalField"(name=volume, y0=-0.088)\nk 1;\n',
    "#includeFunc residuals(name=residuals, fields=(h G ILambda_1_0))\n",
    'angle #calc "-degToRad($angleOfAttack)";\n',
    "vertices #codeStream\n{\n    code #{ os << 1; #};\n};\n",
])
def test_directive_round_trip_is_stable(text):
    first = Dictionary(text).as_text
    second = Dictionary(first).as_text
    assert first == second


# -- real oracle checks (skipped without a sourced OpenFOAM install) -----


@pytest.mark.openfoam
def test_oracle_rejects_bare_dollar(tmp_path):
    result = _foam_dictionary_expand(tmp_path, "a $;\n")
    assert result.returncode != 0


@pytest.mark.openfoam
def test_oracle_rejects_missing_semicolon(tmp_path):
    result = _foam_dictionary_expand(tmp_path, "dict1 { foo 5 }\nbar 6;\n")
    assert result.returncode != 0


@pytest.mark.openfoam
def test_oracle_accepts_our_number_rendering(tmp_path):
    d = Dictionary()
    d.set("a", Number("160.0"))
    result = _foam_dictionary_expand(tmp_path, d.as_text)
    assert result.returncode == 0, result.stderr
    assert "160" in result.stdout and "160.0" not in result.stdout


def test_number_str_handles_large_magnitude_whole_numbers():
    # "-1e30" (meshQualityDict sentinel) once crashed the < 1e20 guard.
    assert str(Number("-1e30")) == "-1E+30"
    assert str(Number("1e30")) == "1E+30"
    assert str(Number("160.0")) == "160"


@pytest.mark.openfoam
@pytest.mark.parametrize("text", [
    "#print\nfoo 1;\n",
    "#inputMode merge\nfoo 1;\n",
    "#remove (foo bar)\nk 1;\n",
    '#include "alphaInternalField"(name=volume, y0=-0.088)\nk 1;\n',
    "#includeFunc residuals(name=residuals, fields=(h G ILambda_1_0))\n",
    'angle #calc "-degToRad($angleOfAttack)";\n',
    "vertices #codeStream\n{\n    code #{ os << 1; #};\n};\n",
    "#codeDict\n{\n    code #{ os << 1; #};\n}\nfoo 1;\n",
])
def test_oracle_accepts_our_directive_rendering(tmp_path, text):
    rendered = Dictionary(text).as_text
    result = _foam_dictionary(tmp_path, rendered)
    assert result.returncode == 0, result.stderr


@pytest.mark.openfoam
@pytest.mark.parametrize("name", [
    "#codeBlock", "#endCodeBlock", "#else", "#endif",
])
def test_oracle_accepts_our_bare_pairing_directive_rendering(tmp_path, name):
    rendered = Dictionary(f"{name}\nfoo 1;\n").as_text
    result = _foam_dictionary(tmp_path, rendered)
    assert result.returncode == 0, result.stderr


@pytest.mark.openfoam
@pytest.mark.parametrize("name", ["#if", "#ifeq", "#elif"])
def test_oracle_accepts_our_conditional_directive_rendering(tmp_path, name):
    # Argument shape is checked ("#if 1" is fatal), pairing is not.
    rendered = Dictionary(f"{name} ($a b)\nfoo 1;\n#endif\n").as_text
    result = _foam_dictionary(tmp_path, rendered)
    assert result.returncode == 0, result.stderr


@pytest.mark.openfoam
def test_oracle_accepts_our_string_continuation_rendering(tmp_path):
    rendered = Dictionary('a "foo\\\nbar";\n').as_text
    result = _foam_dictionary(tmp_path, rendered)
    assert result.returncode == 0, result.stderr


# -- () balance corner cases (real foamDictionary-verified) -----------------


def test_word_value_with_operator_characters():
    d = Dictionary("a a+b=c asdf asd;")
    assert list(d["a"]) == [Word("a+b=c"), Word("asdf"), Word("asd")]


def test_list_item_word_stops_at_the_lists_own_close_paren():
    d = Dictionary("list ( a b grad(u );")
    assert list(d["list"]) == [
        Word("a"), Word("b"), Word("grad(u"),
    ]


@pytest.mark.parametrize(
    "text,key,value",
    [
        ("a_(asdfasdf -1;", "a_(asdfasdf", Label(-1)),
        ("c_(asdf)asdf -3;", "c_(asdf)asdf", Label(-3)),
    ],
)
def test_key_with_unbalanced_or_balanced_embedded_parens(text, key, value):
    # Keys need not balance parens; foamDictionary agrees.
    d = Dictionary(text)
    assert d[key] == value


def test_word_followed_by_stray_close_paren_is_rejected():
    # "c asdfasfd)asdf;" is fatal in foamDictionary too.
    with pytest.raises(ParseError):
        Dictionary("c asdfasfd)asdf;")


def test_stray_close_paren_immediately_after_keyword_is_a_known_gap():
    # foamDictionary accepts "b_asdf)asdf -2;" (value ")" "asdf" "-2");
    # no standalone ")" value token here - not in any tutorial.
    with pytest.raises(ParseError):
        Dictionary("b_asdf)asdf -2;")
