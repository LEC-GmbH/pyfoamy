"""pyparsing grammar for OpenFOAM dictionaries; values validate by parsing."""

import contextvars
import re
from collections import UserList, UserString
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Dict, Tuple, Union

import pyparsing as pp

from pyfoamy.parsing._dimension_table import DIMENSIONS, UNITS

# Off: "123abc" (a number glued to a word) is an error, not two values.
_split_var = contextvars.ContextVar("_split", default=False)


@contextmanager
def _split_mode(enabled):
    token = _split_var.set(enabled)
    try:
        yield
    finally:
        _split_var.reset(token)


class ParseError(ValueError):
    """Parse failure; the pyparsing exception is in `.pyparsing_exc`."""

    def __init__(self, pyparsing_exc: pp.ParseBaseException) -> None:
        super().__init__(str(pyparsing_exc))
        self.pyparsing_exc = pyparsing_exc

    @property
    def line(self) -> int:
        """The 1-based line number the parse failure occurred at."""
        return self.pyparsing_exc.lineno


def _resolve_items(args):
    # List("asdf") is one item, not four characters.
    if len(args) == 1 and isinstance(args[0], (list, tuple, UserList)):
        return list(args[0])
    return list(args)


def _render_inlist(item) -> str:
    if isinstance(item, Dictionary):
        return item.__str__(inlist=True)
    return str(item)


# Mutually recursive with the value classes; defined at the bottom.
_value_unit = pp.Forward()
_dict_body = pp.Forward()


# Like word::valid(): parens need not balance (`CH3N2_cy(-CH*N=NH-` in
# thermoData), and a `)` at depth 0 ends the word (it closes a list).
class _WordScanner(pp.Token):
    def __init__(self, excluded_chars):
        super().__init__()
        self._excluded = frozenset(excluded_chars)

    def parseImpl(self, instring, loc, doActions=True):
        start = loc
        # A leading "(" starts a List.
        if loc < len(instring) and instring[loc] == "(":
            raise pp.ParseException(instring, loc, "expected a WORD", self)
        depth = 0
        n = len(instring)
        while loc < n:
            c = instring[loc]
            if c == "(":
                depth += 1
            elif c == ")":
                if depth == 0:
                    break
                depth -= 1
            elif c.isspace() or c in self._excluded:
                break
            loc += 1
        if loc == start:
            raise pp.ParseException(instring, loc, "expected a WORD", self)
        return loc, instring[start:loc]


class Word(UserString):
    """A bare OpenFOAM word, e.g. `linear`, `grad(U)`.

    Compares unequal to a `String` with the same text.

    Raises:
        ValueError: If `text` is not a single valid word.
    """

    _EXCLUDED_CHARS = frozenset(' \t\n\r\v\f"\'/$;{}')
    grammar = _WordScanner(_EXCLUDED_CHARS)
    grammar.add_condition(
        lambda t: not (t[0][0].isdigit() or t[0][0] in ".-#"),
        message=(
            "a leading digit/'.'/'-' starts a Label/Number, and a "
            "leading '#' always starts a directive - never a Word"
        ),
    )
    grammar.set_name("OpenFOAM WORD")

    def __init__(self, text) -> None:
        try:
            matched = self.grammar.parse_string(str(text), parse_all=True)
        except pp.ParseException as e:
            raise ValueError(
                f"{text!r} is not a valid bare WORD"
            ) from e
        super().__init__(matched[0])

    def __repr__(self) -> str:
        return f"Word({self.data!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Word) and self.data == other.data

    def __hash__(self) -> int:
        return hash((Word, self.data))

    def as_switch(self) -> "Switch":
        """Reinterprets this word as a `Switch`."""
        return Switch(str(self))


class Switch:
    """A boolean spelled `true`/`false`, `on`/`off` or `yes`/`no`.

    Never produced by parsing, since no syntax marks a word as boolean;
    see `Word.as_switch()`.
    """

    _TRUE_SPELLINGS = ("true", "on", "yes")
    _FALSE_SPELLINGS = ("false", "off", "no")

    def __init__(self, value) -> None:
        if isinstance(value, bool):
            self.value = value
            self.spelling = "true" if value else "false"
            return
        if isinstance(value, (str, Word)):
            text = str(value)
            if text in Switch._TRUE_SPELLINGS:
                self.value = True
            elif text in Switch._FALSE_SPELLINGS:
                self.value = False
            else:
                raise ValueError(f"{text!r} is not a valid Switch spelling")
            self.spelling = text
            return
        raise TypeError(f"cannot build a Switch from {value!r}")

    def __bool__(self) -> bool:
        return self.value

    def __repr__(self) -> str:
        return f"Switch({self.spelling!r})"

    def __str__(self) -> str:
        return self.spelling

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Switch):
            return self.value == other.value
        if isinstance(other, bool):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)


class String(UserString):
    """A quoted string; may be empty (only empty keys are invalid)."""

    # A backslash-newline continues the string (ISstream::read(string&));
    # a bare newline is an error there too.
    _unquote = pp.QuotedString('"', esc_char="\\")
    grammar = pp.Regex(r'"(?:[^"\\\n]|\\.|\\\n)*"').add_parse_action(
        lambda t: String._unquote.parse_string(
            t[0].replace("\\\n", "\\n")
        )[0]
    ).set_name("OpenFOAM STRING")

    def __init__(self, text) -> None:
        super().__init__(str(text))

    def __str__(self) -> str:
        escaped = (
            self.data.replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\\n")
        )
        return f'"{escaped}"'

    def __repr__(self) -> str:
        return f"String({self.data!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, String) and self.data == other.data

    def __hash__(self) -> int:
        return hash((String, self.data))


def _no_word_continuation(s, loc, toks):
    # Rejects "123abc" unless split mode is on.
    if _split_var.get():
        return True
    end = loc + len(toks[0])
    if end >= len(s):
        return True
    c = s[end]
    # "4(1 2 3)" is not ambiguous.
    return (
        c.isspace() or c in Word._EXCLUDED_CHARS or c in "()"
    )


def _is_number(value) -> bool:
    # Built from str() without parsing: it is what the grammar accepts as
    # is, except Decimal's "NaN"/"Infinity", which keep being parsed.
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            or isinstance(value, Decimal) and value.is_finite())


class Label(int):
    """An OpenFOAM LABEL; matched before `Number`, which decides the type.

    Raises:
        ValueError: If `value` is not a single valid label.
    """

    # The trailing ~Char(".eE") stops "150.0" from matching "150" as a
    # Label and leaving ".0" behind - Number must get the whole thing.
    grammar = (
        pp.Combine(pp.Optional(pp.Literal("-")) + pp.Word(pp.nums))
        + ~pp.Char(".eE")
    ).set_name("OpenFOAM LABEL")
    grammar.add_condition(_no_word_continuation)

    def __new__(cls, value):
        if isinstance(value, int) and not isinstance(value, bool):
            return super().__new__(cls, value)
        text = str(value)
        try:
            matched = cls.grammar.parse_string(text, parse_all=True)
        except pp.ParseException as e:
            raise ValueError(f"{value!r} is not a valid LABEL") from e
        return super().__new__(cls, matched[0])

    def __str__(self) -> str:
        # Otherwise object.__str__ would return our __repr__.
        return str(int(self))

    def __repr__(self) -> str:
        return f"Label({int(self)!r})"


class Number(Decimal):
    """An OpenFOAM SCALAR; whole numbers print without decimals.

    `Number("160.0")` prints `160`, as time directory names need.

    Raises:
        ValueError: If `value` is not a single valid number.
    """

    # C++ streams write inf/nan; OpenFOAM cannot read them back, but
    # e.g. a diverged run's fields contain them.
    grammar = (
        pp.original_text_for(pp.pyparsing_common.number)
        | pp.Regex(r"-?(inf|nan)")
    )
    grammar.add_condition(
        lambda t: not t[0].startswith("+"),
        message="a leading '+' does not start a Number",
    )
    grammar.add_condition(_no_word_continuation)
    grammar.set_name("OpenFOAM SCALAR")

    def __new__(cls, value):
        if _is_number(value):
            return super().__new__(cls, str(value))
        text = str(value)
        try:
            matched = cls.grammar.parse_string(text, parse_all=True)
        except pp.ParseException as e:
            raise ValueError(f"{value!r} is not a valid NUMBER") from e
        return super().__new__(cls, matched[0])

    def __str__(self) -> str:
        if not self.is_finite():
            sign = "-" if self.is_signed() else ""
            return sign + ("nan" if self.is_nan() else "inf")
        # Whole numbers quantize to 0 decimals ("160.0" -> "160");
        # < 1e20 guard: quantize() can't force integer precision above it.
        if abs(self) < Decimal("1e20") and self == self.to_integral_value():
            return str(super().quantize(Decimal(1)))
        return str(super().normalize())

    def __repr__(self) -> str:
        return f"Number({str(self)!r})"

    def __format__(self, format_spec: str) -> str:
        # Decimal.__format__ ignores the __str__ override.
        return format(str(self), format_spec)


class Variable(UserString):
    """A `$name` or `${...}` reference, stored with its `$`."""

    _EXCLUDED_CHARS = frozenset(" \t\n\r\v\f\"';{}")
    _core_chars = ""
    for _c in pp.printables:
        if _c not in _EXCLUDED_CHARS and _c not in "()":
            _core_chars += _c
    del _c
    _core = pp.Word(_core_chars).set_name("VARIABLE core characters")
    _paren_group = pp.original_text_for(
        pp.nested_expr("(", ")")
    ).set_name("balanced (...) group")
    _brace_group = pp.original_text_for(
        pp.nested_expr("{", "}")
    ).set_name("balanced {...} group")
    _plain = pp.Combine(
        pp.Literal("$")
        + _core
        + pp.ZeroOrMore(_paren_group + pp.Optional(_core))
    )
    _braced = pp.Combine(pp.Literal("$") + _brace_group)
    grammar = (_braced | _plain).set_name("OpenFOAM VARIABLE")

    def __init__(self, text) -> None:
        try:
            matched = self.grammar.parse_string(str(text), parse_all=True)
        except pp.ParseException as e:
            raise ValueError(
                f"{text!r} is not a valid VARIABLE reference"
            ) from e
        super().__init__(matched[0])

    def __repr__(self) -> str:
        return f"Variable({self.data!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Variable) and self.data == other.data

    def __hash__(self) -> int:
        return hash((Variable, self.data))


class Code(UserString):
    """A `#{ ... #}` code block; no escapes, may be empty."""

    grammar = pp.QuotedString(
        quote_char="#{", end_quote_char="#}", multiline=True
    ).set_name("OpenFOAM CODE block")

    def __init__(self, text) -> None:
        super().__init__(str(text))

    def __str__(self) -> str:
        return "#{" + self.data + "#}"

    def __repr__(self) -> str:
        return f"Code({self.data!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Code) and self.data == other.data

    def __hash__(self) -> int:
        return hash((Code, self.data))


class Bracket(UserString):
    """A `[...]` dimension or unit, stored unevaluated (e.g. `Pa s`)."""

    grammar = pp.original_text_for(
        pp.Literal("[") + pp.SkipTo("]") + pp.Literal("]")
    ).set_name("OpenFOAM dimension/unit bracket")

    def __init__(self, text) -> None:
        super().__init__(str(text))

    def __str__(self) -> str:
        return f"[{self.data}]"

    def __repr__(self) -> str:
        return f"Bracket({self.data!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Bracket) and self.data == other.data

    def __hash__(self) -> int:
        return hash((Bracket, self.data))


# Bracket() takes the inner text, not the matched "[...]".
_bracket_as_value_unit = Bracket.grammar.copy().add_parse_action(
    lambda t: Bracket(t[0][1:-1])
)


class UnitNumber(Number):
    """A number with a unit, e.g. `5 [bar]`.

    Units are not converted: `+`/`-` need equal units (a bare number
    takes this one's); `*`/`/` between two `UnitNumber`s raise.
    """

    def __new__(cls, value, unit):
        self = super().__new__(cls, value)
        self.unit = unit if isinstance(unit, Bracket) else Bracket(unit)
        return self

    def __repr__(self) -> str:
        return f"UnitNumber({super().__str__()!r}, {self.unit.data!r})"

    def __str__(self) -> str:
        return f"{super().__str__()} {self.unit}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UnitNumber):
            return Decimal(self) == Decimal(other) and self.unit == other.unit
        return NotImplemented

    def __hash__(self) -> int:
        return hash((Decimal(self), self.unit))

    def _matching_unit(self, other) -> Bracket:
        other_unit = other.unit if isinstance(other, UnitNumber) else \
            self.unit
        if other_unit != self.unit:
            raise ValueError(
                f"cannot combine {self.unit} with {other_unit} - unit "
                "conversion isn't evaluated (see Bracket's docstring)"
            )
        return self.unit

    def __add__(self, other):
        unit = self._matching_unit(other)
        return UnitNumber(Decimal(self) + Decimal(other), unit)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        unit = self._matching_unit(other)
        return UnitNumber(Decimal(self) - Decimal(other), unit)

    def __rsub__(self, other):
        unit = self._matching_unit(other)
        return UnitNumber(Decimal(other) - Decimal(self), unit)

    def __mul__(self, other):
        if isinstance(other, UnitNumber):
            raise TypeError(
                "multiplying two UnitNumbers would need to combine "
                "their units, which isn't evaluated (see Bracket)"
            )
        return UnitNumber(Decimal(self) * Decimal(other), self.unit)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, UnitNumber):
            raise TypeError(
                "dividing by a UnitNumber would need to combine "
                "their units, which isn't evaluated (see Bracket)"
            )
        return UnitNumber(Decimal(self) / Decimal(other), self.unit)


class List(UserList):
    """An OpenFOAM list `(a b ...)`, `List<T> N(...)` or `N{value}`.

    Args:
        uniform: The `N{value}` form; `count` is then N and required.
        type_name: `T` of a `List<T>` header; forces the count on
            output.

    Raises:
        ValueError: If `count` and `uniform` disagree.
    """

    def __init__(
        self, *items, uniform: bool = False, count=None, type_name=None,
    ) -> None:
        super().__init__(to_grammar_value(i) for i in _resolve_items(items))
        if uniform and count is None:
            raise ValueError("a uniform List needs its count")
        if not uniform and count is not None:
            raise ValueError(
                "count is only meaningful when uniform=True - an "
                "ordinary list's count is always len(items)"
            )
        self.uniform = uniform
        self.count = count
        self.type_name = type_name

    def __repr__(self) -> str:
        return (
            f"List({list(self.data)!r}, uniform={self.uniform!r}, "
            f"count={self.count!r}, type_name={self.type_name!r})"
        )

    def __str__(self, leading_count: bool = False) -> str:
        body = " ".join(_render_inlist(item) for item in self.data)
        if self.uniform:
            prefix = f"List<{self.type_name}> " if self.type_name else ""
            return f"{prefix}{self.count}{{{body}}}"
        prefix = f"List<{self.type_name}> " if self.type_name else ""
        if leading_count or self.type_name:
            return f"{prefix}{len(self.data)}({body})"
        return f"{prefix}({body})"

    _count_token = pp.Word(pp.nums).set_name("count").add_parse_action(
        lambda t: int(t[0])
    )
    # Kept, not suppressed: field files need it written back.
    _typed_header = (
        pp.Suppress("List<") + pp.Word(pp.alphanums) + pp.Suppress(">")
    ).set_name("List<Type> header")

    _uniform_body = pp.Suppress("{") + _value_unit + pp.Suppress("}")
    _plain_body = (
        pp.Suppress("(") + pp.Group(pp.ZeroOrMore(_value_unit))
        + pp.Suppress(")")
    )

    # A bare count before "(" is ambiguous ("1 (2 3)"); only counted
    # after a List<Type> header.
    _uniform_alt = pp.Optional(_typed_header) + _count_token + _uniform_body
    _plain_alt = (
        (_typed_header + pp.Optional(_count_token) + _plain_body)
        | _plain_body
    )

    grammar = (_uniform_alt | _plain_alt).set_name("OpenFOAM LIST")

    @staticmethod
    def _build_uniform(toks):
        # The type name is a str, the count an int.
        rest = list(toks)
        type_name = rest.pop(0) if isinstance(rest[0], str) else None
        count, value = rest
        return List(
            [value], uniform=True, count=count, type_name=type_name,
        )

    @staticmethod
    def _build_plain(toks):
        rest = list(toks)
        type_name = rest.pop(0) if isinstance(rest[0], str) else None
        count = rest.pop(0) if len(rest) == 2 else None
        items = list(rest[0])
        # A mismatch is not a counted list; backtrack.
        if count is not None and count != len(items):
            raise pp.ParseException("declared count != item count")
        return List(items, type_name=type_name)


# Wired after the class: set_parse_action() wipes earlier conditions,
# and staticmethods are not callable in the class body before 3.10.
List._uniform_alt.set_parse_action(List._build_uniform)
List._plain_alt.set_parse_action(List._build_plain)


class Compound(UserList):
    """Space-separated values without brackets, e.g. `Gauss linear 1`.

    The value of an entry with several tokens; unlike `List`, it has no
    grammar of its own.
    """

    def __init__(self, *items) -> None:
        super().__init__(to_grammar_value(i) for i in _resolve_items(items))

    def __repr__(self) -> str:
        return f"Compound({list(self.data)!r})"

    def __str__(self) -> str:
        return " ".join(_render_inlist(item) for item in self.data)


class Uniform(Compound):
    """`uniform <value>`, e.g. `Uniform((0, 0, 0))`.

    Equal to the parsed `Compound` of `uniform 0`; needed because the
    text `"uniform 0"` would become one quoted string.
    """

    def __init__(self, value) -> None:
        super().__init__(Word("uniform"), value)

    @property
    def value(self):
        return self.data[1]

    @value.setter
    def value(self, new_value) -> None:
        new_value = to_grammar_value(new_value)
        old_is_seq = isinstance(self.data[1], List)
        new_is_seq = isinstance(new_value, List)
        if old_is_seq != new_is_seq:
            raise TypeError(
                "cannot change Uniform's value from "
                f"{'a sequence' if old_is_seq else 'a scalar'} to "
                f"{'a sequence' if new_is_seq else 'a scalar'}"
            )
        if old_is_seq and len(new_value) != len(self.data[1]):
            raise ValueError(
                "cannot change Uniform's sequence length from "
                f"{len(self.data[1])} to {len(new_value)}"
            )
        self.data[1] = new_value

    def __repr__(self) -> str:
        return f"Uniform({self.data[1]!r})"


def _as_value_unit(cls):
    # add_parse_action: set_parse_action would drop Word's/Number's
    # conditions. copy() keeps cls.grammar itself action-free.
    return cls.grammar.copy().add_parse_action(lambda t: cls(t[0]))


# Tried before bare Label/Number, which would take just the number.
_unit_number_value_unit = (
    (_as_value_unit(Label) | _as_value_unit(Number)) + _bracket_as_value_unit
).add_parse_action(lambda t: UnitNumber(t[0], t[1]))


_directive_name_token = pp.Combine(
    pp.Literal("#") + pp.Word(pp.alphas)
).set_name("directive name")


def _directive_name(names):
    # Read the whole name, then check, so "#include" does not match the
    # start of "#includeFunc".
    names = frozenset(names)
    return _directive_name_token.copy().add_condition(
        lambda t: t[0] in names
    )


class Arguments:
    """Arguments of a directive's `(...)`, e.g. `(name=Tmean, T)`.

    Values are parsed, whereas OpenFOAM keeps named values as text.
    """

    def __init__(self, positional=(), named=None) -> None:
        self.positional = list(positional)
        self.named = dict(named) if named else {}

    def __repr__(self) -> str:
        return (
            f"Arguments(positional={self.positional!r}, "
            f"named={self.named!r})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Arguments)
            and self.positional == other.positional
            and self.named == other.named
        )

    def __str__(self, sep: str = ", ") -> str:
        parts = [str(v) for v in self.positional]
        parts += [f"{k}={v}" for k, v in self.named.items()]
        return sep.join(parts)

    # "=" and "," are word characters in OpenFOAM, so split the text
    # first, then parse each value.
    _arg_key = pp.Word(
        pp.printables, exclude_chars="=,()"
    ).set_name("argument name")
    _arg_value_text = pp.original_text_for(
        pp.OneOrMore(pp.nested_expr("(", ")") | pp.Regex(r"[^,()]+"))
    ).set_name("argument value")
    _named_arg = pp.Group(_arg_key + pp.Suppress("=") + _arg_value_text)
    _positional_arg = pp.Group(_arg_value_text)
    _comma_separated = pp.DelimitedList(_named_arg | _positional_arg)

    # #codeInclude/#remove/#if/#ifeq: tokens, no "=", e.g.
    # `($a 1)` or `(#calc "$U < 10")`.
    _space_separated_item = pp.Group(_value_unit)
    _space_separated = pp.ZeroOrMore(_space_separated_item)

    @staticmethod
    def _parse_value_text(text):
        return _value_unit.parse_string(text.strip(), parse_all=True)[0]

    @staticmethod
    def _build_from_comma_separated(toks):
        positional = []
        named = {}
        for item in toks:
            if len(item) == 2:
                named[item[0]] = Arguments._parse_value_text(item[1])
            else:
                positional.append(Arguments._parse_value_text(item[0]))
        return Arguments(positional, named)

    @classmethod
    def from_comma_separated_text(cls, text):
        """Parses `key=value, ...` text without the outer parens."""
        if not text.strip():
            return Arguments()
        toks = cls._comma_separated.parse_string(text, parse_all=True)
        return cls._build_from_comma_separated(toks)

    @classmethod
    def from_space_separated_text(cls, text):
        """Parses space-separated text without the outer parens."""
        if not text.strip():
            return Arguments()
        toks = cls._space_separated.parse_string(text, parse_all=True)
        return Arguments(positional=[item[0] for item in toks])


class Directive:
    """A `#directive`, stored and never evaluated.

    `#if`/`#else`/`#endif` and `#codeBlock` pairs are not matched, as
    in `foamDictionary` without `-expand`.

    Args:
        args: `None`, a `Word`/`String`/`Variable`, or `Arguments` for
            `#codeInclude`/`#remove`/`#if`.
        call_args: `Arguments` of a trailing `(key=value, ...)`, e.g.
            `#includeFunc residuals(name=residuals)`.
        subdict: The `{ ... }` of `#codeStream`/`#codeDict`.
    """

    def __init__(
        self, name, args=None, call_args=None, subdict=None
    ) -> None:
        self.name = name
        self.args = args
        self.call_args = call_args
        self.subdict = subdict

    def __repr__(self) -> str:
        return (
            f"Directive({self.name!r}, args={self.args!r}, "
            f"call_args={self.call_args!r}, subdict={self.subdict!r})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Directive)
            and self.name == other.name
            and self.args == other.args
            and self.call_args == other.call_args
            and self.subdict == other.subdict
        )

    def __str__(self, indent: int = 0, inlist: bool = False) -> str:
        pad = "" if inlist else "    " * indent
        if self.subdict is not None:
            return f"{pad}{self.name} {self.subdict.__str__(inlist=True)}"
        if self.call_args is not None:
            return f"{pad}{self.name} {self.args}({self.call_args})"
        if isinstance(self.args, Arguments):
            return f"{pad}{self.name} ({self.args.__str__(sep=' ')})"
        if self.args is None:
            return f"{pad}{self.name}"
        return f"{pad}{self.name} {self.args}"

    _zero_arg_names = (
        "#print", "#codeBlock", "#endCodeBlock", "#else", "#endif",
    )
    _name_arg_names = (
        "#inputMode", "#includeFunc", "#includeConstraint",
        "#includeModel",
    )
    _arg_or_list_names = ("#codeInclude", "#remove")
    # Parens required ("#if 1" is fatal) and may nest, unlike
    # readArgOrList's (functionEntry::readArgList).
    _conditional_names = ("#if", "#ifeq", "#elif")
    _include_like_names = ("#include", "#includeIfPresent", "#includeEtc")

    # Names rejected with a dedicated "by design" message.
    _excluded_by_design = frozenset()

    # Glued ("residuals(...)") or on the next line; both are valid.
    _call_args_blob = pp.original_text_for(
        pp.nested_expr("(", ")")
    ).set_name("(key=value, ...) call arguments")

    # Unlike Word, stops at "(", so the name and call_args separate.
    _bare_name_chars = "".join(
        c for c in pp.printables
        if c not in Word._EXCLUDED_CHARS and c not in "()"
    )
    _name_or_string = (
        pp.Word(_bare_name_chars).add_parse_action(lambda t: Word(t[0]))
        | _as_value_unit(String)
    )
    _name_arg_value = _name_or_string + pp.Optional(_call_args_blob)
    _include_like_value = _as_value_unit(String) + pp.Optional(
        _call_args_blob
    )

    _arg_or_list_paren = pp.original_text_for(
        pp.Literal("(") + pp.SkipTo(")") + pp.Literal(")")
    ).set_name("(...) argument list, no nested parens")
    _conditional_paren = pp.original_text_for(
        pp.nested_expr("(", ")")
    ).set_name("(...) condition")
    _arg_or_list_value = (
        _as_value_unit(Word) | _as_value_unit(String)
        | _as_value_unit(Variable) | _arg_or_list_paren
    )

    _zero_arg_rule = _directive_name(_zero_arg_names)
    _name_arg_rule = _directive_name(_name_arg_names) + _name_arg_value
    _arg_or_list_rule = (
        _directive_name(_arg_or_list_names) + _arg_or_list_value
    )
    _conditional_rule = (
        _directive_name(_conditional_names) + _conditional_paren
    )
    _include_like_rule = (
        _directive_name(_include_like_names) + _include_like_value
    )

    _calc_rule = _directive_name(("#calc",)) + _as_value_unit(String)
    # negEntry.C reads exactly one "$name" token.
    _neg_rule = _directive_name(("#neg",)) + _as_value_unit(Variable)
    _code_stream_rule = (
        _directive_name(("#codeStream",))
        + pp.Suppress("{") + _dict_body + pp.Suppress("}")
    )
    # Like #codeStream, but standalone.
    _code_dict_rule = (
        _directive_name(("#codeDict",))
        + pp.Suppress("{") + _dict_body + pp.Suppress("}")
    )

    # Recognized names are excluded so their own argument errors show,
    # not "not supported".
    _recognized_names = frozenset(
        _zero_arg_names + _name_arg_names + _arg_or_list_names
        + _conditional_names + _include_like_names
        + ("#calc", "#neg", "#codeStream", "#codeDict")
    )
    _unsupported = _directive_name_token.copy().add_condition(
        lambda t: t[0] not in Directive._recognized_names
    )

    @staticmethod
    def _build_zero_arg(toks):
        return Directive(str(toks[0]))

    @staticmethod
    def _build_name_and_args(toks):
        return Directive(str(toks[0]), args=toks[1])

    @staticmethod
    def _build_name_with_call_args(toks):
        # [name, value] or [name, value, "(...)"]
        if len(toks) == 3:
            call_args = Arguments.from_comma_separated_text(toks[2][1:-1])
            return Directive(str(toks[0]), args=toks[1], call_args=call_args)
        return Directive(str(toks[0]), args=toks[1])

    @staticmethod
    def _build_name_and_subdict(toks):
        return Directive(str(toks[0]), subdict=toks[1])

    @staticmethod
    def _reject_unsupported(toks):
        name = str(toks[0])
        if name in Directive._excluded_by_design:
            raise pp.ParseFatalException(
                f"directive {name!r} is not supported - it requires "
                "live code compilation, which pyfoamy deliberately "
                "doesn't evaluate"
            )
        raise pp.ParseFatalException(f"directive {name!r} is not supported")

    dict_body_grammar = (
        _zero_arg_rule
        | _name_arg_rule
        | _arg_or_list_rule
        | _conditional_rule
        | _include_like_rule
        | _code_dict_rule
        | _unsupported
    ).set_name("OpenFOAM directive")

    # Structural directives also occur as list items, e.g.
    # `times ( #include "meshTimes" );`.
    value_grammar = (
        _code_stream_rule | _calc_rule | _neg_rule
        | _zero_arg_rule | _name_arg_rule | _arg_or_list_rule
        | _conditional_rule | _include_like_rule | _unsupported
    ).set_name("OpenFOAM value-position directive")


# Wired after the class, as for List above.
Directive._zero_arg_rule.add_parse_action(Directive._build_zero_arg)
for _paren in (Directive._arg_or_list_paren, Directive._conditional_paren):
    _paren.add_parse_action(
        lambda t: Arguments.from_space_separated_text(t[0][1:-1])
    )
del _paren
Directive._name_arg_rule.set_parse_action(
    Directive._build_name_with_call_args
)
Directive._arg_or_list_rule.set_parse_action(
    Directive._build_name_and_args
)
Directive._conditional_rule.set_parse_action(Directive._build_name_and_args)
Directive._include_like_rule.set_parse_action(
    Directive._build_name_with_call_args
)
Directive._calc_rule.set_parse_action(Directive._build_name_and_args)
Directive._neg_rule.set_parse_action(Directive._build_name_and_args)
Directive._code_stream_rule.set_parse_action(
    Directive._build_name_and_subdict
)
Directive._code_dict_rule.set_parse_action(
    Directive._build_name_and_subdict
)
Directive._unsupported.add_parse_action(Directive._reject_unsupported)


class KeywordEntry:
    """A `key value;` or `key { ... }` entry.

    Args:
        value: A value, a `Compound` for several tokens, or a
            `Dictionary`.
    """

    def __init__(self, key, value) -> None:
        self.key = key
        self.value = value

    @property
    def is_dict(self) -> bool:
        """Whether the value is a `Dictionary`."""
        return isinstance(self.value, Dictionary)

    def __repr__(self) -> str:
        return f"KeywordEntry({self.key!r}, {self.value!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, KeywordEntry)
            and self.key == other.key
            and self.value == other.value
        )

    def __str__(self, indent: int = 0, inlist: bool = False) -> str:
        if inlist:
            if self.is_dict:
                return f"{self.key} {self.value.__str__(inlist=True)}"
            return f"{self.key} {self.value};"
        pad = "    " * indent
        if self.is_dict:
            return f"{pad}{self.key}\n{self.value.__str__(indent=indent)}"
        return f"{pad}{str(self.key):<14} {self.value};"


class AnonymousListEntry:
    """A keyword-less `N ( ... )` of entries, e.g. in a `boundary` file."""

    def __init__(self, body, count=None) -> None:
        self.body = body
        self.count = count

    def __repr__(self) -> str:
        return f"AnonymousListEntry({self.body!r}, count={self.count!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AnonymousListEntry)
            and self.body == other.body
            and self.count == other.count
        )

    def __str__(self, indent: int = 0, inlist: bool = False) -> str:
        pad = "    " * indent
        count_line = f"{pad}{self.count}\n" if self.count is not None else ""
        body_text = "\n".join(
            entry.__str__(indent=indent + 1) for entry in self.body
        )
        return f"{count_line}{pad}(\n{body_text}\n{pad})"


class Dictionary(UserList):
    """An ordered list of entries and directives.

    `d["key"]` is the value of the last entry with that key, skipping
    directives; integers and slices index positionally.

    Args:
        source: `None`, dict-body text, a `dict`, or a list of nodes.
        print_at_end: Append a `#print` directive.
        split: Accept `123abc` as two values instead of an error.

    Raises:
        TypeError: If `source` has another type.
    """

    def __init__(
        self, source=None, *, print_at_end: bool = False, split: bool = False
    ) -> None:
        if source is None:
            super().__init__()
        elif isinstance(source, str):
            try:
                with _split_mode(split):
                    parsed = Dictionary.grammar.parse_string(
                        source, parse_all=True
                    )
            except pp.ParseBaseException as exc:
                raise ParseError(exc) from exc
            super().__init__(parsed[0].data)
        elif isinstance(source, dict):
            super().__init__()
            for key, value in source.items():
                self.set(key, value)
        elif isinstance(source, (list, UserList)):
            super().__init__(source)
        else:
            raise TypeError(f"cannot build a Dictionary from {source!r}")
        if print_at_end:
            self.data.append(Directive("#print"))

    @classmethod
    def from_file(cls, path, *, split: bool = False) -> "Dictionary":
        """Parses a dictionary file such as `system/fvSchemes`."""
        return cls(Path(path).read_text(), split=split)

    def set(self, key, value) -> "KeywordEntry":
        """Sets `key` in place (the last duplicate) or appends it.

        Args:
            key: A `Word`/`String`, or a `str` that is a valid word.

        Returns:
            The new `KeywordEntry`.
        """
        key = _to_key(key)
        entry = KeywordEntry(key, to_grammar_value(value))
        for i in range(len(self.data) - 1, -1, -1):
            existing = self.data[i]
            if isinstance(existing, KeywordEntry) and (
                existing.key.data == key.data
            ):
                self.data[i] = entry
                return entry
        self.data.append(entry)
        return entry

    def __getitem__(self, key):
        if isinstance(key, str):
            for entry in reversed(self.data):
                if (
                    isinstance(entry, KeywordEntry)
                    and entry.key.data == key
                ):
                    return entry.value
            raise KeyError(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        if isinstance(key, str):
            self.set(key, value)
            return
        super().__setitem__(key, value)

    def __contains__(self, key) -> bool:
        if isinstance(key, str):
            return any(
                isinstance(entry, KeywordEntry) and entry.key.data == key
                for entry in self.data
            )
        return super().__contains__(key)

    def remove(self, key: str) -> None:
        """Removes every entry with this key.

        Raises:
            KeyError: If there is none.
        """
        if key not in self:
            raise KeyError(f"no such entry: {key!r}")
        self.data = [
            entry for entry in self.data
            if not (
                isinstance(entry, KeywordEntry) and entry.key.data == key
            )
        ]

    def __delitem__(self, key) -> None:
        if isinstance(key, str):
            self.remove(key)
            return
        super().__delitem__(key)

    def __repr__(self) -> str:
        return f"Dictionary({list(self.data)!r})"

    @property
    def as_text(self) -> str:
        """The dictionary as file text, without enclosing braces."""
        return "\n".join(entry.__str__(indent=0) for entry in self.data)

    def __str__(self, indent: int = 0, inlist: bool = False) -> str:
        if inlist:
            body = " ".join(
                entry.__str__(inlist=True) for entry in self.data
            )
            return f"{{ {body} }}" if body else "{}"
        pad = "    " * indent
        body = "\n".join(
            entry.__str__(indent=indent + 1) for entry in self.data
        )
        return f"{pad}{{\n{body}\n{pad}}}"

    # Keys: Word, String (may be a regex) or Variable ("$p;" merges a
    # dictionary). An empty string is valid as a value, fatal as a key.
    _key_string = String.grammar.copy().add_condition(
        lambda t: t[0] != "", message="a dictionary key can't be empty"
    ).add_parse_action(lambda t: String(t[0]))
    _key = (
        _key_string | _as_value_unit(Word) | _as_value_unit(Variable)
    ).set_name("OpenFOAM dictionary key")
    # Empty values: "H2O;" (species name), "$p;" (substitution).
    _entry_value = pp.Group(pp.ZeroOrMore(_value_unit))
    _subdict_entry = _key + pp.Suppress("{") + _dict_body + pp.Suppress("}")
    _plain_entry = _key + _entry_value + pp.Suppress(";")
    # "foo { $bar }": a "$var" closed by "}" without ";".
    _bare_variable_entry = _as_value_unit(Variable) + pp.FollowedBy("}")
    # Keyword-less "N ( ... )" as in boundary files.
    _anonymous_list_entry = (
        pp.Optional(_as_value_unit(Label))
        + pp.Suppress("(") + _dict_body + pp.Suppress(")")
    )

    grammar = _dict_body.set_name("OpenFOAM dictionary body")

    @staticmethod
    def _build_subdict_entry(toks):
        return KeywordEntry(toks[0], toks[1])

    @staticmethod
    def _build_bare_variable_entry(toks):
        return KeywordEntry(toks[0], Compound([]))

    @staticmethod
    def _build_plain_entry(toks):
        # Collapse to a bare value for exactly one unit ("a 1;"),
        # else a Compound (multiple units) or empty Compound ("H2O;").
        items = list(toks[1])
        value = items[0] if len(items) == 1 else Compound(items)
        return KeywordEntry(toks[0], value)

    @staticmethod
    def _build_anonymous_list_entry(toks):
        if len(toks) == 2:
            count, body = toks
            if count != len(body):
                raise pp.ParseException("declared count != item count")
            return AnonymousListEntry(body, count=count)
        return AnonymousListEntry(toks[0])

    @staticmethod
    def _build_dict_body(toks):
        return Dictionary(list(toks))


# Wired after the class, as for List above.
Dictionary._subdict_entry.set_parse_action(Dictionary._build_subdict_entry)
Dictionary._plain_entry.set_parse_action(Dictionary._build_plain_entry)
Dictionary._bare_variable_entry.set_parse_action(
    Dictionary._build_bare_variable_entry
)
Dictionary._anonymous_list_entry.set_parse_action(
    Dictionary._build_anonymous_list_entry
)
_dict_body <<= pp.ZeroOrMore(
    Dictionary._subdict_entry
    | Dictionary._bare_variable_entry
    | Dictionary._plain_entry
    | Directive.dict_body_grammar
    | Dictionary._anonymous_list_entry
    # Stray ";" after "}" ("foo { ... };") is tolerated.
    | pp.Suppress(";")
)
_dict_body.set_parse_action(Dictionary._build_dict_body)

# A keyless "{...}" can be any value, e.g. a list item.
_dict_entry_as_value = (
    pp.Suppress("{") + _dict_body + pp.Suppress("}")
).set_name("bare {...} dictionary value")

# Not a value: "a { b c; }" inside a value is a Word followed by a
# dictionary, not an entry.

# Most specific first: Label before Number decides LABEL vs SCALAR;
# Word last since it would shadow the rest.
_value_unit <<= (
    List.grammar
    | _as_value_unit(Code)
    | _as_value_unit(String)
    | _unit_number_value_unit
    | _as_value_unit(Label)
    | _as_value_unit(Number)
    | _as_value_unit(Variable)
    | _dict_entry_as_value
    | Directive.value_grammar
    | _bracket_as_value_unit
    | _as_value_unit(Word)
)


_GRAMMAR_VALUE_TYPES = (
    Word, String, Label, Number, Variable, Code, Bracket, List, Compound,
    Dictionary, Directive, Uniform,
)


def _to_key(key):
    if isinstance(key, (Word, String, Variable)):
        return key
    if isinstance(key, str):
        return Word(key)
    raise TypeError(
        f"dictionary key must be a str, Word, String, or Variable: {key!r}"
    )


def to_grammar_value(value, *, split: bool = False):
    """Converts a Python value or text to a grammar value.

    `bool` -> `Word`, `dict` -> `Dictionary`, `list`/`tuple`/`range` ->
    `List`, `Path` -> `String`, `Enum` -> its value, numpy scalars and
    arrays -> their `tolist()`; grammar values pass through; `str` and
    numbers are parsed from `str()`.

    Raises:
        ParseError: If the text does not parse.
        TypeError: If `value` is of any other type.
    """
    if isinstance(value, _GRAMMAR_VALUE_TYPES):
        return value
    if isinstance(value, bool):
        return Word("true" if value else "false")
    if isinstance(value, dict):
        return Dictionary(value)
    if isinstance(value, (list, tuple, range)):
        return List(list(value))
    if isinstance(value, Enum) and not isinstance(value, str):
        return to_grammar_value(value.value, split=split)
    if isinstance(value, PurePath):
        return String(value)
    if type(value).__module__ == "numpy" and hasattr(value, "tolist"):
        return to_grammar_value(value.tolist(), split=split)
    if isinstance(value, int):
        return Label(value)
    if isinstance(value, float):
        return Number(value)
    if not isinstance(value, (str, int, float, Decimal)):
        raise TypeError(
            f"cannot convert {type(value).__name__} to an OpenFOAM "
            f"value: {value!r}"
        )
    try:
        with _split_mode(split):
            return _value_unit.parse_string(str(value), parse_all=True)[0]
    except pp.ParseBaseException as exc:
        raise ParseError(exc) from exc


# --- dimensionSet ----------------------------------------------------
# Expressions such as `[kg/m^3]` are only checked against the known
# names, not evaluated to exponents.


@dataclass(frozen=True)
class DimensionSet:
    """The 7 base-dimension exponents of a `dimensionSet`."""

    mass: Any
    length: Any
    time: Any
    temperature: Any
    moles: Any
    current: Any
    luminous_intensity: Any

    def as_tuple(self) -> Tuple:
        """The exponents in OpenFOAM order."""
        return (
            self.mass,
            self.length,
            self.time,
            self.temperature,
            self.moles,
            self.current,
            self.luminous_intensity,
        )


_dimless = DimensionSet(0, 0, 0, 0, 0, 0, 0)


def named_dimension_sets() -> Dict[str, DimensionSet]:
    """Named dimensions of OpenFOAM, e.g. `"pressure"`."""
    return _NAMED_DIMENSION_SETS


_NAMED_DIMENSION_SETS = {
    name: DimensionSet(*exponents) for name, exponents in DIMENSIONS.items()
}
_KNOWN_DIMENSION_UNIT_NAMES = frozenset(DIMENSIONS) | UNITS


_DIMENSION_EXPR_TOKEN_RE = re.compile(
    r"\s*(\^-?\d+|[*/]|[A-Za-z_][A-Za-z0-9_]*)"
)


def _tokenize_dimension_expr(text: str):
    tokens = []
    pos = 0
    while pos < len(text):
        match = _DIMENSION_EXPR_TOKEN_RE.match(text, pos)
        if match is None:
            raise ParseError(
                pp.ParseException(text, pos, "dimension expression")
            )
        tokens.append(match.group(1))
        pos = match.end()
    return tokens


def _validate_dimension_expr(text: str) -> None:
    tokens = _tokenize_dimension_expr(text)
    known = _KNOWN_DIMENSION_UNIT_NAMES
    for token in tokens:
        if token in ("*", "/") or token.startswith("^"):
            continue
        if token not in known:
            raise ValueError(
                f"{token!r} in {text!r} is not a known OpenFOAM "
                "dimension or unit name (see `foamUnits -all`)"
            )


def _to_number(text: str):
    try:
        return int(text)
    except ValueError:
        return float(text)


def parse_dimensions(value) -> Union[DimensionSet, Bracket]:
    """Parses a `dimensions` value.

    Returns:
        A `DimensionSet` for the 5/7 exponent form or a single named
        dimension; otherwise the `Bracket`, after checking its names.

    Raises:
        ValueError: If a name is unknown.
    """
    bracket = value if isinstance(value, Bracket) else Bracket(str(value))
    text = bracket.data
    parts = text.split()
    if not parts:
        # "[]" is dimensionless (dimensionSetIO.C).
        return _dimless
    if len(parts) in (5, 7):
        # dimensionSetIO.C reads 5 exponents, plus current and
        # luminous intensity only if a 6th number follows.
        try:
            values = [_to_number(p) for p in parts]
            if len(values) == 5:
                values += [0, 0]
            return DimensionSet(*values)
        except ValueError:
            pass
    if len(parts) == 1 and parts[0] in named_dimension_sets():
        return named_dimension_sets()[parts[0]]
    _validate_dimension_expr(text)
    return bracket


def dimension_bracket(dims: DimensionSet) -> Bracket:
    """The numeric bracket of `dims`, e.g. `[0 1 -1 0 0 0 0]`."""
    return Bracket(" ".join(str(int(x)) for x in dims.as_tuple()))


def dimensioned(value, dims: DimensionSet) -> UnitNumber:
    """`value [dims]` as a `UnitNumber`."""
    return UnitNumber(value, dimension_bracket(dims))


# .ignore() reaches only sub-expressions already wired in, so last;
# also on _value_unit for standalone value parsing.
_COMMENT = pp.dblSlashComment | pp.cStyleComment
Dictionary.grammar.ignore(_COMMENT)
_value_unit.ignore(_COMMENT)
