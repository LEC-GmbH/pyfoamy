"""Reads ascii or binary, optionally gzipped, OpenFOAM field files."""

import re
from decimal import Decimal
from typing import Dict, Tuple

import numpy as np

from pyfoamy.parsing.grammar import (
    AnonymousListEntry,
    Compound,
    Dictionary,
    Directive,
    KeywordEntry,
    Label,
    List,
    Number,
    String,
)
from pyfoamy.parsing.listfile import (
    _read_bytes,
    _read_count_and_open,
    read_header,
)

# (components, bytes per component). Binary blobs are cut by length,
# since ')' can occur in the data. Tensors untested: no tutorial has one.
_ITEM_SPECS = {
    "scalar": (1, 8),
    "sphericalTensor": (1, 8),
    "vector": (3, 8),
    "symmTensor": (6, 8),
    "tensor": (9, 8),
}

_PLACEHOLDER_PREFIX = "__pyfoamy_list_"

_LABEL_TOKEN = re.compile(r"-?\d+").fullmatch
_NUMBER_TOKEN = re.compile(
    r"-?(?:\d+\.\d*(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+|inf|nan)"
).fullmatch
_ITEM_LIST_END = re.compile(rb"\)\s*\)")
# Binary output writes an empty list as its count alone.
_EMPTY_LIST = re.compile(rb"\s*0(?=\s*;)")


def _plain_number(token: str):
    if _LABEL_TOKEN(token):
        return int.__new__(Label, token)
    if _NUMBER_TOKEN(token):
        return Decimal.__new__(Number, token)
    raise ValueError(token)


def _grammar_list(flat, n_components: int, type_name: str) -> List:
    if n_components == 1:
        return List(flat, type_name=type_name)
    return List(
        [
            List(flat[i:i + n_components])
            for i in range(0, len(flat), n_components)
        ],
        type_name=type_name,
    )


def _shaped(flat: np.ndarray, n_components: int) -> np.ndarray:
    return flat if n_components == 1 else flat.reshape(-1, n_components)


def _decode_blob(blob: bytes, n_components: int, type_name: str, as_array):
    flat = np.frombuffer(blob, "<f8")
    if as_array:
        return _shaped(flat.copy(), n_components)
    # Number() validates through the grammar (~40x slower); fields can
    # have millions of already-valid values.
    return _grammar_list(
        [Decimal.__new__(Number, str(v)) for v in flat.tolist()],
        n_components, type_name,
    )


def _parse_ascii_items(
    data: bytes, open_pos: int, count: int, n_components: int,
    type_name: str, as_array: bool,
):
    # The grammar takes ~0.4 ms per scalar; this builds the same objects
    # for a list of plain numbers and leaves anything else to the grammar.
    if n_components == 1:
        end = data.find(b")", open_pos)
        body = data[open_pos:end]
    else:
        match = _ITEM_LIST_END.search(data, open_pos)
        if match is None:
            return None
        end = match.end() - 1
        body = data[open_pos:end]
        if body.count(b"(") != count or body.count(b")") != count:
            return None
        body = body.replace(b"(", b" ").replace(b")", b" ")
    tokens = body.decode("utf-8").split()
    if count == 0 or end == -1 or len(tokens) != count * n_components:
        return None
    try:
        if as_array:
            return _shaped(np.array(tokens, float), n_components), end
        flat = [_plain_number(token) for token in tokens]
    except ValueError:
        return None
    return _grammar_list(flat, n_components, type_name), end


# (items, index of the list's closing char), or None for the grammar.
def _read_list(data, close_angle, type_name, binary, as_array):
    spec = _ITEM_SPECS.get(type_name)
    if spec is None:
        return None
    n_components, comp_size = spec
    try:
        count, open_pos = _read_count_and_open(data, close_angle + 1)
    except ValueError:
        empty = _EMPTY_LIST.match(data, close_angle + 1)
        if empty is None:
            if binary:
                raise
            return None
        if as_array:
            return _shaped(np.empty(0), n_components), empty.end() - 1
        return List([], type_name=type_name), empty.end() - 1
    if not binary:
        return _parse_ascii_items(
            data, open_pos, count, n_components, type_name, as_array
        )
    blob_end = open_pos + count * n_components * comp_size
    if data[blob_end:blob_end + 1] != b")":
        raise ValueError(
            f"expected ')' after binary List<{type_name}> blob "
            f"at byte {blob_end}"
        )
    return _decode_blob(
        data[open_pos:blob_end], n_components, type_name, as_array
    ), blob_end


def _extract_lists(data: bytes, as_array: bool = False) -> Tuple[str, Dict]:
    _, fmt, header_end = read_header(data)
    binary = fmt == "binary"

    blob_map: Dict = {}
    parts = [data[:header_end].decode("utf-8")]
    pos = header_end
    counter = 0

    while True:
        idx = data.find(b"List<", pos)
        if idx == -1:
            parts.append(data[pos:].decode("utf-8"))
            break
        close_angle = data.index(b">", idx)
        type_name = data[idx + len("List<"):close_angle].decode("ascii")
        found = _read_list(data, close_angle, type_name, binary, as_array)
        if found is None:
            # e.g. List<word>, or an ascii list the grammar must read.
            parts.append(data[pos:close_angle + 1].decode("utf-8"))
            pos = close_angle + 1
            continue
        items, end = found

        placeholder = f"{_PLACEHOLDER_PREFIX}{counter}__"
        counter += 1
        blob_map[placeholder] = items
        # No count: the grammar would check it against the one item.
        parts.append(data[pos:idx].decode("utf-8"))
        parts.append(f'List<{type_name}>("{placeholder}")')
        pos = end + 1

    return "".join(parts), blob_map


def _is_placeholder_list(value) -> bool:
    return (
        isinstance(value, List)
        and value.type_name is not None
        and len(value) == 1
        and isinstance(value[0], String)
        and value[0].data.startswith(_PLACEHOLDER_PREFIX)
    )


def _substitute_value(value, blob_map: Dict[str, List]):
    if _is_placeholder_list(value):
        return blob_map[value[0].data]
    if isinstance(value, Dictionary):
        _substitute_blobs(value, blob_map)
        return value
    if isinstance(value, Compound):
        # In place: Compound() would convert a numpy array item by item.
        value.data = [_substitute_value(v, blob_map) for v in value.data]
    return value


def _substitute_blobs(dictionary: Dictionary, blob_map: Dict[str, List]):
    for node in dictionary:
        if isinstance(node, KeywordEntry):
            node.value = _substitute_value(node.value, blob_map)
        elif isinstance(node, Directive) and node.subdict is not None:
            _substitute_blobs(node.subdict, blob_map)
        elif isinstance(node, AnonymousListEntry):
            _substitute_blobs(node.body, blob_map)


def read_field(path) -> Dictionary:
    """Parses a field file; binary lists read as their ascii form."""
    return _read_dictionary(path, as_array=False)


def _read_dictionary(path, as_array: bool) -> Dictionary:
    text, blob_map = _extract_lists(_read_bytes(path), as_array)
    dictionary = Dictionary(text)
    if blob_map:
        _substitute_blobs(dictionary, blob_map)
    return dictionary
