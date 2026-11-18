"""Reads ascii or binary, optionally gzipped, polyMesh list files."""

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from pyfoamy.parsing.grammar import Dictionary

# Default build (WM_LABEL_SIZE=32, double); the files don't record it.
_LABEL = np.dtype("<i4")
_LABEL_SIZE = _LABEL.itemsize
_SCALAR = np.dtype("<f8")

_GZIP_MAGIC = b"\x1f\x8b"


def _read_bytes(path) -> bytes:
    path = Path(path)
    gz_path = path.with_name(path.name + ".gz")
    if path.exists():
        data = path.read_bytes()
    elif gz_path.exists():
        data = gz_path.read_bytes()
    else:
        raise FileNotFoundError(f"neither {path} nor {gz_path} exists")
    if data[:2] == _GZIP_MAGIC:
        data = gzip.decompress(data)
    return data


def _find_header_end(data: bytes) -> int:
    pos = data.index(b"FoamFile")
    depth = 0
    n = len(data)
    while pos < n:
        c = data[pos:pos + 1]
        if c == b"{":
            depth += 1
        elif c == b"}":
            depth -= 1
            if depth == 0:
                return pos + 1
        pos += 1
    raise ValueError("could not find end of FoamFile header")


def _skip_ascii_prefix(data: bytes, pos: int) -> int:
    n = len(data)
    while True:
        while pos < n and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 2] == b"//":
            pos = data.index(b"\n", pos) + 1
            continue
        return pos


def _read_count_and_open(data: bytes, pos: int) -> Tuple[int, int]:
    pos = _skip_ascii_prefix(data, pos)
    start = pos
    while data[pos:pos + 1].isdigit():
        pos += 1
    if pos == start:
        raise ValueError(f"expected a list count at byte {pos}")
    count = int(data[start:pos])
    pos = _skip_ascii_prefix(data, pos)
    if data[pos:pos + 1] != b"(":
        found = data[pos:pos + 10]
        raise ValueError(f"expected '(' at byte {pos}, found {found!r}")
    return count, pos + 1


def _find_matching_close_paren(data: bytes, pos: int) -> int:
    depth = 1
    n = len(data)
    while pos < n:
        c = data[pos:pos + 1]
        if c == b"(":
            depth += 1
        elif c == b")":
            depth -= 1
            if depth == 0:
                return pos
        pos += 1
    raise ValueError("unterminated list - no matching ')'")


def read_header(data: bytes) -> Tuple[Dictionary, str, int]:
    """Parses the `FoamFile` header from raw file bytes.

    Returns:
        `(header, "ascii" | "binary", byte offset after the header)`.
    """
    header_end = _find_header_end(data)
    header_dict = Dictionary(data[:header_end].decode("ascii"))
    foam_file = header_dict["FoamFile"]
    return header_dict, str(foam_file["format"]), header_end


def _payload(path) -> Tuple[bytes, bool, int, int]:
    data = _read_bytes(path)
    _, fmt, header_end = read_header(data)
    count, pos = _read_count_and_open(data, header_end)
    return data, fmt == "binary", count, pos


def _ascii_numbers(data: bytes, pos: int, dtype) -> np.ndarray:
    end = _find_matching_close_paren(data, pos)
    text = data[pos:end].replace(b"(", b" ").replace(b")", b" ")
    return np.array(text.split(), dtype=dtype)


def read_label_list(path) -> np.ndarray:
    """Reads a label list file such as `owner`."""
    data, binary, count, pos = _payload(path)
    if binary:
        return np.frombuffer(data, _LABEL, count, pos).astype(np.int64)
    return _ascii_numbers(data, pos, np.int64)


def read_vector_list(path) -> np.ndarray:
    """Reads a vector list file such as `points`, as an `(n, 3)` array."""
    data, binary, count, pos = _payload(path)
    if binary:
        flat = np.frombuffer(data, _SCALAR, 3 * count, pos).copy()
    else:
        flat = _ascii_numbers(data, pos, np.float64)
    return flat.reshape(count, 3)


@dataclass(frozen=True)
class CompactLabelListList:
    """A list of label lists such as `faces`: item `i` is
    `values[offsets[i]:offsets[i + 1]]`."""

    offsets: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self.values[self.offsets[i]:self.offsets[i + 1]]

    def sizes(self) -> np.ndarray:
        """The number of labels per item."""
        return np.diff(self.offsets)


def read_compact_label_list_list(path) -> CompactLabelListList:
    """Reads a list of label lists such as `faces` or `cells`."""
    data, binary, count, pos = _payload(path)
    if binary:
        return _read_binary_compact_list_list(data, count, pos)
    return _parse_ascii_compact_list_list(data, pos, count)


read_face_list = read_compact_label_list_list
read_cell_list = read_compact_label_list_list


def _read_binary_compact_list_list(
    data: bytes, offset_count: int, pos: int
) -> CompactLabelListList:
    offsets = np.frombuffer(data, _LABEL, offset_count, pos)
    pos = pos + offset_count * _LABEL_SIZE
    if data[pos:pos + 1] != b")":
        raise ValueError(
            f"expected ')' after offsets list at byte {pos}"
        )
    data_count, data_pos = _read_count_and_open(data, pos + 1)
    if offset_count and data_count != offsets[-1]:
        raise ValueError(
            f"compact list data count {data_count} doesn't match "
            f"the last offset {offsets[-1]}"
        )
    values = np.frombuffer(data, _LABEL, data_count, data_pos)
    if not offset_count:
        offsets = np.zeros(1, _LABEL)
    return CompactLabelListList(
        offsets.astype(np.int64), values.astype(np.int64)
    )


def _parse_ascii_compact_list_list(
    data: bytes, pos: int, count: int
) -> CompactLabelListList:
    tokens = _ascii_numbers(data, pos, np.int64)
    offsets = np.zeros(count + 1, np.int64)
    keep = np.ones(len(tokens), bool)
    at = 0
    # Each item is its size followed by its labels.
    for i in range(count):
        if at >= len(tokens):
            raise ValueError(f"expected {count} items, found {i}")
        size = int(tokens[at])
        keep[at] = False
        offsets[i + 1] = offsets[i] + size
        at += size + 1
    if at != len(tokens):
        raise ValueError(f"item sizes don't match the {len(tokens)} labels")
    return CompactLabelListList(offsets, tokens[keep])
