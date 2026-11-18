import os
from pathlib import Path


_root = os.environ.get("PYFOAMY_TUTORIALS_ROOT") or os.environ.get(
    "FOAM_TUTORIALS"
)
TUTORIALS_ROOT = Path(_root).expanduser() if _root else None
HAVE_TUTORIALS = TUTORIALS_ROOT is not None and TUTORIALS_ROOT.is_dir()

_SKIP_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf",
    ".stl", ".obj", ".gz",
    ".py", ".sh", ".m4", ".org", ".csv", ".dat",
}


def iter_dictionary_files():
    """Yield (path, text) for every real FoamFile-headed, ascii-format
    dictionary/field file under `TUTORIALS_ROOT`."""
    for path in TUTORIALS_ROOT.rglob("*"):
        if not path.is_file() or path.suffix in _SKIP_EXT:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if not raw.strip():
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        header = text[:2000]
        if "FoamFile" not in header:
            continue
        body_end = text.find("}")
        preamble = text[: body_end if body_end != -1 else 2000]
        if "binary" in preamble:
            continue
        yield path, text


def _skip_ws_and_comments(text: str, pos: int) -> int:
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
        elif text[pos:pos + 2] == "//":
            end = text.find("\n", pos)
            pos = len(text) if end == -1 else end + 1
        elif text[pos:pos + 2] == "/*":
            end = text.find("*/", pos + 2)
            pos = len(text) if end == -1 else end + 2
        else:
            break
    return pos


def looks_like_bare_list_file(text: str) -> bool:
    """True for files whose FoamFile header is followed directly by a bare
    numeric list/vector, e.g. mesh files (points/owner/faces/cells) or
    particle cloud positions - these are read by OpenFOAM's List<T> I/O,
    not Foam::dictionary::read(), and are out of scope for this parser
    (even `foamDictionary` cannot read them).

    A minimal, self-contained scanner (not grammar.py's own machinery,
    which has no bare standalone-token concept at all by design) - it
    only needs to recognize "WORD FoamFile", a balanced `{...}` block
    (comments/quoted strings skipped so an embedded `{`/`}` doesn't
    miscount), then check whether what follows is a bare number or
    `(` - real dict files always have another WORD key next instead.
    """
    pos = _skip_ws_and_comments(text, 0)
    word_end = pos + len("FoamFile")
    if text[pos:word_end] != "FoamFile":
        return False
    if word_end < len(text) and (
        text[word_end].isalnum() or text[word_end] == "_"
    ):
        return False
    pos = _skip_ws_and_comments(text, word_end)
    if pos >= len(text) or text[pos] != "{":
        return False
    depth = 0
    while pos < len(text):
        if text[pos:pos + 2] == "//":
            end = text.find("\n", pos)
            pos = len(text) if end == -1 else end + 1
            continue
        if text[pos:pos + 2] == "/*":
            end = text.find("*/", pos + 2)
            pos = len(text) if end == -1 else end + 2
            continue
        if text[pos] == '"':
            end = text.find('"', pos + 1)
            pos = len(text) if end == -1 else end + 1
            continue
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            pos += 1
            if depth == 0:
                break
            continue
        pos += 1
    pos = _skip_ws_and_comments(text, pos)
    if pos >= len(text):
        return False
    c = text[pos]
    return c == "(" or c.isdigit() or c in "+-."
