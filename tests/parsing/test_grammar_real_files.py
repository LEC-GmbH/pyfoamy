import os
import random
from concurrent.futures import ProcessPoolExecutor

import pyparsing as pp
import pytest
from conftest import (
    HAVE_TUTORIALS,
    TUTORIALS_ROOT,
    iter_dictionary_files,
    looks_like_bare_list_file,
)

from pyfoamy.parsing.grammar import Dictionary, Directive

_WORKERS = min(16, os.cpu_count() or 4)
_THERMO_DATA = (
    TUTORIALS_ROOT.parent / "etc" / "thermoData" / "thermoData"
    if HAVE_TUTORIALS else None
)


def _shuffled_paths():
    # Pass paths, not text: pickling thousands of large strings in the
    # main process serialized the start. Shuffle to spread slow files.
    paths = [path for path, _text in iter_dictionary_files()]
    random.Random(0).shuffle(paths)
    return paths


def _is_excluded_by_design(exc: Exception) -> bool:
    if not isinstance(exc, pp.ParseFatalException):
        return False
    return any(f"{name!r}" in str(exc) for name in Directive._excluded_by_design)


def _check_parse(path):
    text = path.read_text()
    try:
        Dictionary(text)
    except Exception as exc:  # noqa: BLE001 - report any failure
        if looks_like_bare_list_file(text):
            return (str(path), "bare_list_skipped", None)
        if _is_excluded_by_design(exc):
            return (str(path), "excluded_by_design", None)
        return (str(path), "failed", f"{type(exc).__name__}: {exc}")
    return (str(path), "ok", None)


def _check_roundtrip(path):
    text = path.read_text()
    try:
        d1 = Dictionary(text)
    except Exception as exc:  # noqa: BLE001
        if looks_like_bare_list_file(text) or _is_excluded_by_design(exc):
            return (str(path), "parse_excluded", None)
        return (str(path), "parse_failed", None)
    try:
        d2 = Dictionary(d1.as_text)
    except Exception as exc:  # noqa: BLE001
        return (str(path), "failed", f"reparse raised: {exc}")
    if d1 != d2:
        return (str(path), "failed", "not equal after reparse")
    return (str(path), "ok", None)


@pytest.fixture(scope="module")
def _pool():
    with ProcessPoolExecutor(max_workers=_WORKERS) as pool:
        yield pool


@pytest.mark.tutorials
@pytest.mark.slow
def test_grammar_handles_every_real_tutorial_dictionary(_pool):
    paths = _shuffled_paths()
    assert len(paths) > 1000, f"expected a large corpus, only found {len(paths)}"

    failures = []
    bare_list_skipped = 0
    excluded_by_design = 0
    for path, status, detail in _pool.map(_check_parse, paths, chunksize=8):
        if status == "bare_list_skipped":
            bare_list_skipped += 1
        elif status == "excluded_by_design":
            excluded_by_design += 1
        elif status == "failed":
            failures.append((path, detail))

    # Known scope limits, pinned so a sudden jump is noticed.
    assert bare_list_skipped < 30
    assert excluded_by_design < 30

    if failures:
        details = "\n".join(f"  {p}: {d}" for p, d in failures[:20])
        pytest.fail(
            f"{len(failures)}/{len(paths)} real dictionary files failed "
            f"to parse for reasons other than the known bare-list-file "
            f"and excluded-by-design limitations:\n{details}"
        )


@pytest.mark.tutorials
@pytest.mark.slow
def test_grammar_roundtrip_preserves_structure_for_every_real_dictionary(
    _pool,
):
    paths = _shuffled_paths()

    failures = []
    checked = 0
    for path, status, detail in _pool.map(_check_roundtrip, paths, chunksize=8):
        if status == "parse_excluded":
            # Already reported by the tutorial-dictionary test.
            continue
        checked += 1
        if status != "ok":
            failures.append((path, detail))

    assert checked > 1000, f"expected a large corpus, only checked {checked}"

    if failures:
        details = "\n".join(f"  {p}: {d}" for p, d in failures[:20])
        pytest.fail(
            f"{len(failures)}/{checked} real dictionaries changed "
            f"structurally after a parse -> write -> reparse round trip:"
            f"\n{details}"
        )


@pytest.mark.slow
@pytest.mark.tutorials
def test_grammar_handles_thermo_data():
    # Header-less ~35000-line thermo database with unbalanced-paren
    # keys such as "CH3N2_cy(-CH*N=NH-".
    text = _THERMO_DATA.read_text()
    d1 = Dictionary(text)
    assert len(d1) > 100
    d2 = Dictionary(d1.as_text)
    assert d1 == d2
