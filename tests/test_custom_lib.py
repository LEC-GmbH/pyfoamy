import os
from pathlib import Path

import pytest

from pyfoamy import custom_lib
from pyfoamy.custom_lib import CustomLibError, _lib_path, ensure_built

LIB_SRC = Path(__file__).parent.parent / "openfoam" / "lib"


@pytest.fixture(autouse=True)
def _user_build(request, monkeypatch):
    # The build tests must build, even where a site has installed the lib.
    if "openfoam" in request.keywords:
        monkeypatch.setattr(custom_lib, "_installed_lib", lambda: None)


def test_missing_openfoam_env_raises_custom_lib_error(monkeypatch):
    monkeypatch.delenv("WM_PROJECT_VERSION", raising=False)
    with pytest.raises(CustomLibError, match="WM_PROJECT_VERSION"):
        ensure_built()


def test_unsupported_version_raises_custom_lib_error(monkeypatch):
    monkeypatch.setenv("WM_PROJECT_VERSION", "13")
    with pytest.raises(CustomLibError, match="13"):
        ensure_built()


@pytest.mark.parametrize("variable", ["FOAM_LIBBIN", "FOAM_SITE_LIBBIN"])
def test_installed_library_is_used_without_building(
    tmp_path, monkeypatch, variable,
):
    for name in ("FOAM_LIBBIN", "FOAM_SITE_LIBBIN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WM_PROJECT_VERSION", "14")
    monkeypatch.setenv(variable, str(tmp_path))
    (tmp_path / "libpyfoamy.so").write_bytes(b"")
    monkeypatch.setattr(custom_lib, "_build", None)
    assert ensure_built() == tmp_path / "libpyfoamy.so"


@pytest.mark.openfoam
def test_ensure_built_produces_a_real_shared_library():
    path = ensure_built()
    assert path == _lib_path()
    assert path.is_file()
    assert path.read_bytes()[:4] == b"\x7fELF"


@pytest.mark.openfoam
def test_repeated_calls_do_not_trigger_a_rebuild():
    ensure_built()
    before = _lib_path().stat().st_mtime_ns
    ensure_built()
    after = _lib_path().stat().st_mtime_ns
    assert before == after


@pytest.mark.slow
@pytest.mark.openfoam
def test_editing_source_triggers_a_rebuild():
    ensure_built()
    before = _lib_path().stat().st_mtime_ns

    version = os.environ["WM_PROJECT_VERSION"]
    src = LIB_SRC / version / "laminarFlameSpeeds" / "piecewisePhi"
    src_file = src / "piecewisePhi.C"
    original = src_file.read_text()
    try:
        src_file.write_text(original + "\n// cache-invalidation test\n")
        ensure_built()
        after = _lib_path().stat().st_mtime_ns
        assert after != before
    finally:
        src_file.write_text(original)
        # Rebuild the .so from the reverted source.
        ensure_built()
