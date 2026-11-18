import pytest

from pyfoamy.openfoam_version import (
    DEFAULT_VERSION, openfoam_version, use_openfoam_version,
)


def test_sourced_version_is_used(monkeypatch):
    monkeypatch.setenv("WM_PROJECT_VERSION", "dev")
    assert openfoam_version() == "dev"


def test_default_without_sourced_openfoam(monkeypatch):
    monkeypatch.delenv("WM_PROJECT_VERSION", raising=False)
    assert openfoam_version() == DEFAULT_VERSION == "14"


@pytest.mark.parametrize("version", ["13", "v2412", "12"])
def test_unsupported_sourced_version_raises(monkeypatch, version):
    monkeypatch.setenv("WM_PROJECT_VERSION", version)
    with pytest.raises(ValueError, match=repr(version)):
        openfoam_version()


def test_override_wins_over_sourced_version_and_is_restored(monkeypatch):
    monkeypatch.setenv("WM_PROJECT_VERSION", "13")
    with use_openfoam_version("dev") as version:
        assert version == openfoam_version() == "dev"
        with use_openfoam_version(14):
            assert openfoam_version() == "14"
        assert openfoam_version() == "dev"
    with pytest.raises(ValueError):
        openfoam_version()


def test_override_with_unsupported_version_raises():
    with pytest.raises(ValueError, match="'13'"):
        with use_openfoam_version("13"):
            pass
