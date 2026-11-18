import pytest

from pyfoamy.parsing.timedirs import (
    find_instance,
    list_time_instances,
    resolve_time_instance,
)


def _touch(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_list_time_instances_parses_numeric_dirs_and_ignores_others(
    tmp_path,
):
    for name in ("0", "0.1", "1", "constant", "system", "processor0"):
        (tmp_path / name).mkdir()
    instances = list_time_instances(tmp_path)
    assert instances == [
        (0.0, "0"),
        (0.0, "constant"),
        (0.1, "0.1"),
        (1.0, "1"),
    ]


def test_list_time_instances_prefers_real_zero_over_constant_at_tie(
    tmp_path,
):
    # Both "0" and "constant" have value 0.0 - real "0" must sort first
    # (Time::times()'s own tie-break, see the module docstring).
    (tmp_path / "0").mkdir()
    (tmp_path / "constant").mkdir()
    instances = list_time_instances(tmp_path)
    assert instances == [(0.0, "0"), (0.0, "constant")]


def test_list_time_instances_without_a_zero_directory(tmp_path):
    (tmp_path / "constant").mkdir()
    (tmp_path / "1").mkdir()
    assert list_time_instances(tmp_path) == [
        (0.0, "constant"),
        (1.0, "1"),
    ]


def test_find_instance_returns_exact_time_match(tmp_path):
    _touch(tmp_path / "constant" / "polyMesh" / "points")
    _touch(tmp_path / "0.1" / "polyMesh" / "points")
    found = find_instance(tmp_path, 0.1, "points")
    assert found == tmp_path / "0.1" / "polyMesh" / "points"


def test_find_instance_falls_back_to_an_earlier_instance(tmp_path):
    # Moving mesh: a time without "points" falls back to constant.
    _touch(tmp_path / "constant" / "polyMesh" / "points")
    (tmp_path / "0.1").mkdir()
    found = find_instance(tmp_path, 0.1, "points")
    assert found == tmp_path / "constant" / "polyMesh" / "points"


def test_find_instance_falls_back_through_multiple_earlier_instances(
    tmp_path,
):
    _touch(tmp_path / "constant" / "polyMesh" / "points")
    _touch(tmp_path / "0.1" / "polyMesh" / "points")
    (tmp_path / "0.2").mkdir()  # no points override here
    found = find_instance(tmp_path, 0.2, "points")
    assert found == tmp_path / "0.1" / "polyMesh" / "points"


def test_find_instance_ignores_instances_after_the_requested_time(
    tmp_path,
):
    _touch(tmp_path / "constant" / "polyMesh" / "points")
    _touch(tmp_path / "0.5" / "polyMesh" / "points")
    found = find_instance(tmp_path, 0.1, "points")
    # 0.5 is later than the requested time - must not be picked, even
    # though it's the only other instance with the file.
    assert found == tmp_path / "constant" / "polyMesh" / "points"


def test_find_instance_recognises_a_gzipped_file(tmp_path):
    _touch(tmp_path / "constant" / "polyMesh" / "points.gz")
    found = find_instance(tmp_path, 0.0, "points")
    # Returns the plain (non-.gz) path - listfile.py's own readers
    # already handle the .gz fallback transparently from there.
    assert found == tmp_path / "constant" / "polyMesh" / "points"


def test_find_instance_raises_when_never_found(tmp_path):
    (tmp_path / "constant").mkdir()
    with pytest.raises(FileNotFoundError):
        find_instance(tmp_path, 0.0, "points")


def test_find_instance_raises_when_no_instance_at_or_before_time(
    tmp_path,
):
    _touch(tmp_path / "0.5" / "polyMesh" / "points")
    with pytest.raises(FileNotFoundError):
        find_instance(tmp_path, 0.1, "points")


def test_resolve_time_instance_exact_match(tmp_path):
    (tmp_path / "0.1").mkdir()
    (tmp_path / "0.2").mkdir()
    assert resolve_time_instance(tmp_path, 0.1) == "0.1"


def test_resolve_time_instance_at_zero(tmp_path):
    (tmp_path / "0").mkdir()
    assert resolve_time_instance(tmp_path, 0.0) == "0"


def test_resolve_time_instance_never_matches_constant(tmp_path):
    # Unlike find_instance(), no fallback to "constant".
    (tmp_path / "constant").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_time_instance(tmp_path, 0.0)


def test_resolve_time_instance_does_not_fall_back_to_an_earlier_time(
    tmp_path,
):
    (tmp_path / "0.1").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_time_instance(tmp_path, 0.2)


def test_resolve_time_instance_raises_when_no_time_dirs_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_time_instance(tmp_path, 0.1)
