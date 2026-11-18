import json
import os
import signal
import subprocess
import time
from decimal import Decimal

import pytest

ct = pytest.importorskip("cantera")
np = pytest.importorskip("numpy")

from pyfoamy.cantera.mechanism import Mechanism  # noqa: E402
from pyfoamy.cantera.laminar_flame_speed import (  # noqa: E402
    LaminarFlameSpeedDatabase, PlausibilityStatus, _GracefulStop,
    _solve_flame_point, evaluate_gulder,
)
from pyfoamy.parsing.grammar import Dictionary  # noqa: E402

# gri30.yaml ships with Cantera. A FreeFlame solve takes 20-30 s, so
# real solves are kept to a minimum.
_MECHANISM = os.path.join(
    os.path.dirname(ct.__file__), "data", "gri30.yaml"
)
_FUEL = {"CH4": 1.0}
_OXIDANT = {"O2": 1.0, "N2": 3.76}

# Required at construction; never solved by fit/write-only tests.
_DUMMY_RANGE_KWARGS = dict(
    equivalence_ratios=[1.0], temperatures=[300.0], pressures=[101325.0],
    egr_values=[0.0],
)


def _synthetic_lfs_db(T, p, egr, phi, **true_coeffs):
    sL = evaluate_gulder(true_coeffs, p, T, phi, egr)
    results = [
        {
            "T": float(t), "p": float(pp), "egr": float(e),
            "phi": float(ph), "sL": float(s), "width": 0.0,
            "thickness": 0.0, "maxT": 0.0, "species": {},
        }
        for t, pp, e, ph, s in zip(T, p, egr, phi, sL)
    ]
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=None, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    lfs_db.results = results
    return lfs_db


@pytest.fixture
def mech():
    return Mechanism(_MECHANISM)


@pytest.fixture
def lfs_db(mech):
    return LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        equivalence_ratios=[1.0], temperatures=[300],
        pressures=[101325], egr_values=[0.0, 0.15], max_processes=2,
    )


def test_post_init_converts_decimal_ranges_to_float():
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=None, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=(Decimal("1.0e5"), Decimal("2.0e5")),
        temperatures=[Decimal("300"), Decimal("350")],
        equivalence_ratios=(Decimal("0.8"), Decimal("1.0")),
        egr_values=[Decimal("0.1")],
    )
    for values in (
        lfs_db.pressures, lfs_db.temperatures,
        lfs_db.equivalence_ratios, lfs_db.egr_values,
    ):
        assert all(type(v) is float for v in values)
    assert lfs_db.pressures == [1e5, 2e5]
    assert lfs_db.temperatures == [300.0, 350.0]
    assert lfs_db.equivalence_ratios == [0.8, 1.0]
    assert lfs_db.egr_values == [0.1]


def test_is_lambda_inverts_equivalence_ratios_on_construction():
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=None, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=[101325.0], temperatures=[300.0],
        equivalence_ratios=[0.8, Decimal("1.25")],
        egr_values=[0.0], is_lambda=True,
    )
    assert lfs_db.equivalence_ratios == pytest.approx([1.25, 0.8])
    # is_lambda must not affect any of the other ranges
    assert lfs_db.pressures == [101325.0]


def test_egr_diluent_is_pure_stoichiometric_products(mech):
    # Mixture = (fuel+oxidant)*(1-egr) + products*egr, with products at
    # phi=1 regardless of the sweep's phi; egr=1 is pure diluent.
    products = mech.complete_combustion_products(
        _FUEL, _OXIDANT, equivalence_ratio=1.0,
    )
    gas = ct.Solution(yaml=mech.mechanism_yaml)
    gas.set_equivalence_ratio(
        0.5, _FUEL, _OXIDANT, basis="mass",
        diluent=products, fraction={"diluent": 1.0},
    )
    result = gas.mass_fraction_dict(threshold=1e-9)
    for name, fraction in products.items():
        assert result.get(name, 0.0) == pytest.approx(fraction, abs=1e-9)


@pytest.mark.slow
def test_solve_flame_point_returns_physically_plausible_values(mech):
    products = mech.complete_combustion_products(
        _FUEL, _OXIDANT, equivalence_ratio=1.0,
    )
    result = _solve_flame_point(
        _MECHANISM, _FUEL, _OXIDANT, products,
        101325, 300, 1.0, 0.0, "mass", 0.001, 40, "mixture-averaged",
    )
    # Stoichiometric CH4/air at 300 K/1 atm is ~0.36-0.40 m/s; wide
    # bounds, this is a sanity check.
    assert 0.1 < result["sL"] < 1.0
    assert 1800 < result["maxT"] < 2400
    assert result["thickness"] > 0
    assert result["width"] > 0
    assert result["T"] == 300 and result["p"] == 101325
    assert result["egr"] == 0.0 and result["phi"] == 1.0
    assert "CH4" in result["species"]


def _always_failing_solve(monkeypatch, exception):
    calls = []

    def solve(self, *args, **kwargs):
        calls.append(self.grid[-1])
        raise exception

    monkeypatch.setattr(ct.FreeFlame, "solve", solve)
    return calls


def test_solve_flame_point_gives_up_after_repeated_failures(
    mech, monkeypatch,
):
    calls = _always_failing_solve(monkeypatch, ct.CanteraError("boom"))
    products = mech.complete_combustion_products(
        _FUEL, _OXIDANT, equivalence_ratio=1.0,
    )
    with pytest.warns(UserWarning, match="sL=nan"):
        result = _solve_flame_point(
            _MECHANISM, _FUEL, _OXIDANT, products,
            101325, 300, 1.0, 0.0, "mass", 0.001, 10, "mixture-averaged",
        )
    assert np.isnan(result["sL"])
    assert len(calls) == 7
    assert calls[-1] == pytest.approx(0.001 * 2 ** 6)


def test_solve_flame_point_does_not_retry_non_cantera_errors(
    mech, monkeypatch,
):
    calls = _always_failing_solve(monkeypatch, KeyError("X"))
    products = mech.complete_combustion_products(
        _FUEL, _OXIDANT, equivalence_ratio=1.0,
    )
    with pytest.raises(KeyError):
        _solve_flame_point(
            _MECHANISM, _FUEL, _OXIDANT, products,
            101325, 300, 1.0, 0.0, "mass", 0.001, 10, "mixture-averaged",
        )
    assert len(calls) == 1


@pytest.mark.slow
def test_sweep_stores_results_and_egr_reduces_flame_speed(lfs_db):
    returned = lfs_db.sweep()
    # forgetting "data = lfs_db.sweep(...)" must not lose the result
    assert lfs_db.results is returned
    assert len(returned) == 2

    by_egr = {r["egr"]: r["sL"] for r in returned}
    assert by_egr[0.15] < by_egr[0.0]  # dilution slows the flame down


@pytest.mark.slow
def test_sweep_directory_caches_and_reload_skips_recompute(
    lfs_db, tmp_path,
):
    config = dict(
        mechanism=lfs_db.mechanism, fuel=_FUEL,
        oxidant=_OXIDANT, equivalence_ratios=[1.0], temperatures=[300],
        pressures=[101325], egr_values=[0.0],
    )
    first = LaminarFlameSpeedDatabase(**config).sweep(directory=tmp_path)

    cached_files = list(tmp_path.glob("lfs_*.json"))
    assert len(cached_files) == 1
    assert not list(tmp_path.glob(".*.partial"))
    mtime_before = cached_files[0].stat().st_mtime_ns

    second = LaminarFlameSpeedDatabase(**config).sweep(directory=tmp_path)

    assert second == first
    # a cache hit must not re-solve and rewrite the file
    assert cached_files[0].stat().st_mtime_ns == mtime_before


def test_cache_filename_matches_what_sweep_would_write(mech, tmp_path):
    kwargs = dict(
        pressures=[101325.0, 2e5], temperatures=[300.0, 350.0],
        equivalence_ratios=[0.8, 1.0], egr_values=[0.0, 0.1],
    )
    actual = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT, **kwargs,
    )
    predicted = actual.cache_filename

    fuel, oxidant = actual._mass_basis_composition()
    actual_path = actual._cache_path(tmp_path, None, fuel, oxidant)

    assert predicted == actual_path.name


def test_cache_filename_is_lambda_matches_equivalent_phi_config(mech):
    kwargs = dict(
        pressures=[101325.0], temperatures=[300.0], egr_values=[0.0],
    )
    phi_name = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        equivalence_ratios=[0.5, 1.0], **kwargs,
    ).cache_filename
    lambda_name = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        equivalence_ratios=[2.0, 1.0], is_lambda=True, **kwargs,
    ).cache_filename
    assert phi_name == lambda_name


def test_cache_filename_raises_without_a_mechanism():
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=None, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    with pytest.raises(ValueError, match="mechanism"):
        lfs_db.cache_filename


def _write_synthetic_cache(mechanism, tmp_path):
    original = LaminarFlameSpeedDatabase(
        mechanism=mechanism, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=[101325.0, 2e5], temperatures=[300.0],
        equivalence_ratios=[0.8, 1.0], egr_values=[0.0],
    )
    original.results = [
        {
            "p": p, "T": 300.0, "phi": phi, "egr": 0.0, "sL": 0.3,
            "width": 0.01, "thickness": 0.0005, "maxT": 2000.0,
            "species": {"CH4": 0.05},
        }
        for p in [101325.0, 2e5] for phi in [0.8, 1.0]
    ]
    # The cache always stores the mass-basis composition, whatever
    # original.basis was.
    fuel, oxidant = original._mass_basis_composition()
    cache_path = original._cache_path(tmp_path, None, fuel, oxidant)
    original._write_cache(cache_path, original.results, fuel, oxidant)
    return original, cache_path, fuel, oxidant


def test_from_json_reconstructs_configuration_and_results(mech, tmp_path):
    original, cache_path, fuel, oxidant = _write_synthetic_cache(
        mech, tmp_path,
    )

    loaded = LaminarFlameSpeedDatabase.from_json(cache_path)

    assert loaded.mechanism is None
    assert loaded.fuel == fuel
    assert loaded.oxidant == oxidant
    assert loaded.basis == "mass"
    assert loaded.pressures == [101325.0, 2e5]
    assert loaded.temperatures == [300.0]
    assert loaded.equivalence_ratios == [0.8, 1.0]
    assert loaded.egr_values == [0.0]
    assert loaded.transport_model == original.transport_model
    assert loaded.results == original.results

    # no Cantera needed for these
    _, quality = loaded.fit_gulder()
    assert quality["n_points"] == 4


def test_from_json_mechanism_needed_only_for_sweeping(mech, tmp_path):
    _, cache_path, _, _ = _write_synthetic_cache(mech, tmp_path)
    loaded = LaminarFlameSpeedDatabase.from_json(cache_path)

    with pytest.raises(ValueError, match="mechanism"):
        loaded.sweep(directory=tmp_path)
    with pytest.raises(ValueError, match="mechanism"):
        loaded.solve_point(p=101325.0, T=300.0, phi=1.0, egr=0.0)

    restored = LaminarFlameSpeedDatabase.from_json(
        cache_path, mechanism=mech, transport_model="const",
        max_processes=5,
    )
    assert restored.mechanism is mech
    assert restored.transport_model == "const"
    assert restored.max_processes == 5


@pytest.mark.slow
def test_from_json_mechanism_true_reconstructs_from_embedded_yaml(
    mech, tmp_path,
):
    # The cache's embedded mechanism_yaml is enough; no mechanism file.
    _, cache_path, _, _ = _write_synthetic_cache(mech, tmp_path)
    restored = LaminarFlameSpeedDatabase.from_json(
        cache_path, mechanism=True,
    )
    assert restored.mechanism is not None
    assert restored.mechanism is not mech
    assert (
        restored.mechanism.gas.species_names == mech.gas.species_names
    )
    result = restored.solve_point(p=101325.0, T=300.0, phi=1.0, egr=0.0)
    assert 0.1 < result["sL"] < 1.0


def test_graceful_stop_first_signal_sets_requested_without_raising():
    stop = _GracefulStop("stopping")
    with stop:
        stop._handle(signal.SIGINT, None)
        assert stop.requested is True


def test_graceful_stop_second_signal_raises_keyboard_interrupt():
    stop = _GracefulStop("stopping")
    with stop:
        stop._handle(signal.SIGINT, None)
        with pytest.raises(KeyboardInterrupt):
            stop._handle(signal.SIGINT, None)


def test_graceful_stop_restores_previous_handler_on_exit():
    previous = signal.getsignal(signal.SIGINT)
    with _GracefulStop("stopping"):
        assert signal.getsignal(signal.SIGINT) is not previous
    assert signal.getsignal(signal.SIGINT) == previous


def test_sweep_stops_early_after_single_interrupt_and_leaves_resumable(
    mech, tmp_path, monkeypatch,
):
    # A real SIGINT to this process (where _GracefulStop handles it),
    # sent from a fake worker. max_processes=1: one point in flight.
    main_pid = os.getpid()
    call_log = tmp_path / "calls.log"

    def fake_solve_flame_point(
        mechanism_path, fuel, oxidant, products, p, T, phi, egr,
        basis, initial_width, num_points, transport_model,
        loglevel=0, log=False,
    ):
        with call_log.open("a") as handle:
            handle.write(f"{p}\n")
        if p == 101325.0:
            os.kill(main_pid, signal.SIGINT)
            # Give the polling loop time to notice first.
            time.sleep(0.3)
        return {
            "p": p, "T": T, "phi": phi, "egr": egr, "sL": 0.3,
            "width": 0.01, "thickness": 0.001, "maxT": 2000.0,
            "species": {},
        }

    monkeypatch.setattr(
        "pyfoamy.cantera.laminar_flame_speed._solve_flame_point",
        fake_solve_flame_point,
    )

    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=[101325.0, 2e5, 5e5], temperatures=[300.0],
        equivalence_ratios=[1.0], egr_values=[0.0], max_processes=1,
    )
    results = lfs_db.sweep(directory=tmp_path)

    solved = {float(line) for line in call_log.read_text().splitlines()}
    assert solved == {101325.0}
    assert results == lfs_db.results
    assert len(results) == 1
    # Stopped early: no cache, partial file kept for a resume.
    assert not list(tmp_path.glob("lfs_*.json"))
    assert list(tmp_path.glob(".*.partial"))

    call_log.write_text("")  # reset before the resumed attempt
    resumed = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=[101325.0, 2e5, 5e5], temperatures=[300.0],
        equivalence_ratios=[1.0], egr_values=[0.0], max_processes=1,
    ).sweep(directory=tmp_path)

    assert {r["p"] for r in resumed} == {101325.0, 2e5, 5e5}
    # only the two still-missing points get (re-)solved
    resumed_calls = {
        float(line) for line in call_log.read_text().splitlines()
    }
    assert resumed_calls == {2e5, 5e5}
    assert list(tmp_path.glob("lfs_*.json"))
    assert not list(tmp_path.glob(".*.partial"))


def test_sweep_aborts_immediately_on_second_interrupt(
    mech, tmp_path, monkeypatch,
):
    main_pid = os.getpid()

    def fake_solve_flame_point(
        mechanism_path, fuel, oxidant, products, p, T, phi, egr,
        basis, initial_width, num_points, transport_model,
        loglevel=0, log=False,
    ):
        if p == 101325.0:
            os.kill(main_pid, signal.SIGINT)
            time.sleep(0.1)  # let the first signal actually be handled
            os.kill(main_pid, signal.SIGINT)
            time.sleep(0.3)
        return {
            "p": p, "T": T, "phi": phi, "egr": egr, "sL": 0.3,
            "width": 0.01, "thickness": 0.001, "maxT": 2000.0,
            "species": {},
        }

    monkeypatch.setattr(
        "pyfoamy.cantera.laminar_flame_speed._solve_flame_point",
        fake_solve_flame_point,
    )

    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=[101325.0, 2e5, 5e5], temperatures=[300.0],
        equivalence_ratios=[1.0], egr_values=[0.0], max_processes=1,
    )
    with pytest.raises(KeyboardInterrupt):
        lfs_db.sweep(directory=tmp_path)


def test_sweep_checkpoints_partial_progress_before_a_crash(
    mech, tmp_path, monkeypatch,
):
    # Tests checkpointing (imap_unordered + per-point _append_partial),
    # not the solver; max_processes=1 makes the crash point the last.
    def fake_solve_flame_point(
        mechanism_path, fuel, oxidant, products, p, T, phi, egr,
        basis, initial_width, num_points, transport_model,
        loglevel=0, log=False,
    ):
        if p == 5e5:
            raise RuntimeError("simulated crash")
        return {
            "p": p, "T": T, "phi": phi, "egr": egr, "sL": 0.3,
            "width": 0.01, "thickness": 0.001, "maxT": 2000.0,
            "species": {},
        }

    monkeypatch.setattr(
        "pyfoamy.cantera.laminar_flame_speed._solve_flame_point",
        fake_solve_flame_point,
    )

    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=[101325.0, 2e5, 5e5], temperatures=[300.0],
        equivalence_ratios=[1.0], egr_values=[0.0], max_processes=1,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        lfs_db.sweep(directory=tmp_path)

    # the real cache file must never appear - the sweep never finished
    assert not list(tmp_path.glob("lfs_*.json"))

    partial_files = list(tmp_path.glob(".*.partial"))
    assert len(partial_files) == 1
    assert partial_files[0].name.startswith(".")  # hidden file
    # JSON Lines: one point per line.
    lines = partial_files[0].read_text().splitlines()
    checkpoint = [json.loads(line) for line in lines]
    assert {r["p"] for r in checkpoint} == {101325.0, 2e5}


def test_sweep_resumes_from_partial_progress_after_a_crash(
    mech, tmp_path, monkeypatch,
):
    # max_processes=1 keeps the point order deterministic. The fake
    # solver runs in a forked worker, so crash/call state is tracked in
    # files; a closed-over Python object would not propagate back.
    crash_marker = tmp_path / "crashed_once"
    call_log = tmp_path / "calls.log"

    def fake_solve_flame_point(
        mechanism_path, fuel, oxidant, products, p, T, phi, egr,
        basis, initial_width, num_points, transport_model,
        loglevel=0, log=False,
    ):
        with call_log.open("a") as handle:
            handle.write(f"{p}\n")
        if p == 5e5 and not crash_marker.is_file():
            crash_marker.write_text("1")
            raise RuntimeError("simulated crash")
        return {
            "p": p, "T": T, "phi": phi, "egr": egr, "sL": 0.3,
            "width": 0.01, "thickness": 0.001, "maxT": 2000.0,
            "species": {},
        }

    monkeypatch.setattr(
        "pyfoamy.cantera.laminar_flame_speed._solve_flame_point",
        fake_solve_flame_point,
    )

    config = dict(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=[101325.0, 2e5, 5e5], temperatures=[300.0],
        equivalence_ratios=[1.0], egr_values=[0.0], max_processes=1,
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        LaminarFlameSpeedDatabase(**config).sweep(directory=tmp_path)
    first_attempt = [
        float(line) for line in call_log.read_text().splitlines()
    ]
    assert sorted(first_attempt) == [101325.0, 2e5, 5e5]

    call_log.write_text("")  # reset before the resumed attempt
    resumed = LaminarFlameSpeedDatabase(**config).sweep(directory=tmp_path)

    assert {r["p"] for r in resumed} == {101325.0, 2e5, 5e5}
    # Only the crashed point is re-solved; the others come from the
    # partial file.
    second_attempt = [
        float(line) for line in call_log.read_text().splitlines()
    ]
    assert second_attempt == [5e5]
    assert list(tmp_path.glob("lfs_*.json"))
    assert not list(tmp_path.glob(".*.partial"))


def test_read_partial_returns_empty_for_a_fully_corrupted_file(
    mech, tmp_path,
):
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    partial_path = tmp_path / ".broken.json.partial"
    partial_path.write_text("{not valid json")

    assert lfs_db._read_partial(partial_path, log=False) == []


def test_read_partial_keeps_valid_lines_before_a_truncated_one(
    mech, tmp_path,
):
    # A crash mid-write loses only the line being written.
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    good_point = {
        "p": 101325.0, "T": 300.0, "phi": 1.0, "egr": 0.0, "sL": 0.3,
        "width": 0.01, "thickness": 0.001, "maxT": 2000.0, "species": {},
    }
    partial_path = tmp_path / ".resumable.json.partial"
    partial_path.write_text(json.dumps(good_point) + "\n{truncated")

    result = lfs_db._read_partial(partial_path, log=False)
    assert result == [good_point]


def test_sweep_recomputes_everything_if_partial_is_corrupted(
    mech, tmp_path, monkeypatch,
):
    def fake_solve_flame_point(
        mechanism_path, fuel, oxidant, products, p, T, phi, egr,
        basis, initial_width, num_points, transport_model,
        loglevel=0, log=False,
    ):
        return {
            "p": p, "T": T, "phi": phi, "egr": egr, "sL": 0.3,
            "width": 0.01, "thickness": 0.001, "maxT": 2000.0,
            "species": {},
        }

    monkeypatch.setattr(
        "pyfoamy.cantera.laminar_flame_speed._solve_flame_point",
        fake_solve_flame_point,
    )

    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=[101325.0, 2e5], temperatures=[300.0],
        equivalence_ratios=[1.0], egr_values=[0.0], max_processes=1,
    )
    fuel, oxidant = lfs_db._mass_basis_composition()
    cache_path = lfs_db._cache_path(tmp_path, None, fuel, oxidant)
    partial_path = cache_path.with_name(f".{cache_path.name}.partial")
    partial_path.write_text("{not valid json")

    result = lfs_db.sweep(directory=tmp_path)

    assert {r["p"] for r in result} == {101325.0, 2e5}
    assert not partial_path.is_file()


def test_to_array_raises_before_any_sweep(mech):
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    with pytest.raises(ValueError, match="sweep"):
        lfs_db.to_array()


@pytest.mark.slow
def test_to_array_matches_results_and_excludes_species(lfs_db):
    lfs_db.sweep()
    columns, array = lfs_db.to_array()
    assert columns == [
        "p", "T", "phi", "egr", "sL", "width", "thickness", "maxT",
    ]
    assert array.shape == (2, 8)
    sl_col = columns.index("sL")
    assert sorted(array[:, sl_col]) == sorted(
        r["sL"] for r in lfs_db.results
    )


def _grid_lfs_db(true_coeffs):
    pressures = [1e5, 2e5]
    temperatures = [300.0, 400.0]
    egr_values = [0.0, 0.1]
    phis = [0.8, 1.0, 1.2, 1.4]
    results = []
    for p in pressures:
        for T in temperatures:
            for egr in egr_values:
                for phi in phis:
                    sL = float(evaluate_gulder(true_coeffs, p, T, phi, egr))
                    results.append({
                        "p": p, "T": T, "phi": phi, "egr": egr, "sL": sL,
                        "width": 0.0, "thickness": 0.0, "maxT": 0.0,
                        "species": {},
                    })
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=None, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    lfs_db.results = results
    return lfs_db, pressures, temperatures, egr_values, phis


def test_curve_extracts_and_sorts_matching_points():
    true = {
        "W": 0.35, "eta": 0.2, "xi": 4.5, "alpha": 1.2, "beta": -0.3,
        "f": 0.8,
    }
    lfs_db, pressures, temperatures, egr_values, phis = _grid_lfs_db(true)

    x, sL = lfs_db.curve("phi", {"p": 2e5, "T": 400.0, "egr": 0.1})

    assert list(x) == sorted(phis)
    expected = evaluate_gulder(true, 2e5, 400.0, x, 0.1)
    assert sL == pytest.approx(expected)


def test_curve_raises_for_unknown_variable():
    lfs_db, *_ = _grid_lfs_db({
        "W": 0.35, "eta": 0.2, "xi": 4.5, "alpha": 1.2, "beta": -0.3,
        "f": 0.8,
    })
    with pytest.raises(ValueError, match="variable must be one of"):
        lfs_db.curve("bogus", {"p": 1e5, "T": 300.0, "egr": 0.0})


def test_curve_raises_when_fixed_has_wrong_axes():
    lfs_db, *_ = _grid_lfs_db({
        "W": 0.35, "eta": 0.2, "xi": 4.5, "alpha": 1.2, "beta": -0.3,
        "f": 0.8,
    })
    with pytest.raises(ValueError, match="fixed must specify exactly"):
        lfs_db.curve("phi", {"p": 1e5, "T": 300.0})


def test_curve_raises_when_nothing_matches():
    lfs_db, *_ = _grid_lfs_db({
        "W": 0.35, "eta": 0.2, "xi": 4.5, "alpha": 1.2, "beta": -0.3,
        "f": 0.8,
    })
    with pytest.raises(ValueError, match="no results match"):
        lfs_db.curve("phi", {"p": 999.0, "T": 300.0, "egr": 0.0})


def test_curve_raises_before_any_sweep(mech):
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    with pytest.raises(ValueError, match="sweep"):
        lfs_db.curve("phi", {"p": 1e5, "T": 300.0, "egr": 0.0})


def test_evaluate_gulder_matches_fit_on_noise_free_data():
    true = {
        "W": 0.4, "eta": 0.3, "xi": 5.0, "alpha": 1.8, "beta": -0.3,
        "f": 1.2,
    }
    rng = np.random.default_rng(7)
    n = 200
    phi = rng.uniform(0.7, 1.3, n)
    T = rng.uniform(300, 600, n)
    p = rng.uniform(1e5, 3e5, n)
    egr = rng.uniform(0.0, 0.2, n)
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)

    coeffs, _ = lfs_db.fit_gulder()
    predicted = evaluate_gulder(coeffs, p, T, phi, egr)
    actual = evaluate_gulder(true, p, T, phi, egr)
    assert predicted == pytest.approx(actual, rel=1e-2)


def test_evaluate_gulder_broadcasts_scalar_and_array_inputs():
    coeffs = {
        "W": 0.35, "eta": 0.2, "xi": 4.5, "alpha": 1.2, "beta": -0.3,
        "f": 0.8,
    }
    scalar = evaluate_gulder(coeffs, 1e5, 300.0, 1.0, 0.0)
    assert np.ndim(scalar) == 0

    phi_array = np.array([0.8, 1.0, 1.2])
    array_result = evaluate_gulder(coeffs, 1e5, 300.0, phi_array, 0.0)
    assert array_result.shape == phi_array.shape
    assert array_result[1] == pytest.approx(float(scalar))


def test_write_csv_produces_a_readable_header_and_values(tmp_path):
    results = [
        {"T": 300, "p": 101325, "egr": 0.0, "phi": 1.0, "sL": 0.36,
         "width": 0.004, "thickness": 5e-4, "maxT": 2226.0,
         "species": {"CH4": 0.055, "O2": 0.22}},
        {"T": 300, "p": 101325, "egr": 0.15, "phi": 1.0, "sL": 0.30,
         "width": 0.005, "thickness": 6e-4, "maxT": 2000.0,
         "species": {"CH4": 0.047, "CO2": 0.03}},
    ]
    path = tmp_path / "lfs.csv"
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=None, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    lfs_db.results = results
    lfs_db.write_csv(path)

    header = path.read_text().splitlines()[0].split(",")
    assert header == [
        "p", "T", "phi", "egr", "sL", "width", "thickness", "maxT",
        "CH4", "CO2", "O2",
    ]
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    assert data.shape == (2, 11)
    # Missing species are 0.0, not NaN or a shifted column.
    assert data[0, header.index("CO2")] == 0.0
    assert data[1, header.index("O2")] == 0.0
    assert data[0, header.index("CH4")] == pytest.approx(0.055)


def test_fit_gulder_recovers_known_coefficients_all_axes_varying():
    rng = np.random.default_rng(0)
    n = 300
    phi = rng.uniform(0.7, 1.3, n)
    T = rng.uniform(300, 600, n)
    p = rng.uniform(1e5, 3e5, n)
    egr = rng.uniform(0.0, 0.2, n)
    true = {
        "W": 0.4, "eta": 0.3, "xi": 5.0, "alpha": 1.8, "beta": -0.3,
        "f": 1.2,
    }
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    coeffs, quality = lfs_db.fit_gulder()
    for key, value in true.items():
        assert coeffs[key] == pytest.approx(value, rel=1e-2)
    assert quality["n_points"] == n
    assert quality["r2"] == pytest.approx(1.0, abs=1e-3)


def test_fit_gulder_fixes_eta_xi_when_phi_is_constant():
    rng = np.random.default_rng(1)
    n = 100
    phi = np.full(n, 1.0)
    T = rng.uniform(300, 600, n)
    p = rng.uniform(1e5, 3e5, n)
    egr = rng.uniform(0.0, 0.2, n)
    true = {
        "W": 0.5, "eta": 0.0, "xi": 0.0, "alpha": 1.5, "beta": -0.2,
        "f": 0.8,
    }
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    coeffs, quality = lfs_db.fit_gulder()
    assert coeffs["eta"] == 0.0
    assert coeffs["xi"] == 0.0
    for key in ("W", "alpha", "beta", "f"):
        assert coeffs[key] == pytest.approx(true[key], rel=1e-2)
    assert quality["r2"] == pytest.approx(1.0, abs=1e-3)


def test_fit_gulder_fixes_alpha_when_temperature_is_constant():
    rng = np.random.default_rng(2)
    n = 100
    phi = rng.uniform(0.7, 1.3, n)
    T = np.full(n, 350.0)
    p = rng.uniform(1e5, 3e5, n)
    egr = rng.uniform(0.0, 0.2, n)
    true = {
        "W": 0.35, "eta": 0.2, "xi": 4.5, "alpha": 0.0, "beta": -0.4,
        "f": 0.5,
    }
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    coeffs, quality = lfs_db.fit_gulder()
    assert coeffs["alpha"] == 0.0
    for key in ("W", "eta", "xi", "beta", "f"):
        assert coeffs[key] == pytest.approx(true[key], rel=1e-2)
    assert quality["r2"] == pytest.approx(1.0, abs=1e-3)


def test_fit_gulder_fixes_beta_when_pressure_is_constant():
    rng = np.random.default_rng(3)
    n = 100
    phi = rng.uniform(0.7, 1.3, n)
    T = rng.uniform(300, 600, n)
    p = np.full(n, 1.5e5)
    egr = rng.uniform(0.0, 0.2, n)
    true = {
        "W": 0.35, "eta": 0.2, "xi": 4.5, "alpha": 1.2, "beta": 0.0,
        "f": 0.5,
    }
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    coeffs, quality = lfs_db.fit_gulder()
    assert coeffs["beta"] == 0.0
    for key in ("W", "eta", "xi", "alpha", "f"):
        assert coeffs[key] == pytest.approx(true[key], rel=1e-2)
    assert quality["r2"] == pytest.approx(1.0, abs=1e-3)


def test_fit_gulder_fixes_f_when_egr_is_constant():
    rng = np.random.default_rng(4)
    n = 100
    phi = rng.uniform(0.7, 1.3, n)
    T = rng.uniform(300, 600, n)
    p = rng.uniform(1e5, 3e5, n)
    egr = np.zeros(n)
    true = {
        "W": 0.35, "eta": 0.2, "xi": 4.5, "alpha": 1.2, "beta": -0.3,
        "f": 0.0,
    }
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    coeffs, quality = lfs_db.fit_gulder()
    assert coeffs["f"] == 0.0
    for key in ("W", "eta", "xi", "alpha", "beta"):
        assert coeffs[key] == pytest.approx(true[key], rel=1e-2)
    assert quality["r2"] == pytest.approx(1.0, abs=1e-3)


def test_fit_gulder_quality_excludes_failed_points():
    rng = np.random.default_rng(5)
    # n=1000, not 100: the nan-floored points are log-space outliers
    # that can flip a coefficient's sign among too few others.
    n = 1000
    phi = rng.uniform(0.7, 1.3, n)
    T = rng.uniform(300, 600, n)
    p = rng.uniform(1e5, 3e5, n)
    egr = rng.uniform(0.0, 0.2, n)
    true = {
        "W": 0.35, "eta": 0.2, "xi": 4.5, "alpha": 1.2, "beta": -0.3,
        "f": 0.8,
    }
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    # Points with sL=nan must be excluded from n_points and the error
    # statistics.
    failed = [3, 17, 42]
    for index in failed:
        lfs_db.results[index]["sL"] = float("nan")

    _, quality = lfs_db.fit_gulder()
    assert quality["n_points"] == n - len(failed)
    assert np.isfinite(quality["relative_error_mean"])
    assert np.isfinite(quality["relative_error_std"])
    assert np.isfinite(quality["relative_error_max"])
    assert np.isfinite(quality["r2"])


def test_fit_gulder_raises_for_physically_implausible_coefficients():
    rng = np.random.default_rng(6)
    n = 300
    phi = rng.uniform(0.7, 1.3, n)
    T = rng.uniform(300, 600, n)
    p = rng.uniform(1e5, 3e5, n)
    egr = rng.uniform(0.0, 0.2, n)
    # xi<0 and beta>0 are implausible on purpose.
    true = {
        "W": 0.35, "eta": 0.2, "xi": -4.5, "alpha": 1.2, "beta": 0.3,
        "f": 0.8,
    }
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    with pytest.raises(ValueError, match="xi=.*beta="):
        lfs_db.fit_gulder()


def test_fit_gulder_raises_with_helpful_note_when_phi_is_underdetermined():
    # Real NH3/air sL with only 3 distinct phi values: eta/xi are not
    # identifiable, so fit_gulder() raises "xi < 0"; the message must
    # name the narrow phi range as the likely cause.
    points = [
        {"T": 700.0, "p": 6000000.0, "phi": 0.9523809523809523,
         "sL": 0.14201508811625077},
        {"T": 700.0, "p": 6000000.0, "phi": 1.0, "sL": 0.15316626526450436},
        {"T": 700.0, "p": 6000000.0, "phi": 1.0526315789473684,
         "sL": 0.1712013820701542},
        {"T": 950.0, "p": 6000000.0, "phi": 0.9523809523809523,
         "sL": 0.38525763954070413},
        {"T": 950.0, "p": 6000000.0, "phi": 1.0, "sL": 0.41018719095531203},
        {"T": 950.0, "p": 6000000.0, "phi": 1.0526315789473684,
         "sL": 0.4434074848232831},
        {"T": 1200.0, "p": 6000000.0, "phi": 0.9523809523809523,
         "sL": 0.9156823856016589},
        {"T": 1200.0, "p": 6000000.0, "phi": 1.0, "sL": 0.9619619609421913},
        {"T": 1200.0, "p": 6000000.0, "phi": 1.0526315789473684,
         "sL": 1.0111540624691668},
        {"T": 700.0, "p": 8500000.0, "phi": 0.9523809523809523,
         "sL": 0.13061699466757512},
        {"T": 700.0, "p": 8500000.0, "phi": 1.0, "sL": 0.14058685157045214},
        {"T": 700.0, "p": 8500000.0, "phi": 1.0526315789473684,
         "sL": 0.15744037772089983},
        {"T": 950.0, "p": 8500000.0, "phi": 0.9523809523809523,
         "sL": 0.35610595048042337},
        {"T": 950.0, "p": 8500000.0, "phi": 1.0, "sL": 0.3788639325992005},
        {"T": 950.0, "p": 8500000.0, "phi": 1.0526315789473684,
         "sL": 0.40967100103449705},
        {"T": 1200.0, "p": 8500000.0, "phi": 0.9523809523809523,
         "sL": 0.8538080627870546},
        {"T": 1200.0, "p": 8500000.0, "phi": 1.0, "sL": 0.8957969080009108},
        {"T": 1200.0, "p": 8500000.0, "phi": 1.0526315789473684,
         "sL": 0.9433368749612888},
        {"T": 700.0, "p": 11000000.0, "phi": 0.9523809523809523,
         "sL": 0.12414937067558368},
        {"T": 700.0, "p": 11000000.0, "phi": 1.0, "sL": 0.13318317148977082},
        {"T": 700.0, "p": 11000000.0, "phi": 1.0526315789473684,
         "sL": 0.14876360493360977},
        {"T": 950.0, "p": 11000000.0, "phi": 0.9523809523809523,
         "sL": 0.3374147697623337},
        {"T": 950.0, "p": 11000000.0, "phi": 1.0, "sL": 0.3587192572378178},
        {"T": 950.0, "p": 11000000.0, "phi": 1.0526315789473684,
         "sL": 0.38784331136849765},
        {"T": 1200.0, "p": 11000000.0, "phi": 0.9523809523809523,
         "sL": 0.8133425378970344},
        {"T": 1200.0, "p": 11000000.0, "phi": 1.0, "sL": 0.8539599671908131},
        {"T": 1200.0, "p": 11000000.0, "phi": 1.0526315789473684,
         "sL": 0.897034057512932},
    ]
    for point in points:
        point.update(
            egr=0.0, width=0.0, thickness=0.0, maxT=0.0, species={},
        )
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=None, fuel={"NH3": 1.0}, oxidant={"O2": 1.0, "N2": 3.76},
        **_DUMMY_RANGE_KWARGS,
    )
    lfs_db.results = points

    with pytest.raises(ValueError, match="xi=.*distinct equivalence ratio"):
        lfs_db.fit_gulder()


def test_fit_gulder_still_raises_plainly_when_phi_is_well_identified():
    # With a wide phi range a negative xi is a real violation, so no
    # identifiability note.
    rng = np.random.default_rng(7)
    n = 300
    phi = rng.uniform(0.7, 1.3, n)
    T = rng.uniform(300, 600, n)
    p = rng.uniform(1e5, 3e5, n)
    egr = rng.uniform(0.0, 0.2, n)
    true = {
        "W": 0.35, "eta": 0.2, "xi": -4.5, "alpha": 1.2, "beta": -0.3,
        "f": 0.8,
    }
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    with pytest.raises(ValueError, match="xi=") as excinfo:
        lfs_db.fit_gulder()
    assert "distinct equivalence ratio" not in str(excinfo.value)


def test_fit_gulder_recovers_known_coefficients_with_high_egr_data():
    # egr=0.5 makes f >= 2 infeasible (1 - f*egr <= 0); the nan these
    # give must not win np.argmin.
    rng = np.random.default_rng(7)
    n = 300
    phi = rng.uniform(0.7, 1.3, n)
    T = rng.uniform(300, 600, n)
    p = rng.uniform(1e5, 3e5, n)
    egr = rng.uniform(0.0, 0.5, n)
    true = {
        "W": 0.4, "eta": 0.3, "xi": 5.0, "alpha": 1.8, "beta": -0.3,
        "f": 1.2,
    }
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    coeffs, quality = lfs_db.fit_gulder()
    for key, value in true.items():
        assert coeffs[key] == pytest.approx(value, rel=1e-2)
    assert quality["r2"] == pytest.approx(1.0, abs=1e-3)


def test_fit_gulder_raises_when_f_bounds_entirely_infeasible():
    # egr=0.5 needs f < 2; f_bounds=(3, 5) has no feasible candidate.
    rng = np.random.default_rng(8)
    n = 100
    phi = np.full(n, 1.0)
    T = np.full(n, 400.0)
    p = np.full(n, 1e5)
    egr = rng.uniform(0.0, 0.5, n)
    true = {"W": 0.4, "eta": 0.0, "xi": 0.0, "alpha": 0.0, "beta": 0.0,
            "f": 1.2}
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    with pytest.raises(ValueError, match="f_bounds"):
        lfs_db.fit_gulder(f_bounds=(3.0, 5.0))


def test_fit_gulder_egr_restriction_fits_only_the_given_window():
    rng = np.random.default_rng(9)
    n = 300
    phi = rng.uniform(0.7, 1.3, n)
    T = rng.uniform(300, 600, n)
    p = rng.uniform(1e5, 3e5, n)
    egr = rng.uniform(0.0, 0.5, n)
    true = {
        "W": 0.4, "eta": 0.3, "xi": 5.0, "alpha": 1.8, "beta": -0.3,
        "f": 1.2,
    }
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    coeffs, quality = lfs_db.fit_gulder(egr=(0.0, 0.1))
    assert quality["n_points"] == int(np.sum(egr <= 0.1))
    for key, value in true.items():
        assert coeffs[key] == pytest.approx(value, rel=1e-2)


def test_fit_gulder_restriction_on_a_constant_axis_excludes_it():
    # A single phi makes it constant: eta/xi are fixed at 0, not fit.
    rng = np.random.default_rng(10)
    n = 300
    # Discrete, so some points are exactly phi=1.0.
    phi = rng.choice([0.8, 1.0, 1.2], size=n)
    T = rng.uniform(300, 600, n)
    p = rng.uniform(1e5, 3e5, n)
    egr = rng.uniform(0.0, 0.2, n)
    true = {
        "W": 0.4, "eta": 0.3, "xi": 5.0, "alpha": 1.8, "beta": -0.3,
        "f": 1.2,
    }
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    coeffs, quality = lfs_db.fit_gulder(phi=(1.0, 1.0))
    assert quality["n_points"] == int(np.sum(phi == 1.0))
    assert coeffs["eta"] == 0.0
    assert coeffs["xi"] == 0.0


def test_fit_gulder_raises_when_restrictions_exclude_every_point():
    rng = np.random.default_rng(11)
    n = 100
    phi = np.full(n, 1.0)
    T = np.full(n, 400.0)
    p = np.full(n, 1e5)
    egr = rng.uniform(0.0, 0.1, n)
    true = {"W": 0.4, "eta": 0.0, "xi": 0.0, "alpha": 0.0, "beta": 0.0,
            "f": 0.5}
    lfs_db = _synthetic_lfs_db(T, p, egr, phi, **true)
    with pytest.raises(ValueError, match="exclude every point"):
        lfs_db.fit_gulder(egr=(0.5, 0.9))


def test_fit_gulder_raises_before_any_sweep(mech):
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    with pytest.raises(ValueError, match="sweep"):
        lfs_db.fit_gulder()


def _write_gulder_cache(mech, tmp_path, true_coeffs, T, p, egr, phi):
    sL = evaluate_gulder(true_coeffs, p, T, phi, egr)
    results = [
        {
            "T": float(t), "p": float(pp), "egr": float(e),
            "phi": float(ph), "sL": float(s), "width": 0.0,
            "thickness": 0.0, "maxT": 0.0, "species": {},
        }
        for t, pp, e, ph, s in zip(T, p, egr, phi, sL)
    ]
    writer_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    cache_path = tmp_path / writer_db.cache_filename
    fuel, oxidant = writer_db._mass_basis_composition()
    writer_db._write_cache(cache_path, results, fuel, oxidant)
    return results


def test_fit_gulder_loads_from_directory_when_results_is_none(
    mech, tmp_path,
):
    rng = np.random.default_rng(12)
    n = 300
    phi = rng.uniform(0.7, 1.3, n)
    T = rng.uniform(300, 600, n)
    p = rng.uniform(1e5, 3e5, n)
    egr = rng.uniform(0.0, 0.2, n)
    true = {
        "W": 0.4, "eta": 0.3, "xi": 5.0, "alpha": 1.8, "beta": -0.3,
        "f": 1.2,
    }
    _write_gulder_cache(mech, tmp_path, true, T, p, egr, phi)

    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    assert lfs_db.results is None
    coeffs, quality = lfs_db.fit_gulder(directory=tmp_path)

    assert lfs_db.results is not None
    assert quality["n_points"] == n
    for key, value in true.items():
        assert coeffs[key] == pytest.approx(value, rel=1e-2)


def test_fit_gulder_directory_ignored_when_results_already_set(
    mech, tmp_path,
):
    # The cache directory is irrelevant once self.results is set.
    lfs_db = _synthetic_lfs_db(
        [300.0, 400.0], [101325.0, 101325.0], [0.0, 0.0], [1.0, 1.0],
        W=0.4, eta=0.0, xi=0.0, alpha=0.0, beta=0.0, f=0.0,
    )
    coeffs, _ = lfs_db.fit_gulder(directory=tmp_path / "does-not-exist")
    assert coeffs["W"] == pytest.approx(0.4, rel=1e-2)


def test_fit_gulder_raises_a_clear_error_for_a_missing_cache_file(
    mech, tmp_path,
):
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    with pytest.raises(ValueError, match="no cache file found"):
        lfs_db.fit_gulder(directory=tmp_path)


def _piecewise_phi_lfs_db(
    phi_nodes, pressures, temperatures, egr_values, su0_fn, f_fn,
):
    results = []
    for phi in phi_nodes:
        for p in pressures:
            for T in temperatures:
                sL0 = su0_fn(phi, p, T)
                f = f_fn(phi, p, T)
                for egr in egr_values:
                    results.append({
                        "T": float(T), "p": float(p), "egr": float(egr),
                        "phi": float(phi),
                        "sL": float(sL0 * (1 - f * egr)),
                        "width": 0.0, "thickness": 0.0, "maxT": 0.0,
                        "species": {},
                    })
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=None, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    lfs_db.results = results
    return lfs_db


def test_fit_piecewise_phi_recovers_known_su0_and_f():
    def su0_fn(phi, p, T):
        return 0.1 * phi + 1e-3 * (p / 1e5) + 1e-4 * T

    def f_fn(phi, p, T):
        return 1.0 + 0.1 * phi

    pressures = [1e5, 2e5]
    temperatures = [300.0, 400.0]
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[0.8, 1.0], pressures=pressures,
        temperatures=temperatures,
        egr_values=[0.0, 0.02, 0.05, 0.08, 0.10],
        su0_fn=su0_fn, f_fn=f_fn,
    )
    table, quality = lfs_db.fit_piecewise_phi()

    assert set(table) == {"phi0", "phi1"}
    assert table["phi0"]["Phi"] == 0.8
    assert table["phi1"]["Phi"] == 1.0
    for key, phi in (("phi0", 0.8), ("phi1", 1.0)):
        for entry, fn in (("pTu", su0_fn), ("f", f_fn)):
            grid = [
                [fn(phi, p, T) for T in temperatures] for p in pressures
            ]
            assert table[key][entry]["x"] == pressures
            assert table[key][entry]["y"] == temperatures
            assert table[key][entry]["outOfBounds"] == "clamp"
            assert np.array(table[key][entry]["values"]) == (
                pytest.approx(np.array(grid))
            )
    assert quality["n_corners"] == 8
    assert quality["relative_error_max"] < 1e-9


def test_fit_piecewise_phi_egr_max_excludes_points_beyond_the_window():
    # Beyond egr=0.10 one point is corrupted (sL rises with egr): the
    # default egr_max ignores it, a wider one trips the negative-f check.
    true_f = 2.0

    def su0_fn(phi, p, T):
        return 0.5

    def f_fn(phi, p, T):
        return true_f

    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[1.0], pressures=[1e5], temperatures=[300.0],
        egr_values=[0.0, 0.05, 0.10], su0_fn=su0_fn, f_fn=f_fn,
    )
    lfs_db.results.append({
        "T": 300.0, "p": 1e5, "egr": 0.20, "phi": 1.0, "sL": 0.9,
        "width": 0.0, "thickness": 0.0, "maxT": 0.0, "species": {},
    })

    table, _ = lfs_db.fit_piecewise_phi(egr_max=0.10)
    assert np.array(table["phi0"]["f"]["values"]) == (
        pytest.approx(np.array([[true_f]]))
    )

    with pytest.raises(ValueError, match="f < 0"):
        lfs_db.fit_piecewise_phi(egr_max=0.20)


def test_fit_piecewise_phi_raises_for_missing_baseline():
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[1.0], pressures=[1e5], temperatures=[300.0],
        egr_values=[0.02, 0.05],  # no egr=0.0 point at all
        su0_fn=lambda phi, p, T: 0.5, f_fn=lambda phi, p, T: 2.0,
    )
    with pytest.raises(ValueError, match="no usable egr=0 baseline"):
        lfs_db.fit_piecewise_phi()


def test_fit_piecewise_phi_defaults_f_to_zero_for_pure_egr_zero_sweep():
    # A pure egr=0 sweep is legitimate: f defaults to scalar 0.
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[1.0], pressures=[1e5], temperatures=[300.0],
        egr_values=[0.0],
        su0_fn=lambda phi, p, T: 0.5, f_fn=lambda phi, p, T: 2.0,
    )
    table, quality = lfs_db.fit_piecewise_phi()
    assert table["phi0"]["f"] == 0
    assert table["phi0"]["pTu"]["values"] == [[0.5]]
    assert quality["n_points"] == 0
    assert quality["relative_error_max"] == 0.0


def test_fit_piecewise_phi_raises_for_insufficient_slope_points():
    # egr varies at T=300, so a T=400 corner with only egr=0 is a gap
    # and must raise.
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[1.0], pressures=[1e5], temperatures=[300.0],
        egr_values=[0.0, 0.05, 0.10],
        su0_fn=lambda phi, p, T: 0.5, f_fn=lambda phi, p, T: 2.0,
    )
    lfs_db.results.extend(
        r for r in _piecewise_phi_lfs_db(
            phi_nodes=[1.0], pressures=[1e5], temperatures=[400.0],
            egr_values=[0.0],
            su0_fn=lambda phi, p, T: 0.6, f_fn=lambda phi, p, T: 2.0,
        ).results
    )
    with pytest.raises(ValueError, match="fewer than 2 usable points"):
        lfs_db.fit_piecewise_phi(temperatures=[300.0, 400.0])


def test_fit_piecewise_phi_raises_for_negative_f():
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[1.0], pressures=[1e5], temperatures=[300.0],
        egr_values=[0.0, 0.05, 0.10],
        su0_fn=lambda phi, p, T: 0.5,
        f_fn=lambda phi, p, T: -1.0,  # EGR increasing flame speed
    )
    with pytest.raises(ValueError, match="f < 0"):
        lfs_db.fit_piecewise_phi()


def test_fit_piecewise_phi_explicit_axes_sort_and_restrict():
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[0.6, 0.8, 1.0], pressures=[1e5, 2e5],
        temperatures=[300.0], egr_values=[0.0, 0.05, 0.10],
        su0_fn=lambda phi, p, T: 0.5, f_fn=lambda phi, p, T: 2.0,
    )
    # Out of order and only a subset of the phi nodes.
    table, _ = lfs_db.fit_piecewise_phi(phi_nodes=[1.0, 0.6])
    assert set(table) == {"phi0", "phi1"}
    assert table["phi0"]["Phi"] == 0.6
    assert table["phi1"]["Phi"] == 1.0


def test_fit_piecewise_phi_raises_before_any_sweep(mech):
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    with pytest.raises(ValueError, match="sweep\\(\\) hasn't been run"):
        lfs_db.fit_piecewise_phi()


def test_fit_piecewise_phi_loads_from_directory_when_results_is_none(
    mech, tmp_path,
):
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        equivalence_ratios=[1.0], temperatures=[300.0],
        pressures=[101325.0], egr_values=[0.0, 0.05],
    )
    populated = _piecewise_phi_lfs_db(
        phi_nodes=[1.0], pressures=[101325.0], temperatures=[300.0],
        egr_values=[0.0, 0.05, 0.10],
        su0_fn=lambda phi, p, T: 0.5, f_fn=lambda phi, p, T: 2.0,
    )
    lfs_db._write_cache(
        tmp_path / lfs_db.cache_filename, populated.results,
        *lfs_db._mass_basis_composition(),
    )
    table, _ = lfs_db.fit_piecewise_phi(directory=tmp_path)
    assert np.array(table["phi0"]["f"]["values"]) == (
        pytest.approx(np.array([[2.0]]))
    )


@pytest.mark.openfoam
def test_fit_piecewise_phi_round_trips_through_a_real_dict(tmp_path):
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[0.6, 1.0], pressures=[4e6, 1e7],
        temperatures=[600.0, 900.0], egr_values=[0.0, 0.05, 0.10],
        su0_fn=lambda phi, p, T: 0.5 * phi,
        f_fn=lambda phi, p, T: 2.0,
    )
    table, _ = lfs_db.fit_piecewise_phi()
    result = {"model": "piecewisePhi", "fuel": "naturalGas",
              "naturalGas": table}

    d = Dictionary({"unstrainedLaminarFlameSpeed": result})
    text = d.as_text
    reparsed = Dictionary(text)
    phi0 = (
        reparsed["unstrainedLaminarFlameSpeed"]["naturalGas"]["phi0"]
    )
    assert phi0["Phi"] == Decimal("0.6")

    (tmp_path / "testDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; "
        'object testDict; }\n' + text
    )
    result = subprocess.run(
        ["foamDictionary", "-expand", "testDict"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.slow
def test_solve_point_matches_sweep_and_leaves_results_untouched(lfs_db):
    result = lfs_db.solve_point(T=300, p=101325, egr=0.0, phi=1.0)
    assert 0.1 < result["sL"] < 1.0
    assert result["T"] == 300 and result["phi"] == 1.0
    assert lfs_db.results is None


@pytest.mark.slow
def test_solve_point_accepts_decimal_with_logging(lfs_db):
    # Decimal inputs from an OpenFOAM dict must be converted before
    # _log_inputs() computes "p/1e5" (Decimal/float TypeError).
    result = lfs_db.solve_point(
        T=Decimal("300"), p=Decimal("101325"), egr=Decimal("0.0"),
        phi=Decimal("1.0"), log=True,
    )
    assert 0.1 < result["sL"] < 1.0
    assert result["T"] == 300.0 and result["phi"] == 1.0


_VALIDATE_TRUE_GULDER = {
    "W": 0.35, "eta": 0.2, "xi": 4.5, "alpha": 1.2, "beta": -0.3, "f": 0.8,
}


def _validate_grid_lfs_db(
    true_coeffs, pressures, temperatures, egr_values, phis,
):
    results = []
    for p in pressures:
        for T in temperatures:
            for egr in egr_values:
                for phi in phis:
                    sL = float(evaluate_gulder(true_coeffs, p, T, phi, egr))
                    results.append({
                        "p": p, "T": T, "phi": phi, "egr": egr, "sL": sL,
                        "width": 0.0, "thickness": 0.0, "maxT": 0.0,
                        "species": {},
                    })
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=None, fuel=_FUEL, oxidant=_OXIDANT,
        **_DUMMY_RANGE_KWARGS,
    )
    lfs_db.results = results
    return lfs_db


def _find_result(results, p, T, phi, egr):
    for i, r in enumerate(results):
        if (r["p"], r["T"], r["phi"], r["egr"]) == (p, T, phi, egr):
            return i
    raise AssertionError("point not found in results")


def test_validate_flags_an_isolated_overshoot_point():
    pressures = [1e5, 1.3e5, 1.6e5, 2e5]
    temperatures = [300.0, 340.0, 380.0, 420.0]
    egr_values = [0.0, 0.05, 0.1]
    phis = [0.7, 0.85, 1.0, 1.075, 1.2, 1.35]
    lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    i = _find_result(lfs_db.results, 1.3e5, 340.0, 1.0, 0.0)
    lfs_db.results[i]["sL"] *= 3.0  # Cantera can only overshoot

    status = lfs_db.validate()

    assert status[i] & PlausibilityStatus.MONOTONICITY
    assert status[i] & PlausibilityStatus.GLOBAL_OUTLIER
    assert int(np.sum(status != PlausibilityStatus.GOOD)) == 1


def test_validate_monotonicity_blames_the_lower_temperature_point():
    # A temperature violation blames the lower-T point, unlike pressure
    # and EGR.
    pressures = [1e5, 1.3e5, 1.6e5, 2e5]
    temperatures = [300.0, 340.0, 380.0, 420.0]
    egr_values = [0.0, 0.05, 0.1]
    phis = [0.7, 0.85, 1.0, 1.075, 1.2, 1.35]
    lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    lower = _find_result(lfs_db.results, 1.6e5, 340.0, 1.0, 0.05)
    higher = _find_result(lfs_db.results, 1.6e5, 380.0, 1.0, 0.05)
    lfs_db.results[lower]["sL"] *= 3.0

    status = lfs_db.validate()

    assert status[lower] & PlausibilityStatus.MONOTONICITY
    assert not (status[higher] & PlausibilityStatus.MONOTONICITY)


def test_global_outlier_flags_protects_a_systematic_region():
    # A uniformly scaled phi>=1.2 region is not flagged point-wise. Uses
    # the true coefficients to isolate the neighbour logic from the fit.
    pressures = [1e5, 2e5]
    temperatures = [300.0, 400.0]
    egr_values = [0.0, 0.05]
    phis = [
        0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.075, 1.15, 1.2, 1.35,
    ]
    region_lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    for r in region_lfs_db.results:
        if r["phi"] >= 1.2:
            r["sL"] *= 2.5
    columns, array = region_lfs_db.to_array()
    axis_values = region_lfs_db._axis_values(columns, array)
    grid_row = region_lfs_db._grid_row_index(columns, array, axis_values)
    region_flags = region_lfs_db._global_outlier_flags(
        columns, array, _VALIDATE_TRUE_GULDER, 0.15, grid_row,
    )
    region_idx = [
        i for i, r in enumerate(region_lfs_db.results) if r["phi"] >= 1.2
    ]
    assert not any(region_flags[i] for i in region_idx)

    isolated_lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    isolated_lfs_db.results[10]["sL"] *= 3.0
    columns, array = isolated_lfs_db.to_array()
    axis_values = isolated_lfs_db._axis_values(columns, array)
    grid_row = isolated_lfs_db._grid_row_index(columns, array, axis_values)
    isolated_flags = isolated_lfs_db._global_outlier_flags(
        columns, array, _VALIDATE_TRUE_GULDER, 0.15, grid_row,
    )
    assert isolated_flags[10]
    assert int(isolated_flags.sum()) == 1


def test_validate_flags_non_converged_points():
    pressures = [1e5, 1.3e5, 1.6e5, 2e5]
    temperatures = [300.0, 340.0, 380.0, 420.0]
    egr_values = [0.0, 0.05, 0.1]
    phis = [0.7, 0.85, 1.0, 1.075, 1.2, 1.35]
    lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    i = _find_result(lfs_db.results, 1.3e5, 340.0, 1.0, 0.0)
    lfs_db.results[i]["sL"] = float("nan")

    status = lfs_db.validate()

    assert status[i] & PlausibilityStatus.NON_CONVERGED


def test_egr_slice_flags_needs_a_real_egr_zero_anchor():
    # f needs the egr=0 point (Su0 cancels); without it the slice is
    # skipped.
    pressures = [1e5]
    temperatures = [300.0]
    phis = [1.0]
    egr_values = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    bad = _find_result(lfs_db.results, 1e5, 300.0, 1.0, 0.06)
    lfs_db.results[bad]["sL"] *= 3.0  # breaks f's sign for this slice

    columns, array = lfs_db.to_array()
    axis_values = lfs_db._axis_values(columns, array)
    flags = lfs_db._egr_slice_flags(columns, array, 3, 0.10, axis_values)
    assert flags[bad]
    assert int(flags.sum()) == 1

    zero = _find_result(lfs_db.results, 1e5, 300.0, 1.0, 0.0)
    lfs_db.results[zero]["sL"] = float("nan")  # discard the anchor
    columns, array = lfs_db.to_array()
    axis_values = lfs_db._axis_values(columns, array)
    flags = lfs_db._egr_slice_flags(columns, array, 3, 0.10, axis_values)
    assert not flags.any()


def test_validate_recovers_from_a_few_scattered_bad_points():
    pressures = [1e5 * (1 + 0.3 * k) for k in range(6)]
    temperatures = [300.0 + 30 * k for k in range(6)]
    egr_values = [0.02 * k for k in range(4)]
    phis = [0.6 + 0.1 * k for k in range(6)]
    lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    corrupted = [37, 210, 400, 611]
    for i in corrupted:
        lfs_db.results[i]["sL"] *= 2.5

    status = lfs_db.validate()

    for i in corrupted:
        assert status[i] != PlausibilityStatus.GOOD
    assert int(np.sum(status != PlausibilityStatus.GOOD)) == len(corrupted)


def test_validate_raises_when_max_discard_fraction_exceeded():
    pressures = [1e5, 1.3e5, 1.6e5, 2e5]
    temperatures = [300.0, 340.0, 380.0, 420.0]
    egr_values = [0.0, 0.05, 0.1]
    phis = [0.7, 0.85, 1.0, 1.075, 1.2, 1.35]
    lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    # Half the points are random: no trimming within budget helps.
    for i, r in enumerate(lfs_db.results):
        if i % 2 == 0:
            r["sL"] = 10.0 + 0.001 * (i % 7)

    with pytest.raises(ValueError, match="discarded"):
        lfs_db.validate()


def test_report_plausibility_raises_before_validate():
    lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, [1e5], [300.0], [0.0], [1.0],
    )
    with pytest.raises(ValueError, match="validate"):
        lfs_db.report_plausibility()


def test_report_plausibility_prints_a_summary(capsys):
    pressures = [1e5, 1.3e5, 1.6e5, 2e5]
    temperatures = [300.0, 340.0, 380.0, 420.0]
    egr_values = [0.0, 0.05, 0.1]
    phis = [0.7, 0.85, 1.0, 1.075, 1.2, 1.35]
    lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    i = _find_result(lfs_db.results, 1.3e5, 340.0, 1.0, 0.0)
    lfs_db.results[i]["sL"] *= 3.0
    lfs_db.validate()

    capsys.readouterr()
    lfs_db.report_plausibility()
    out = capsys.readouterr().out
    assert "GOOD" in out
    assert "MONOTONICITY: 1 " in out
    assert "UNCHECKED" not in out


def test_sweep_plausibility_persists_and_round_trips_through_json(
    mech, tmp_path,
):
    original, cache_path, fuel, oxidant = _write_synthetic_cache(
        mech, tmp_path,
    )
    original.validate()
    original._write_cache(cache_path, original.results, fuel, oxidant)

    loaded = LaminarFlameSpeedDatabase.from_json(cache_path)
    assert loaded.plausibility is not None
    assert list(loaded.plausibility) == list(original.plausibility)


def test_sweep_cache_hit_computes_and_persists_missing_plausibility(
    mech, tmp_path,
):
    _, cache_path, _, _ = _write_synthetic_cache(mech, tmp_path)
    data = json.loads(cache_path.read_text())
    assert "plausibility" not in data  # an "older" cache file

    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=[101325.0, 2e5], temperatures=[300.0],
        equivalence_ratios=[0.8, 1.0], egr_values=[0.0],
    )
    lfs_db.sweep(directory=tmp_path, plausibility_check=True)

    assert lfs_db.plausibility is not None
    data = json.loads(cache_path.read_text())
    assert "plausibility" in data


def test_sweep_plausibility_check_failure_warns_and_marks_unchecked(
    mech, tmp_path,
):
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=[101325.0, 2e5], temperatures=[300.0],
        equivalence_ratios=[0.8, 1.0], egr_values=[0.0],
    )
    # A 0 % discard budget on data that needs trimming.
    lfs_db.results = [
        {
            "p": p, "T": 300.0, "phi": phi, "egr": 0.0,
            "sL": (10.0 if i == 0 else 0.3), "width": 0.0,
            "thickness": 0.0, "maxT": 0.0, "species": {},
        }
        for i, (p, phi) in enumerate(
            (p, phi)
            for p in [101325.0, 2e5, 3e5, 4e5]
            for phi in [0.8, 1.0]
        )
    ]

    def _always_fails(*args, **kwargs):
        raise ValueError("validate(): forced failure for this test")

    lfs_db.validate = _always_fails
    with pytest.warns(UserWarning, match="plausibility check failed"):
        lfs_db._run_plausibility_check(log=False)
    assert list(lfs_db.plausibility) == [
        int(PlausibilityStatus.UNCHECKED)
    ] * len(lfs_db.results)


def _flag_index(lfs_db, index, status):
    lfs_db.plausibility = np.array(
        [
            int(status) if i == index else int(PlausibilityStatus.GOOD)
            for i in range(len(lfs_db.results))
        ],
        dtype=np.int64,
    )


def test_fit_gulder_require_plausible_excludes_still_bad_points():
    pressures = [1e5, 1.3e5, 1.6e5, 2e5]
    temperatures = [300.0, 340.0, 380.0, 420.0]
    egr_values = [0.0, 0.05, 0.1]
    phis = [0.7, 0.85, 1.0, 1.075, 1.2, 1.35]
    lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    n_total = len(lfs_db.results)
    i = _find_result(lfs_db.results, 1.3e5, 340.0, 1.0, 0.0)
    lfs_db.results[i]["sL"] *= 3.0  # MONOTONICITY | GLOBAL_OUTLIER

    coefficients, quality = lfs_db.fit_gulder(
        require_plausible=True, resolve=False,
    )

    assert quality["n_points"] == n_total - 1
    for name in ("eta", "xi", "alpha", "beta"):
        assert coefficients[name] == pytest.approx(
            _VALIDATE_TRUE_GULDER[name], rel=0.05,
        )


def test_fit_gulder_require_plausible_keeps_non_converged_floored():
    # NON_CONVERGED stays floored and included under require_plausible
    # (flammability-limit data); checked against floored vs removed.
    pressures = [1e5, 1.3e5, 1.6e5, 2e5]
    temperatures = [300.0, 340.0, 380.0, 420.0]
    egr_values = [0.0, 0.05, 0.1]
    phis = [0.7, 0.85, 1.0, 1.075, 1.2, 1.35]

    with_nan = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    with_nan.results[
        _find_result(with_nan.results, 1.3e5, 340.0, 1.0, 0.0)
    ]["sL"] = float("nan")

    removed = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    del removed.results[
        _find_result(removed.results, 1.3e5, 340.0, 1.0, 0.0)
    ]

    coef_floored, _ = with_nan.fit_gulder()
    coef_removed, _ = removed.fit_gulder()
    assert any(
        coef_floored[name] != pytest.approx(coef_removed[name])
        for name in coef_floored
    )

    coef_plausible, _ = with_nan.fit_gulder(
        require_plausible=True, resolve=False,
    )
    for name in coef_floored:
        assert coef_plausible[name] == pytest.approx(coef_floored[name])


def test_fit_gulder_require_plausible_raises_when_everything_flagged():
    pressures = [1e5, 1.3e5, 1.6e5, 2e5]
    temperatures = [300.0, 340.0, 380.0, 420.0]
    egr_values = [0.0, 0.05, 0.1]
    phis = [0.7, 0.85, 1.0, 1.075, 1.2, 1.35]
    lfs_db = _validate_grid_lfs_db(
        _VALIDATE_TRUE_GULDER, pressures, temperatures, egr_values, phis,
    )
    i = _find_result(lfs_db.results, 1.3e5, 340.0, 1.0, 0.0)
    lfs_db.results[i]["sL"] *= 3.0

    with pytest.raises(ValueError, match="could not be resolved"):
        lfs_db.fit_gulder(
            require_plausible=True, resolve=False,
            p=(1.29e5, 1.31e5), T=(339.0, 341.0),
            phi=(0.99, 1.01), egr=(-0.01, 0.01),
        )


def test_resolve_flagged_marks_resolved_and_keeps_flag_history(
    mech, monkeypatch,
):
    pressures = [1e5, 1.3e5, 1.6e5, 2e5]
    temperatures = [300.0, 340.0, 380.0, 420.0]
    egr_values = [0.0, 0.05, 0.1]
    phis = [0.7, 0.85, 1.0, 1.075, 1.2, 1.35]
    lfs_db = LaminarFlameSpeedDatabase(
        mechanism=mech, fuel=_FUEL, oxidant=_OXIDANT,
        pressures=pressures, temperatures=temperatures,
        equivalence_ratios=phis, egr_values=egr_values, max_processes=2,
    )
    lfs_db.results = [
        {
            "p": p, "T": T, "phi": phi, "egr": egr,
            "sL": float(evaluate_gulder(_VALIDATE_TRUE_GULDER, p, T, phi, egr)),
            "width": 0.01, "thickness": 0.001, "maxT": 2000.0,
            "species": {},
        }
        for p in pressures for T in temperatures
        for egr in egr_values for phi in phis
    ]
    i = _find_result(lfs_db.results, 1.3e5, 340.0, 1.0, 0.0)
    lfs_db.results[i]["sL"] *= 3.0
    lfs_db.validate()
    assert lfs_db.plausibility[i] & int(PlausibilityStatus.MONOTONICITY)

    true_sL = float(
        evaluate_gulder(_VALIDATE_TRUE_GULDER, 1.3e5, 340.0, 1.0, 0.0)
    )

    def fake_solve_flame_point(
        mechanism_path, fuel, oxidant, products, p, T, phi, egr,
        basis, initial_width, num_points, transport_model,
        loglevel=0, log=False,
    ):
        return {
            "p": p, "T": T, "phi": phi, "egr": egr, "sL": true_sL,
            "width": 0.01, "thickness": 0.001, "maxT": 2000.0,
            "species": {},
        }

    monkeypatch.setattr(
        "pyfoamy.cantera.laminar_flame_speed._solve_flame_point",
        fake_solve_flame_point,
    )

    fixed = lfs_db._resolve_flagged([i], "test")

    assert fixed == {i}
    assert lfs_db.results[i]["sL"] == pytest.approx(true_sL)
    assert lfs_db.plausibility[i] & int(PlausibilityStatus.MONOTONICITY)
    assert lfs_db.plausibility[i] & int(PlausibilityStatus.RESOLVED)


def test_bracket_reconstruct_averages_both_axes_when_both_bracket():
    grid = [
        [1.0, 2.0, float("nan")],
        [3.0, float("nan"), 5.0],
        [5.0, 6.0, 7.0],
    ]
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 2.0]
    value = LaminarFlameSpeedDatabase._bracket_reconstruct(
        grid, xs, ys, 1, 1,
    )
    assert value == pytest.approx(4.0)


def test_bracket_reconstruct_uses_whichever_single_axis_brackets():
    grid = [[1.0], [float("nan")], [5.0]]
    xs = [0.0, 1.0, 2.0]
    ys = [0.0]
    value = LaminarFlameSpeedDatabase._bracket_reconstruct(
        grid, xs, ys, 1, 0,
    )
    assert value == pytest.approx(3.0)


def test_bracket_reconstruct_returns_none_at_a_true_edge():
    grid = [[float("nan")], [5.0]]
    xs = [0.0, 1.0]
    ys = [0.0]
    assert LaminarFlameSpeedDatabase._bracket_reconstruct(
        grid, xs, ys, 0, 0,
    ) is None


def test_fit_piecewise_phi_reconstructs_a_flagged_middle_corner():
    def su0_fn(phi, p, T):
        return 0.3 + 1e-6 * p

    def f_fn(phi, p, T):
        return 1.5

    pressures = [1e5, 2e5, 3e5]
    temperatures = [300.0]
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[1.0], pressures=pressures, temperatures=temperatures,
        egr_values=[0.0, 0.05, 0.10], su0_fn=su0_fn, f_fn=f_fn,
    )
    bad = _find_result(lfs_db.results, 2e5, 300.0, 1.0, 0.0)
    _flag_index(lfs_db, bad, PlausibilityStatus.MONOTONICITY)

    table, quality = lfs_db.fit_piecewise_phi(
        require_plausible=True, resolve=False,
    )

    expected = su0_fn(1.0, 2e5, 300.0)
    assert table["phi0"]["pTu"]["values"][1][0] == pytest.approx(expected)
    assert table["phi0"]["f"]["values"][1][0] == pytest.approx(1.5)
    assert (1.0, 2e5, 300.0, "pTu") in quality["reconstructed_corners"]
    assert (1.0, 2e5, 300.0, "f") in quality["reconstructed_corners"]


def test_fit_piecewise_phi_raises_when_a_flagged_edge_corner_cant_reconstruct():
    def su0_fn(phi, p, T):
        return 0.3 + 1e-6 * p

    pressures = [1e5, 2e5, 3e5]
    temperatures = [300.0]
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[1.0], pressures=pressures, temperatures=temperatures,
        egr_values=[0.0, 0.05, 0.10], su0_fn=su0_fn,
        f_fn=lambda phi, p, T: 1.5,
    )
    bad = _find_result(lfs_db.results, 1e5, 300.0, 1.0, 0.0)
    _flag_index(lfs_db, bad, PlausibilityStatus.MONOTONICITY)

    with pytest.raises(ValueError, match="no usable egr=0 baseline"):
        lfs_db.fit_piecewise_phi(require_plausible=True, resolve=False)


def test_fit_piecewise_phi_require_plausible_false_ignores_plausibility():
    # Without require_plausible, self.plausibility changes nothing.
    def su0_fn(phi, p, T):
        return 0.3 + 1e-6 * p

    pressures = [1e5, 2e5, 3e5]
    temperatures = [300.0]
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[1.0], pressures=pressures, temperatures=temperatures,
        egr_values=[0.0, 0.05, 0.10], su0_fn=su0_fn,
        f_fn=lambda phi, p, T: 1.5,
    )
    bad = _find_result(lfs_db.results, 2e5, 300.0, 1.0, 0.0)
    _flag_index(lfs_db, bad, PlausibilityStatus.MONOTONICITY)

    table, _ = lfs_db.fit_piecewise_phi()
    assert table["phi0"]["pTu"]["values"][1][0] == pytest.approx(
        su0_fn(1.0, 2e5, 300.0)
    )


def test_fit_piecewise_phi_pure_egr_zero_sweep_ignores_f_when_reconstructing():
    # A missing baseline must not matter for a pure egr=0 sweep.
    def su0_fn(phi, p, T):
        return 0.3 + 1e-6 * p

    pressures = [1e5, 2e5, 3e5]
    temperatures = [300.0]
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[1.0], pressures=pressures, temperatures=temperatures,
        egr_values=[0.0], su0_fn=su0_fn, f_fn=lambda phi, p, T: 0.0,
    )
    bad = _find_result(lfs_db.results, 2e5, 300.0, 1.0, 0.0)
    _flag_index(lfs_db, bad, PlausibilityStatus.MONOTONICITY)

    table, quality = lfs_db.fit_piecewise_phi(
        require_plausible=True, resolve=False,
    )

    assert table["phi0"]["f"] == 0
    assert quality["reconstructed_corners"] == [(1.0, 2e5, 300.0, "pTu")]


def test_fit_piecewise_phi_resolves_a_flagged_point_before_reconstructing(
    mech, monkeypatch,
):
    # Decreasing with p: validate()'s reference fit rejects beta > 0.
    def su0_fn(phi, p, T):
        return 0.6 - 1e-6 * p

    pressures = [1e5, 2e5, 3e5]
    temperatures = [300.0]
    lfs_db = _piecewise_phi_lfs_db(
        phi_nodes=[1.0], pressures=pressures, temperatures=temperatures,
        egr_values=[0.0], su0_fn=su0_fn, f_fn=lambda phi, p, T: 0.0,
    )
    lfs_db.mechanism = mech
    bad = _find_result(lfs_db.results, 2e5, 300.0, 1.0, 0.0)
    _flag_index(lfs_db, bad, PlausibilityStatus.MONOTONICITY)
    true_value = su0_fn(1.0, 2e5, 300.0)

    def fake_solve_flame_point(
        mechanism_path, fuel, oxidant, products, p, T, phi, egr,
        basis, initial_width, num_points, transport_model,
        loglevel=0, log=False,
    ):
        return {
            "p": p, "T": T, "phi": phi, "egr": egr, "sL": true_value,
            "width": 0.01, "thickness": 0.001, "maxT": 2000.0,
            "species": {},
        }

    monkeypatch.setattr(
        "pyfoamy.cantera.laminar_flame_speed._solve_flame_point",
        fake_solve_flame_point,
    )

    table, quality = lfs_db.fit_piecewise_phi(require_plausible=True)

    assert quality["reconstructed_corners"] == []
    assert lfs_db.plausibility[bad] & int(PlausibilityStatus.RESOLVED)
    assert table["phi0"]["pTu"]["values"][1][0] == pytest.approx(true_value)
