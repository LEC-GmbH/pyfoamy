"""One `FreeFlame` solve per point, as a process-pool worker."""

import os
import signal
import time
import warnings
from contextlib import nullcontext
from typing import Dict

from pyfoamy.cantera.mechanism import ct, Mechanism, np
from pyfoamy.cantera._lfs_common import (
    _MAX_FAILED_SOLVES,
    _MAX_INTERNAL_WIDTH_GROWTH,
    _MIN_PRESSURE,
    _REFINE_CRITERIA,
    _REGEN_WIDTH_SHRINK,
    _TIME_STEPS,
    _TRACE_THRESHOLD,
    LFSPoint,
)


# Value/Lock reach Pool workers only via initializer=, not as task
# arguments; _worker_log_lock also prevents interleaved stdout.
_worker_counter = None
_worker_total = None
_worker_log_lock = None


def _init_worker(counter, total: int, log_lock) -> None:
    # Only the main process handles Ctrl+C (see _GracefulStop).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _worker_counter, _worker_total, _worker_log_lock
    _worker_counter = counter
    _worker_total = total
    _worker_log_lock = log_lock


def _print_locked(lock, text: str) -> None:
    # flush=True inside the lock, or the write can leave the critical
    # section.
    with lock if lock is not None else nullcontext():
        print(text, flush=True)


# Dispatch-loop poll interval - negligible next to real solve times.
_POLL_INTERVAL = 0.05


# Double Ctrl+C: the first SIGINT sets `.requested` and prints a
# notice; the second restores the previous handler and raises
# KeyboardInterrupt. Off the main thread (`signal.signal` raises
# ValueError) it falls back to default interrupt behaviour.
class _GracefulStop:

    def __init__(self, message: str, log_lock=None) -> None:
        self.requested = False
        self._message = message
        self._log_lock = log_lock
        self._previous_handler = None
        self._installed = False

    def __enter__(self) -> "_GracefulStop":
        try:
            self._previous_handler = signal.signal(
                signal.SIGINT, self._handle,
            )
            self._installed = True
        except ValueError:
            pass
        return self

    def _handle(self, signum, frame) -> None:
        if self.requested:
            signal.signal(signal.SIGINT, self._previous_handler)
            raise KeyboardInterrupt()
        self.requested = True
        _print_locked(self._log_lock, self._message)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._installed:
            signal.signal(signal.SIGINT, self._previous_handler)
        return False


_LOG_HEADER = (
    f"{'p_bar':>8} {'T_K':>5} {'phi':>5} {'egr%':>5}   "
    f"{'sL_cms':>9} {'delta_um':>8} {'maxT_K':>7} {'dw_mm':>6} "
    f"{'t_s':>7}"
)


def _log_inputs(p: float, T: float, phi: float, egr: float) -> str:
    return f"{p/1e5:5.1f}bar {T:4.0f}K {phi:5.3f} {100*egr:4.1f}%"


def _check_pressure(p: float) -> None:
    if p < _MIN_PRESSURE:
        raise ValueError(
            f"pressure {p:g} Pa is below the sanity floor of "
            f"{_MIN_PRESSURE:g} Pa - check for a missing unit "
            "conversion (e.g. 20 instead of 20e5)"
        )


def _check_temperature(T: float, min_temp: float) -> None:
    if T < min_temp:
        raise ValueError(
            f"temperature {T:g} K is below the mechanism's own "
            f"Tmin={min_temp:g} K"
        )


def _check_phi(phi: float) -> None:
    if phi < 0:
        raise ValueError(
            f"equivalence ratio phi={phi:g} must not be negative"
        )


def _check_egr(egr: float) -> None:
    if not (0 <= egr < 1):
        raise ValueError(
            f"egr={egr:g} must satisfy 0 <= egr < 1 (egr is a mass "
            "fraction, not a percentage)"
        )


def _validate_point(
    gas: "ct.Solution", p: float, T: float, phi: float, egr: float,
) -> None:
    _check_pressure(p)
    _check_temperature(T, gas.min_temp)
    _check_phi(phi)
    _check_egr(egr)


def _log_result(
    sL: float, thickness: float, maxT: float, width: float, elapsed: float,
) -> str:
    return (
        f"{100*sL:5.1f}cm/s {1e6*thickness:6.1f}um {maxT:6.1f}K "
        f"{1000*width:4.1f}mm {elapsed:6.1f}s"
    )


def _log_progress() -> str:
    if _worker_counter is None:
        return ""
    with _worker_counter.get_lock():
        _worker_counter.value += 1
        n = _worker_counter.value
    digits = len(str(_worker_total))
    return f"  [{n:>{digits}d}/{_worker_total}]"


def _available_cpus() -> int:
    # sched_getaffinity respects a SLURM/taskset mask but is Linux-only.
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _solve_flame_point(
    mechanism_path: str,
    fuel: Dict[str, float],
    oxidant: Dict[str, float],
    products: Dict[str, float],
    p: float,
    T: float,
    phi: float,
    egr: float,
    basis: str,
    initial_width: float,
    num_points: int,
    transport_model: str,
    loglevel: int = 0,
    log: bool = False,
) -> LFSPoint:
    # mechanism_path, never mechanism_yaml (see Mechanism.frozen_path).
    # Every retry (exception or regen) rebuilds gas/flame.
    mechanism = Mechanism(mechanism_path)
    _validate_point(mechanism.gas, p, T, phi, egr)

    if loglevel > 0:
        _print_locked(
            _worker_log_lock,
            f"laminar_flame_speed: starting "
            f"{_log_inputs(p, T, phi, egr)}   "
            f"fuel={fuel} oxidant={oxidant}   "
            f"width={1000*initial_width:.3f}mm num_points={num_points}",
        )

    start = time.time()
    width = initial_width
    failed_solves = 0
    gas = mechanism.gas
    while True:
        gas.set_equivalence_ratio(
            phi, fuel, oxidant, basis=basis,
            diluent=products, fraction={"diluent": egr},
        )
        gas.TP = T, p
        species = gas.mass_fraction_dict(threshold=_TRACE_THRESHOLD)

        grid = np.linspace(0, width, num_points)
        flame = ct.FreeFlame(gas, grid=grid)
        flame.transport_model = transport_model
        flame.set_refine_criteria(**_REFINE_CRITERIA)
        flame.set_time_step(*_TIME_STEPS)
        try:
            flame.solve(loglevel=loglevel, auto=True)
        except ct.CanteraError:
            failed_solves += 1
            if failed_solves > _MAX_FAILED_SOLVES:
                warnings.warn(
                    f"laminar_flame_speed: {_log_inputs(p, T, phi, egr)} "
                    f"failed {failed_solves} times up to "
                    f"dw={width*1000:.1f}mm, sL=nan"
                )
                sL = thickness = max_T = float("nan")
                result_width = width
                break
            if log:
                _print_locked(
                    _worker_log_lock,
                    f"laminar_flame_speed: {_log_inputs(p, T, phi, egr)}"
                    f"   FAILED dw={width*1000:.1f}mm - retrying at "
                    f"{2*width*1000:.1f}mm",
                )
            width *= 2
            gas = ct.Solution(mechanism_path)
            continue

        result_width = float(flame.grid[-1])
        max_T = float(np.max(flame.T))
        min_T = float(np.min(flame.T))
        max_grad_T = float(np.max(np.gradient(flame.T, flame.grid)))
        thickness = (max_T - min_T) / max_grad_T

        if result_width > width:
            if result_width >= width * _MAX_INTERNAL_WIDTH_GROWTH:
                # Hit cantera's doubling cap: never converged (see the
                # module-level comment).
                warnings.warn(
                    f"laminar_flame_speed: {_log_inputs(p, T, phi, egr)} "
                    f"never converged (dw={width*1000:.1f}mm -> "
                    f"{result_width*1000:.1f}mm, sL=nan) - try a "
                    "smaller initial_width or more num_points"
                )
                sL = float("nan")
                break

            # regen - see the module-level comment above.
            new_width = _REGEN_WIDTH_SHRINK * result_width
            if log:
                _print_locked(
                    _worker_log_lock,
                    f"laminar_flame_speed: {_log_inputs(p, T, phi, egr)}"
                    f"   REGEN dw={width*1000:.1f}mm grew to "
                    f"{result_width*1000:.1f}mm - retrying at "
                    f"{new_width*1000:.1f}mm",
                )
            width = new_width
            gas = ct.Solution(mechanism_path)
            continue

        sL = float(flame.velocity[0])
        break

    result = {
        "p": p, "T": T, "phi": phi, "egr": egr,
        "sL": sL,
        "width": result_width,
        "thickness": thickness,
        "maxT": max_T,
        "species": species,
    }

    if log:
        elapsed = time.time() - start
        _print_locked(
            _worker_log_lock,
            f"laminar_flame_speed: {_log_inputs(p, T, phi, egr)}   "
            + _log_result(
                result["sL"], result["thickness"], result["maxT"],
                result["width"], elapsed,
            )
            + _log_progress(),
        )

    return result
