"""Laminar flame speed sweeps with Cantera's `FreeFlame` in a process pool."""

import hashlib
import itertools
import json
import time
import warnings
from dataclasses import dataclass
from multiprocessing import Lock, Pool, Value
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pyfoamy.cantera.mechanism import (
    Mechanism,
    np,
    require_cantera,
)
from pyfoamy.cantera._lfs_common import (
    _AXES,
    _FIXED_COLUMNS,
    _RESOLVE_NUM_POINTS,
    _RESOLVE_WIDTH_SHRINK,
    LFSPoint,
    PlausibilityStatus,
)
from pyfoamy.cantera._lfs_solver import (
    _available_cpus,
    _check_egr,
    _check_phi,
    _check_pressure,
    _check_temperature,
    _GracefulStop,
    _init_worker,
    _LOG_HEADER,
    _POLL_INTERVAL,
    _solve_flame_point,
)
from pyfoamy.cantera._lfs_fits import (
    evaluate_gulder,  # noqa: F401
    FitsMixin,
)
from pyfoamy.cantera._lfs_validation import (
    ValidationMixin,
)


def _solve_flame_point_star(args):
    # imap_unordered takes one argument (unlike starmap) but streams
    # results as they finish.
    return _solve_flame_point(*args)


@dataclass
class LaminarFlameSpeedDatabase(FitsMixin, ValidationMixin):
    """Sweeps laminar flame speed (plus flame thickness and adiabatic
    temperature) over pressure, temperature, equivalence ratio and EGR.

    One database is one fixed sweep configuration. `from_json()` reloads a
    `sweep(directory=...)` result without Cantera.

    Args:
        fuel: Fuel `{specie: fraction}`.
        oxidant: Oxidant `{specie: fraction}`.
        pressures: Pressures [Pa].
        temperatures: Unburnt mixture temperatures [K].
        equivalence_ratios: `phi` values, or air-fuel equivalence ratios
            `lambda = 1/phi` if `is_lambda`.
        egr_values: EGR mass fractions.
        basis: `"mass"` or `"mole"` for `fuel`/`oxidant`.
        max_processes: Worker process cap.
        mechanism: A mechanism path/name or `Mechanism`; only needed for
            `sweep()`/`solve_point()`.

    Attributes:
        results: Return value of the last `sweep()`.
        plausibility: `PlausibilityStatus` array of the last `validate()`,
            in `results` order; `None` until then or if the cache has
            none.
    """
    fuel: Dict[str, float]
    oxidant: Dict[str, float]
    pressures: Sequence[float]
    temperatures: Sequence[float]
    equivalence_ratios: Sequence[float]
    egr_values: Sequence[float]
    mechanism: Optional[Union[str, Mechanism]] = None
    is_lambda: bool = False
    basis: str = "mass"
    transport_model: str = "mixture-averaged"
    max_processes: Optional[int] = None
    results: Optional[List[LFSPoint]] = None
    plausibility: Optional["np.ndarray"] = None

    def __post_init__(self) -> None:
        self.pressures = self._to_float_list(self.pressures)
        self.temperatures = self._to_float_list(self.temperatures)
        self.equivalence_ratios = self._to_float_list(
            self.equivalence_ratios
        )
        self.egr_values = self._to_float_list(self.egr_values)
        if self.is_lambda:
            self.equivalence_ratios = [
                1.0 / lambda_ for lambda_ in self.equivalence_ratios
            ]
        if self.mechanism is not None:
            self.mechanism = Mechanism.coerce(self.mechanism)

        for p in self.pressures:
            _check_pressure(p)
        for phi in self.equivalence_ratios:
            _check_phi(phi)
        for egr in self.egr_values:
            _check_egr(egr)
        # Tmin needs a live mechanism; skipped for fit-only databases.
        if self.mechanism is not None:
            min_temp = self.mechanism.gas.min_temp
            for T in self.temperatures:
                _check_temperature(T, min_temp)

    @staticmethod
    def _is_constant(x) -> bool:
        return bool(np.ptp(x) == 0)

    @staticmethod
    def _grid_search_1d(
        residual: Callable[[float], float], lo: float, hi: float,
        n_candidates: int = 25, n_rounds: int = 3,
    ) -> float:
        best = 0.5 * (lo + hi)
        for _ in range(n_rounds):
            candidates = np.linspace(lo, hi, n_candidates)
            residuals = [residual(c) for c in candidates]
            best = candidates[int(np.argmin(residuals))]
            step = (hi - lo) / n_candidates
            lo, hi = best - 2 * step, best + 2 * step
        return float(best)

    @staticmethod
    def _to_float_list(values: Sequence) -> List[float]:
        return [float(value) for value in values]

    @staticmethod
    def _round_sig(value: float, figures: int = 5) -> float:
        return float(f"{value:.{figures}g}")

    @staticmethod
    def _round_sorted(values: Sequence[float]) -> List[float]:
        return sorted(
            LaminarFlameSpeedDatabase._round_sig(value) for value in values
        )

    @staticmethod
    def _round_mapping(mapping: Dict[str, float]) -> Dict[str, float]:
        return {
            name: LaminarFlameSpeedDatabase._round_sig(value)
            for name, value in mapping.items()
        }

    @staticmethod
    def _stable_mechanism_yaml(mechanism_yaml: str) -> str:
        # write_yaml() stamps the time into a `date:` line; strip it so
        # the cache key is stable.
        return "\n".join(
            line for line in mechanism_yaml.splitlines()
            if not line.startswith("date: ")
        )

    @classmethod
    def from_json(
        cls, path,
        mechanism: Optional[Union[str, Mechanism, bool]] = None,
        transport_model: Optional[str] = None,
        max_processes: Optional[int] = None,
    ) -> "LaminarFlameSpeedDatabase":
        """Loads a `sweep(directory=...)` cache file with `results` populated.

        Cantera is needed only if `mechanism` is given.

        Args:
            path: Cache file path.
            mechanism: A path/name, a `Mechanism`, `True` to rebuild one from
                the file's embedded metadata, or `None`.
            transport_model: Overrides the file's recorded value.
        """
        with Path(path).open() as handle:
            data = json.load(handle)
        metadata = data["metadata"]
        if mechanism is True:
            mechanism = Mechanism.from_yaml(metadata["mechanism_yaml"])
        db = cls(
            fuel=metadata["fuel"],
            oxidant=metadata["oxidant"],
            basis=metadata["basis"],
            pressures=metadata["pressures"],
            temperatures=metadata["temperatures"],
            equivalence_ratios=metadata["equivalence_ratios"],
            egr_values=metadata["egr_values"],
            mechanism=mechanism,
            transport_model=(
                transport_model if transport_model is not None
                else metadata["transport_model"]
            ),
            max_processes=max_processes,
        )
        db.results = data["results"]
        if data.get("plausibility") is not None:
            db.plausibility = np.array(data["plausibility"], dtype=np.int64)
        return db

    @property
    def cache_filename(self) -> str:
        """The cache filename `sweep(directory=...)` would use, e.g.
        `"lfs_1a2b3c4d5e6f7a8b.json"`.

        Raises:
            ValueError: If `self.mechanism` is `None`.
        """
        self._check_mechanism("cache_filename")
        fuel, oxidant = self._mass_basis_composition()
        return f"lfs_{self._cache_key(fuel, oxidant)}.json"

    def sweep(
        self,
        initial_width: float = 0.001,
        num_points: int = 40,
        directory: Optional[str] = None,
        redo: bool = False,
        name: Optional[str] = None,
        loglevel: int = 0,
        log: bool = False,
        plausibility_check: bool = False,
    ) -> List[LFSPoint]:
        """Solves one `FreeFlame` per point of the full tensor grid
        `pressures x temperatures x equivalence_ratios x egr_values`, in
        parallel.

        The EGR diluent is always the products of complete combustion at
        stoichiometric conditions, whatever the point's `phi`, matching
        OpenFOAM's `uInhomogeneousEGR` mixing law.

        Args:
            initial_width: Starting `FreeFlame` domain width [m]; keep small,
                refinement grows it.
            num_points: Initial grid point count.
            directory: If given, load the cache file from there instead of
                recomputing; `None` never touches disk.
            redo: Ignore an existing cache and recompute.
            name: Cache file name without extension (default: hash of the
                configuration).
            loglevel: Passed to `flame.solve(loglevel=...)`.
            log: Print a status line per point plus progress/retry/cache
                notices.
            plausibility_check: Run `validate()` once the full grid is
                available (not after a cache load or early stop that already
                has one) and cache it. A `validate()` failure warns and marks
                every point `UNCHECKED`; results are never lost.

        Returns:
            One dict per point: `T`, `p`, `egr`, `phi`, `sL` [m/s], `width`
            [m], `thickness` [m], `maxT` [K], `species`; also `self.results`.

        Raises:
            ValueError: If `self.mechanism` is `None`.
        """
        require_cantera()
        self._check_mechanism("sweep")
        fuel, oxidant = self._mass_basis_composition()

        cache_path = (
            self._cache_path(directory, name, fuel, oxidant)
            if directory is not None else None
        )
        # Leading dot: hidden, so the in-progress file does not clutter
        # `ls`/globs of the cache directory.
        partial_path = (
            cache_path.with_name(f".{cache_path.name}.partial")
            if cache_path is not None else None
        )
        if cache_path is not None and not redo and cache_path.is_file():
            self.results, self.plausibility = self._read_cache(cache_path)
            if log:
                print(
                    f"laminar_flame_speed: loaded "
                    f"{len(self.results)} cached points from "
                    f"{cache_path}"
                )
            if plausibility_check and self.plausibility is None:
                self._run_plausibility_check(log)
                self._write_cache(cache_path, self.results, fuel, oxidant)
            return self.results

        self.results = []
        if partial_path is not None and not redo and partial_path.is_file():
            self.results = self._read_partial(partial_path, log)

        products = self.mechanism.complete_combustion_products(
            fuel, oxidant, basis="mass", equivalence_ratio=1.0,
        )

        with self.mechanism.frozen_path() as mech_path:
            points = list(itertools.product(
                self.pressures, self.temperatures,
                self.equivalence_ratios, self.egr_values,
            ))
            # Points recovered from partial_path are already in
            # self.results.
            done = {
                (r["p"], r["T"], r["phi"], r["egr"]) for r in self.results
            }
            points = [point for point in points if point not in done]
            tasks = [
                (
                    mech_path, fuel, oxidant, products, p, T, phi, egr,
                    "mass", initial_width, num_points,
                    self.transport_model, loglevel, log,
                )
                for p, T, phi, egr in points
            ]
            if log and done:
                print(
                    f"laminar_flame_speed: resuming - {len(done)} "
                    f"point(s) already in {partial_path}, "
                    f"{len(tasks)} remaining"
                )
            total_expected = len(done) + len(tasks)
            if tasks:
                available = _available_cpus()
                processes = min(
                    self.max_processes or available, available,
                    len(tasks),
                )
                if log:
                    print(
                        f"laminar_flame_speed: solving {len(tasks)} "
                        f"points on {processes} workers"
                    )
                    print(f"laminar_flame_speed: {_LOG_HEADER}")
                # counter/log_lock reach workers via Pool's initializer
                # (see _init_worker); log_lock serializes log=True
                # prints.
                counter = Value("i", 0) if log else None
                log_lock = Lock() if log else None
                stop = _GracefulStop(
                    f"laminar_flame_speed: stopping after the "
                    f"{processes} point(s) currently running finish "
                    "(progress so far is saved - call sweep() again "
                    "to resume) - press Ctrl+C again to abort "
                    "immediately",
                    log_lock=log_lock,
                )
                with stop, Pool(
                    processes=processes, initializer=_init_worker,
                    initargs=(counter, len(tasks), log_lock),
                ) as pool:
                    # Bounded sliding window, not imap_unordered (which
                    # pulls the whole iterable up front); a stop request
                    # just isn't refilled.
                    pending_tasks = iter(tasks)
                    active = [
                        pool.apply_async(_solve_flame_point_star, (task,))
                        for task in itertools.islice(pending_tasks, processes)
                    ]
                    while active:
                        still_active = []
                        refilled = []
                        for async_result in active:
                            if not async_result.ready():
                                still_active.append(async_result)
                                continue
                            result = async_result.get()
                            self.results.append(result)
                            if partial_path is not None:
                                self._append_partial(partial_path, result)
                            if not stop.requested:
                                next_task = next(pending_tasks, None)
                                if next_task is not None:
                                    refilled.append(pool.apply_async(
                                        _solve_flame_point_star,
                                        (next_task,),
                                    ))
                        active = still_active + refilled
                        if active:
                            time.sleep(_POLL_INTERVAL)

        # Deterministic, tidy order, not completion order.
        self.results.sort(
            key=lambda r: (r["p"], r["T"], r["phi"], r["egr"])
        )
        completed = len(self.results) == total_expected

        if plausibility_check and completed:
            self._run_plausibility_check(log)

        if cache_path is not None:
            if completed:
                self._write_cache(cache_path, self.results, fuel, oxidant)
                if partial_path.is_file():
                    partial_path.unlink()
                if log:
                    print(f"laminar_flame_speed: wrote cache {cache_path}")
            elif log:
                print(
                    f"laminar_flame_speed: stopped early "
                    f"({len(self.results)}/{total_expected} points) - "
                    f"progress saved in {partial_path}, call sweep() "
                    "again to resume"
                )
        return self.results

    def _run_plausibility_check(self, log: bool) -> None:
        # Warn and mark UNCHECKED rather than lose a sweep's results.
        try:
            self.validate()
        except ValueError as exc:
            warnings.warn(f"laminar_flame_speed: plausibility check failed - {exc}")
            self.plausibility = np.full(
                len(self.results), int(PlausibilityStatus.UNCHECKED),
                dtype=np.int64,
            )
        if log:
            self.report_plausibility()

    def solve_point(
        self, p: float, T: float, phi: float, egr: float,
        initial_width: float = 0.001, num_points: int = 40,
        loglevel: int = 0, log: bool = False,
    ) -> LFSPoint:
        """Solves one `FreeFlame` point synchronously; `self.results` is
        untouched and the point may lie outside the declared ranges.

        Args:
            p: Pressure [Pa].
            T: Unburnt mixture temperature [K].
            phi: Equivalence ratio.
            egr: EGR mass fraction.

        Returns:
            The point, as in `sweep()`.

        Raises:
            ValueError: If `self.mechanism` is `None`.
        """
        require_cantera()
        self._check_mechanism("solve_point")
        p, T, phi, egr = float(p), float(T), float(phi), float(egr)
        fuel, oxidant = self._mass_basis_composition()
        products = self.mechanism.complete_combustion_products(
            fuel, oxidant, basis="mass", equivalence_ratio=1.0,
        )
        if log:
            print(f"laminar_flame_speed: {_LOG_HEADER}")
        with self.mechanism.frozen_path() as mech_path:
            return _solve_flame_point(
                mech_path, fuel, oxidant, products, p, T, phi, egr,
                "mass", initial_width, num_points, self.transport_model,
                loglevel, log,
            )

    def _check_mechanism(self, caller: str) -> None:
        if self.mechanism is None:
            raise ValueError(
                f"{caller}() needs a mechanism - none was given at "
                "construction (to_array()/write_csv() don't need one "
                "at all; sweep()/solve_point() always do; "
                "fit_gulder()/fit_piecewise_phi() only do when "
                "require_plausible=True, resolve=True, and there's a "
                "flagged point that needs re-solving)"
            )

    def _mass_basis_composition(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        return (
            self.mechanism.mass_fractions(self.fuel, self.basis),
            self.mechanism.mass_fractions(self.oxidant, self.basis),
        )

    def _metadata(
        self, fuel: Dict[str, float], oxidant: Dict[str, float],
    ) -> Dict[str, Any]:
        return {
            "fuel": fuel,
            "oxidant": oxidant,
            "basis": "mass",
            "pressures": list(self.pressures),
            "temperatures": list(self.temperatures),
            "equivalence_ratios": list(self.equivalence_ratios),
            "egr_values": list(self.egr_values),
            "transport_model": self.transport_model,
            "mechanism_yaml": self.mechanism.mechanism_yaml,
        }

    def _cache_key(
        self, fuel: Dict[str, float], oxidant: Dict[str, float],
    ) -> str:
        # basis/transport_model are excluded: fuel/oxidant are already
        # mass basis here.
        payload = {
            "fuel": self._round_mapping(fuel),
            "oxidant": self._round_mapping(oxidant),
            "pressures": self._round_sorted(self.pressures),
            "temperatures": self._round_sorted(self.temperatures),
            "equivalence_ratios": self._round_sorted(
                self.equivalence_ratios
            ),
            "egr_values": self._round_sorted(self.egr_values),
            "mechanism_yaml": self._stable_mechanism_yaml(
                self.mechanism.mechanism_yaml
            ),
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _cache_path(
        self, directory, name: Optional[str],
        fuel: Dict[str, float], oxidant: Dict[str, float],
    ) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = (
            name if name is not None
            else f"lfs_{self._cache_key(fuel, oxidant)}"
        )
        return directory / f"{stem}.json"

    def _write_cache(
        self, cache_path: Path, results: List[LFSPoint],
        fuel: Dict[str, float], oxidant: Dict[str, float],
    ) -> None:
        # One plain JSON file, sized for thousands of points.
        payload = {"metadata": self._metadata(fuel, oxidant), "results": results}
        if self.plausibility is not None:
            payload["plausibility"] = [int(x) for x in self.plausibility]
        cache_path.write_text(json.dumps(payload, indent=2))

    def _read_cache(
        self, cache_path: Path,
    ) -> Tuple[List[LFSPoint], Optional["np.ndarray"]]:
        with cache_path.open() as handle:
            data = json.load(handle)
        plausibility = data.get("plausibility")
        if plausibility is not None:
            plausibility = np.array(plausibility, dtype=np.int64)
        return data["results"], plausibility

    def _append_partial(self, partial_path: Path, result: LFSPoint) -> None:
        # JSON Lines: O(1) append, and a crash costs only one line.
        with partial_path.open("a") as handle:
            handle.write(json.dumps(result) + "\n")

    def _read_partial(
        self, partial_path: Path, log: bool,
    ) -> List[LFSPoint]:
        results = []
        skipped = 0
        with partial_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    # Only the truncated line is lost; others still parse.
                    skipped += 1
        if skipped and log:
            print(
                f"laminar_flame_speed: {partial_path} has {skipped} "
                f"corrupted/truncated line(s) - ignoring them, "
                f"keeping {len(results)} valid point(s)"
            )
        return results

    def to_array(self) -> Tuple[List[str], "np.ndarray"]:
        """Converts `self.results` to `(column_names, array)` of the numeric
        columns (`p`, `T`, `phi`, `egr`, `sL`, `width`, `thickness`, `maxT`).

        Raises:
            ValueError: If `sweep()` hasn't been run.
        """
        if self.results is None:
            raise ValueError("sweep() hasn't been run yet - no results")
        array = np.array(
            [[result[c] for c in _FIXED_COLUMNS] for result in self.results]
        )
        return list(_FIXED_COLUMNS), array

    def curve(
        self, variable: str, fixed: Dict[str, float],
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """Extracts one `(x, sL)` curve from `self.results`.

        Takes points whose other three axes match `fixed` (`numpy.isclose`),
        sorted by `variable`.

        Args:
            variable: Axis used as `x`: `"p"`, `"T"`, `"phi"` or `"egr"`.
            fixed: Values of exactly the other three axes, e.g.
                `{"T": 900.0, "phi": 0.5, "egr": 0.0}` for `variable="p"`.

        Returns:
            `(x, sL)` as `numpy.ndarray`, sorted by `x`.

        Raises:
            ValueError: If `sweep()` hasn't been run, `variable` or `fixed`
                has wrong axes, or no point matches.
        """
        if variable not in _AXES:
            raise ValueError(
                f"variable must be one of {_AXES}, got {variable!r}"
            )
        expected = set(_AXES) - {variable}
        if set(fixed) != expected:
            raise ValueError(
                f"fixed must specify exactly {sorted(expected)} (the "
                f"axes other than {variable!r}), got {sorted(fixed)}"
            )

        columns, array = self.to_array()
        mask = self._axis_mask(columns, array, fixed)
        if not mask.any():
            raise ValueError(f"no results match {fixed}")

        x = array[mask, columns.index(variable)]
        sL = array[mask, columns.index("sL")]
        order = np.argsort(x)
        return x[order], sL[order]

    @staticmethod
    def _axis_mask(
        columns: List[str], array: "np.ndarray", fixed: Dict[str, float],
    ) -> "np.ndarray":
        mask = np.ones(len(array), dtype=bool)
        for key, value in fixed.items():
            mask &= np.isclose(array[:, columns.index(key)], float(value))
        return mask

    def write_csv(self, path) -> None:
        """Writes `self.results` to a plain CSV (`numpy.genfromtxt()` and
        Excel readable).

        Raises:
            ValueError: If `sweep()` hasn't been run.
        """
        if self.results is None:
            raise ValueError("sweep() hasn't been run yet - no results")
        species_names = sorted(
            {name for result in self.results for name in result["species"]}
        )
        rows = [
            [result[column] for column in _FIXED_COLUMNS]
            + [result["species"].get(name, 0.0) for name in species_names]
            for result in self.results
        ]
        header = ",".join(_FIXED_COLUMNS + species_names)
        np.savetxt(path, rows, delimiter=",", header=header, comments="")

    def _load_results_from_directory(self, directory, caller: str) -> None:
        if self.results is not None or directory is None:
            return
        cache_path = Path(directory) / self.cache_filename
        if not cache_path.is_file():
            raise ValueError(
                f"{caller}(): no cache file found at {cache_path} "
                "for this database's own configuration"
            )
        self.results, self.plausibility = self._read_cache(cache_path)

    def _resolve_flagged(
        self, indices, caller: str, num_points: int = _RESOLVE_NUM_POINTS,
        loglevel: int = 0, log: bool = False,
    ) -> set:
        # Re-solves flagged points in parallel with a different starting
        # width (_RESOLVE_WIDTH_SHRINK) and more grid points, mutating
        # self.results/self.plausibility. A point that re-validates
        # clean keeps its flag(s) with RESOLVED OR'd in (see
        # _is_trustworthy()). No Pool initializer: a small batch.
        if not indices:
            return set()
        self._check_mechanism(caller)
        fuel, oxidant = self._mass_basis_composition()
        products = self.mechanism.complete_combustion_products(
            fuel, oxidant, basis="mass", equivalence_ratio=1.0,
        )
        original_status = {i: int(self.plausibility[i]) for i in indices}
        with self.mechanism.frozen_path() as mech_path:
            tasks = [
                (
                    mech_path, fuel, oxidant, products,
                    self.results[i]["p"], self.results[i]["T"],
                    self.results[i]["phi"], self.results[i]["egr"],
                    "mass",
                    _RESOLVE_WIDTH_SHRINK * (
                        self.results[i]["width"]
                        if self.results[i]["width"] > 0 else 0.001
                    ),
                    num_points, self.transport_model, loglevel, log,
                )
                for i in indices
            ]
            available = _available_cpus()
            processes = min(
                self.max_processes or available, available, len(tasks),
            )
            if log:
                print(
                    f"laminar_flame_speed: re-solving {len(tasks)} "
                    f"flagged point(s) on {processes} worker(s)"
                )
                print(f"laminar_flame_speed: {_LOG_HEADER}")
            with Pool(processes=processes) as pool:
                resolved = pool.map(_solve_flame_point_star, tasks)
        for i, point in zip(indices, resolved):
            self.results[i] = point

        self.validate()
        fixed = set()
        for i in indices:
            if int(self.plausibility[i]) == int(PlausibilityStatus.GOOD):
                self.plausibility[i] = (
                    original_status[i] | int(PlausibilityStatus.RESOLVED)
                )
                fixed.add(i)
        return fixed

    @staticmethod
    def _bracket_reconstruct(grid, xs, ys, i, j) -> Optional[float]:
        # Walks outward from (i, j) along each axis to the nearest valid
        # neighbor on both sides and interpolates linearly, as
        # nonUniformTable2's runtime bilinear lookup would without this
        # node. None if neither axis brackets (table edge or too many
        # bad neighbours); never extrapolates.
        def bracket(axis, index, column):
            lo = None
            for k in range(index - 1, -1, -1):
                value = column(k)
                if not np.isnan(value):
                    lo = (axis[k], value)
                    break
            hi = None
            for k in range(index + 1, len(axis)):
                value = column(k)
                if not np.isnan(value):
                    hi = (axis[k], value)
                    break
            if lo is None or hi is None:
                return None
            (x0, y0), (x1, y1) = lo, hi
            return y0 + (y1 - y0) * (axis[index] - x0) / (x1 - x0)

        estimates = [
            e for e in (
                bracket(xs, i, lambda k: grid[k][j]),
                bracket(ys, j, lambda k: grid[i][k]),
            )
            if e is not None
        ]
        return float(np.mean(estimates)) if estimates else None

    def _fill_grid_gaps(self, nodes, grid_key, unfilled, xs, ys):
        # unfilled: (node_idx, i, j, phi, p, T) tuples whose
        # nodes[node_idx][grid_key][i][j] is nan. Returns (filled,
        # still_missing) via _bracket_reconstruct(): (phi, p, T,
        # grid_key) for quality reporting, (phi, p, T) for raises.
        filled = []
        still_missing = []
        for node_idx, i, j, phi, p, T in unfilled:
            grid = nodes[node_idx][grid_key]
            value = self._bracket_reconstruct(grid, xs, ys, i, j)
            if value is None:
                still_missing.append((phi, p, T))
            else:
                grid[i][j] = value
                filled.append((phi, p, T, grid_key))
        return filled, still_missing
