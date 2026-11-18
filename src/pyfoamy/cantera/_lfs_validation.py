"""Plausibility checks of a laminar flame speed sweep."""

import itertools
import time

from pyfoamy.cantera.mechanism import np
from pyfoamy.cantera._lfs_common import (
    _AXES,
    _FIXED_COLUMNS,
    _GULDER_SIGN_TOLERANCE,
    _MAX_DISCARD_FRACTION,
    _MONOTONICITY_TOLERANCE,
    _OUTLIER_THRESHOLD,
    _SUB_FIT_MIN_POINTS,
    _TRIM_FRACTION,
    _VALIDATE_EGR_MAX,
    PlausibilityStatus,
)
from pyfoamy.cantera._lfs_fits import (
    _gulder_sign_violations,
    evaluate_gulder,
)


class ValidationMixin:

    @staticmethod
    def _axis_values(columns, array):
        # Values actually in `self.results`; self.pressures etc. need not
        # match assigned/filtered results (e.g. in tests).
        return {
            a: sorted(set(array[:, columns.index(a)].tolist()))
            for a in _AXES
        }

    def _axis_slices(self, columns, array, variable, axis_values):
        # Yields, per combination of the other three axes, the row
        # indices of that slice sorted by `variable`.
        others = [a for a in _AXES if a != variable]
        var_col = columns.index(variable)
        for combo in itertools.product(*(axis_values[a] for a in others)):
            mask = self._axis_mask(columns, array, dict(zip(others, combo)))
            idx = np.flatnonzero(mask)
            if idx.size:
                yield idx[np.argsort(array[idx, var_col])]

    @staticmethod
    def _grid_row_index(columns, array, axis_values):
        # Maps the flat array onto a dense (p,T,phi,egr) grid of row
        # indices (sentinel len(array) = missing); neighbors are then a
        # vectorized shifted lookup.
        shape = tuple(len(axis_values[a]) for a in _AXES)
        grid = np.full(shape, len(array), dtype=np.int64)
        idx = tuple(
            np.searchsorted(axis_values[a], array[:, columns.index(a)])
            for a in _AXES
        )
        grid[idx] = np.arange(len(array))
        return grid

    @staticmethod
    def _point_str(columns, array, i):
        return " ".join(
            f"{a}={array[i, columns.index(a)]:.4g}"
            for a in _FIXED_COLUMNS[:5]
        )

    def _monotonicity_flags(
        self, columns, array, tolerance, axis_values, verbose=0,
    ):
        # Cantera can only overshoot, so in a violating neighbor pair
        # the point that should be lower is the bad one.
        flagged = np.zeros(len(array), dtype=bool)
        sL_col = columns.index("sL")

        def flag(i, reason):
            flagged[i] = True
            if verbose >= 2:
                print(
                    f"validate(): MONOTONICITY discard ({reason}) - "
                    f"{self._point_str(columns, array, i)}", flush=True,
                )

        def check(variable, non_increasing):
            for order in self._axis_slices(
                columns, array, variable, axis_values,
            ):
                sL = array[order, sL_col]
                valid = ~np.isnan(sL)
                for k in range(len(order) - 1):
                    if not (valid[k] and valid[k + 1]):
                        continue
                    if non_increasing:
                        # expect sL[k+1] <= sL[k]; violation -> later
                        # (higher-axis) point overshot.
                        if sL[k + 1] > sL[k] * (1 + tolerance):
                            flag(order[k + 1], f"{variable} up, sL up")
                    else:
                        # expect sL[k+1] >= sL[k]; violation -> earlier
                        # (lower-axis) point overshot.
                        if sL[k] > sL[k + 1] * (1 + tolerance):
                            flag(order[k], f"{variable} up, sL down")

        check("p", non_increasing=True)
        check("egr", non_increasing=True)
        check("T", non_increasing=False)

        # phi: only a rule outside 0.8-1.2, moving away from the center;
        # a pair straddling the peak band has none.
        phi_col = columns.index("phi")
        for order in self._axis_slices(columns, array, "phi", axis_values):
            sL = array[order, sL_col]
            phi = array[order, phi_col]
            valid = ~np.isnan(sL)
            for k in range(len(order) - 1):
                if not (valid[k] and valid[k + 1]):
                    continue
                pa, pb = phi[k], phi[k + 1]
                if pa <= 0.8 and pb <= 0.8:
                    if sL[k] > sL[k + 1] * (1 + tolerance):
                        flag(order[k], "phi away from center, sL up")
                elif pa >= 1.2 and pb >= 1.2:
                    if sL[k + 1] > sL[k] * (1 + tolerance):
                        flag(order[k + 1], "phi away from center, sL up")
        return flagged

    _PROGRESS_EVERY = 200

    @staticmethod
    def _power_law_exponent(x, y):
        # Closed-form OLS slope/intercept for y = intercept + exponent*x
        # (a 2-column lstsq without the overhead). None if degenerate.
        x_mean = np.mean(x)
        var_x = np.sum(np.square(x - x_mean))
        if var_x == 0:
            return None, None
        exponent = np.sum((x - x_mean) * (y - np.mean(y))) / var_x
        intercept = np.mean(y) - exponent * x_mean
        return exponent, intercept

    def _pressure_slice_flags(
        self, columns, array, min_points, axis_values, verbose=0,
    ):
        # Within a p-slice Su0 is constant, so ln(sL) = ln(Su0) +
        # beta*ln(p) is a 2-parameter OLS fit; no lstsq needed.
        flagged = np.zeros(len(array), dtype=bool)
        sL_col = columns.index("sL")

        n_slices = 0
        for order in self._axis_slices(columns, array, "p", axis_values):
            n_slices += 1
            if verbose >= 2 and n_slices % self._PROGRESS_EVERY == 0:
                print(
                    f"validate(): sub-fit(p) - {n_slices} slices...",
                    flush=True,
                )
            idx = list(order[~np.isnan(array[order, sL_col])])
            removed = []
            while len(idx) >= min_points:
                p = np.log(array[idx, columns.index("p")])
                sL = np.log(array[idx, sL_col])
                beta, intercept = self._power_law_exponent(p, sL)
                if beta is None:
                    break
                if beta <= _GULDER_SIGN_TOLERANCE:
                    for i in removed:
                        flagged[i] = True
                    break
                residual = np.abs(sL - (intercept + beta * p))
                worst_pos = int(np.argmax(residual))
                worst = idx.pop(worst_pos)
                removed.append(worst)
                if verbose >= 2:
                    print(
                        f"validate(): SUB_FIT(p) discard (beta="
                        f"{beta:.3g}) - "
                        f"{self._point_str(columns, array, worst)}",
                        flush=True,
                    )
        return flagged

    def _egr_slice_flags(
        self, columns, array, min_points, egr_max, axis_values, verbose=0,
    ):
        # Su0 cancels in sL(egr)/sL(0)=1-f*egr, so f is closed form (as
        # in fit_piecewise_phi()). No egr=0 point means no anchor: skip.
        flagged = np.zeros(len(array), dtype=bool)
        sL_col = columns.index("sL")
        egr_col = columns.index("egr")

        n_slices = 0
        for order in self._axis_slices(columns, array, "egr", axis_values):
            n_slices += 1
            if verbose >= 2 and n_slices % self._PROGRESS_EVERY == 0:
                print(
                    f"validate(): sub-fit(egr) - {n_slices} slices...",
                    flush=True,
                )
            idx = order[array[order, egr_col] <= egr_max]
            idx = idx[~np.isnan(array[idx, sL_col])]
            zero = idx[array[idx, egr_col] == 0.0]
            if zero.size == 0:
                continue
            sL0 = float(array[zero[0], sL_col])
            # The egr=0 anchor's residual is zero by construction, so
            # only the points f is solved from are `active`.
            active = list(idx[array[idx, egr_col] > 0.0])
            if len(active) < min_points - 1:
                continue
            removed = []
            while len(active) >= min_points - 1:
                egr = array[active, egr_col]
                y = array[active, sL_col] / sL0
                f = float(np.sum(egr * (1 - y)) / np.sum(egr * egr))
                if f >= -_GULDER_SIGN_TOLERANCE:
                    for i in removed:
                        flagged[i] = True
                    break
                residual = np.abs(y - (1 - f * egr))
                worst_pos = int(np.argmax(residual))
                worst = active.pop(worst_pos)
                removed.append(worst)
                if verbose >= 2:
                    print(
                        f"validate(): SUB_FIT(egr) discard (f={f:.3g}) "
                        f"- {self._point_str(columns, array, worst)}",
                        flush=True,
                    )
        return flagged

    def _temperature_growth_flags(
        self, columns, array, min_points, axis_values, verbose=0,
    ):
        # Grows from the lowest T upward, flagging a newly added point
        # that breaks plausibility; alpha is a closed-form OLS slope.
        flagged = np.zeros(len(array), dtype=bool)
        sL_col = columns.index("sL")
        T_col = columns.index("T")

        n_slices = 0
        for order in self._axis_slices(columns, array, "T", axis_values):
            n_slices += 1
            if verbose >= 2 and n_slices % self._PROGRESS_EVERY == 0:
                print(
                    f"validate(): sub-fit(T) - {n_slices} slices...",
                    flush=True,
                )
            idx = order[~np.isnan(array[order, sL_col])]
            if idx.size < min_points:
                continue
            active = list(idx[:min_points])
            for candidate in idx[min_points:]:
                trial = active + [int(candidate)]
                alpha, _ = self._power_law_exponent(
                    np.log(array[trial, T_col]), np.log(array[trial, sL_col]),
                )
                if alpha is None or alpha < -_GULDER_SIGN_TOLERANCE:
                    flagged[candidate] = True
                    if verbose >= 2:
                        shown = alpha if alpha is not None else float("nan")
                        print(
                            f"validate(): SUB_FIT(T) reject (alpha="
                            f"{shown:.3g}) - "
                            f"{self._point_str(columns, array, candidate)}",
                            flush=True,
                        )
                else:
                    active = trial
        return flagged

    def _global_robust_fit(
        self, columns, array, f_bounds, trim_fraction, max_discard_fraction,
        verbose=0,
    ):
        # Drops the worst trim_fraction per round until sign-plausible;
        # batched as a per-removal nested f-search is too slow.
        p_col, T_col, phi_col, egr_col, sL_col = (
            columns.index(a) for a in ("p", "T", "phi", "egr", "sL")
        )
        n_total = len(array)
        active = np.ones(n_total, dtype=bool)
        discarded = 0
        round_num = 0
        while True:
            round_num += 1
            idx = np.flatnonzero(active)
            if idx.size < 2:
                raise ValueError(
                    "validate(): ran out of points trying to reach a "
                    "physically plausible Gulder reference fit - the "
                    "sweep likely has a widespread issue"
                )
            fit = self._fit_gulder_raw(
                array[idx, p_col], array[idx, T_col], array[idx, phi_col],
                array[idx, egr_col], array[idx, sL_col], f_bounds,
            )
            violations = _gulder_sign_violations(fit["coefficients"])
            if not violations:
                return fit, active, discarded
            residual = np.abs(fit["target"] - fit["design"] @ fit["coef"])
            n_remove = min(
                max(1, round(trim_fraction * n_total)), idx.size - 1,
            )
            worst = np.argsort(residual)[-n_remove:]
            if verbose >= 2:
                print(
                    f"validate(): global fit round {round_num} - "
                    f"implausible ({'; '.join(violations)}) - "
                    f"discarding {n_remove} worst-residual point(s) "
                    f"({discarded}/{n_total} so far)", flush=True,
                )
            active[idx[worst]] = False
            discarded += n_remove
            if discarded / n_total > max_discard_fraction:
                raise ValueError(
                    f"validate(): discarded {discarded}/{n_total} "
                    f"points ({100*discarded/n_total:.1f}%) trying to "
                    "reach a physically plausible Gulder reference "
                    "fit, exceeding max_discard_fraction="
                    f"{100*max_discard_fraction:.0f}% - the sweep "
                    "likely has a widespread issue, not just isolated "
                    "bad points"
                )

    def _global_outlier_flags(
        self, columns, array, coefficients, threshold, grid_row,
        verbose=0,
    ):
        # Large deviation from the global fit, isolated from every
        # existing neighbor; vectorized via grid_row.
        sL_col = columns.index("sL")
        p_col, T_col, phi_col, egr_col = (
            columns.index(a) for a in _AXES
        )
        sL = array[:, sL_col]
        predicted = evaluate_gulder(
            coefficients, array[:, p_col], array[:, T_col],
            array[:, phi_col], array[:, egr_col],
        )
        relative_error = np.full(len(array), np.inf)
        valid = ~np.isnan(sL)
        relative_error[valid] = np.abs(
            (sL[valid] - predicted[valid]) / sL[valid]
        )
        is_outlier = relative_error > threshold

        valid_grid = grid_row < len(array)
        outlier_grid = np.append(is_outlier, False)[grid_row]
        has_neighbor = np.zeros_like(outlier_grid)
        neighbor_is_outlier = np.zeros_like(outlier_grid)
        for axis in range(len(_AXES)):
            for src_slice, dst_slice in (
                (slice(1, None), slice(0, -1)),
                (slice(0, -1), slice(1, None)),
            ):
                src = [slice(None)] * len(_AXES)
                dst = [slice(None)] * len(_AXES)
                src[axis], dst[axis] = src_slice, dst_slice
                src, dst = tuple(src), tuple(dst)
                has_neighbor[dst] |= valid_grid[src]
                neighbor_is_outlier[dst] |= outlier_grid[src]

        flagged_grid = outlier_grid & has_neighbor & ~neighbor_is_outlier
        flagged = np.zeros(len(array) + 1, dtype=bool)
        flagged[grid_row.ravel()] = flagged_grid.ravel()
        flagged = flagged[:len(array)]

        if verbose >= 2:
            for i in np.flatnonzero(flagged):
                print(
                    f"validate(): GLOBAL_OUTLIER discard "
                    f"(relative_error={relative_error[i]:.3g}) - "
                    f"{self._point_str(columns, array, i)}", flush=True,
                )
        return flagged

    def validate(
        self,
        monotonicity_tolerance: float = _MONOTONICITY_TOLERANCE,
        sub_fit_min_points: int = _SUB_FIT_MIN_POINTS,
        egr_max: float = _VALIDATE_EGR_MAX,
        f_bounds=(0.5, 5.0),
        trim_fraction: float = _TRIM_FRACTION,
        max_discard_fraction: float = _MAX_DISCARD_FRACTION,
        outlier_threshold: float = _OUTLIER_THRESHOLD,
        verbose: int = 0,
    ) -> "np.ndarray":
        """Flags each point of `self.results` with a `PlausibilityStatus`
        bitmask, diagnosing wrong (not just `nan`) `sL` values.

        Checks: `NON_CONVERGED` (`sL` is `nan`); `MONOTONICITY` (wrong sL
        direction against a neighbor); `SUB_FIT` (a per-slice pressure/EGR/
        temperature fit had to discard the point); `GLOBAL_OUTLIER` (large,
        isolated deviation from a sweep-wide fit). A point can have several.

        Args:
            monotonicity_tolerance: Relative tolerance of the pairwise check.
            sub_fit_min_points: Minimum points for a per-slice sub-fit.
            egr_max: Restricts the EGR sub-fit to this window.
            f_bounds: Search range of `f` in the global fit (sub-fits are
                closed form).
            trim_fraction: Fraction of the total discarded per round by the
                global robust fit.
            max_discard_fraction: Raise if the robust fit must discard more
                than this.
            outlier_threshold: Relative error threshold of `GLOBAL_OUTLIER`.
            verbose: `0` silent; `1` one line per phase (counts, time); `2`
                also every discard/reject (point and reason) and periodic
                progress in the slowest phases.

        Returns:
            `self.plausibility`, an int array of `PlausibilityStatus` bits in
            `self.results` order.

        Raises:
            ValueError: If `sweep()` hasn't been run or the robust fit exceeds
                `max_discard_fraction`.
        """
        if self.results is None:
            raise ValueError("sweep() hasn't been run yet - no results")
        columns, array = self.to_array()
        n = len(array)
        status = np.zeros(n, dtype=np.int64)
        # Computed once: recomputing per call is costly on a large sweep.
        axis_values = self._axis_values(columns, array)

        def report(phase, count, t0):
            if verbose:
                print(
                    f"validate(): {phase} - {count} point(s) "
                    f"({time.time()-t0:.1f}s)", flush=True,
                )

        t0 = time.time()
        non_converged = np.isnan(array[:, columns.index("sL")])
        status[non_converged] |= int(PlausibilityStatus.NON_CONVERGED)
        report("NON_CONVERGED", int(non_converged.sum()), t0)

        t0 = time.time()
        mono = self._monotonicity_flags(
            columns, array, monotonicity_tolerance, axis_values,
            verbose=verbose,
        )
        status[mono] |= int(PlausibilityStatus.MONOTONICITY)
        report("MONOTONICITY", int(mono.sum()), t0)

        t0 = time.time()
        p_sub = self._pressure_slice_flags(
            columns, array, sub_fit_min_points, axis_values,
            verbose=verbose,
        )
        report("SUB_FIT(p)", int(p_sub.sum()), t0)
        t0 = time.time()
        egr_sub = self._egr_slice_flags(
            columns, array, sub_fit_min_points, egr_max, axis_values,
            verbose=verbose,
        )
        report("SUB_FIT(egr)", int(egr_sub.sum()), t0)
        t0 = time.time()
        T_sub = self._temperature_growth_flags(
            columns, array, sub_fit_min_points, axis_values,
            verbose=verbose,
        )
        report("SUB_FIT(T)", int(T_sub.sum()), t0)
        sub_fit = p_sub | egr_sub | T_sub
        status[sub_fit] |= int(PlausibilityStatus.SUB_FIT)

        t0 = time.time()
        fit, _active, discarded = self._global_robust_fit(
            columns, array, f_bounds, trim_fraction, max_discard_fraction,
            verbose=verbose,
        )
        report("global fit", discarded, t0)
        t0 = time.time()
        grid_row = self._grid_row_index(columns, array, axis_values)
        global_outlier = self._global_outlier_flags(
            columns, array, fit["coefficients"], outlier_threshold,
            grid_row, verbose=verbose,
        )
        status[global_outlier] |= int(PlausibilityStatus.GLOBAL_OUTLIER)
        report("GLOBAL_OUTLIER", int(global_outlier.sum()), t0)

        self.plausibility = status
        return status

    def report_plausibility(self) -> None:
        """Prints per-flag counts of the last `validate()`.

        Raises:
            ValueError: If `validate()` hasn't been run.
        """
        if self.plausibility is None:
            raise ValueError("validate() hasn't been run yet")
        n = len(self.plausibility)
        good = int(np.sum(
            self.plausibility == int(PlausibilityStatus.GOOD)
        ))
        print(f"validate(): {n} points, {good} GOOD ({100*good/n:.1f}%)")
        for flag in PlausibilityStatus:
            if flag in (PlausibilityStatus.GOOD, PlausibilityStatus.UNCHECKED):
                continue
            count = int(np.sum((self.plausibility & int(flag)) != 0))
            print(f"  {flag.name}: {count} ({100*count/n:.1f}%)")
