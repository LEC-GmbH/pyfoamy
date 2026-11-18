"""Gulder and piecewise-phi fits of a laminar flame speed sweep."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from pyfoamy.cantera.mechanism import np
from pyfoamy.cantera._lfs_common import (
    _BAD_FLAGS,
    _GULDER_IDENTIFIABILITY_TOLERANCE,
    _GULDER_PHI_PEAK,
    _GULDER_PHI_PEAK_SHIFT_PROBE,
    _GULDER_PREF,
    _GULDER_SIGN_TOLERANCE,
    _GULDER_TREF,
    _GULDER_ZERO_FLOOR,
    _HARD_BAD_FLAGS,
    _is_trustworthy,
    _PIECEWISE_PHI_SIGN_TOLERANCE,
    _RESOLVE_NUM_POINTS,
    PlausibilityStatus,
)


def evaluate_gulder(
    coefficients: Dict[str, float], p, T, phi, egr,
):
    """Evaluates the Gulder correlation with `fit_gulder()` coefficients.

    Args:
        coefficients: `{"W", "eta", "xi", "alpha", "beta", "f"}`.
        p: Pressure(s) [Pa].
        T: Unburnt mixture temperature(s) [K].
        phi: Equivalence ratio(s).
        egr: EGR mass fraction(s).

    Returns:
        Laminar flame speed(s) [m/s].
    """
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)
    phi = np.asarray(phi, dtype=float)
    egr = np.asarray(egr, dtype=float)
    Su0 = (
        coefficients["W"] * np.power(phi, coefficients["eta"])
        * np.exp(-coefficients["xi"] * np.square(phi - _GULDER_PHI_PEAK))
    )
    return (
        Su0 * np.power(p / _GULDER_PREF, coefficients["beta"])
        * np.power(T / _GULDER_TREF, coefficients["alpha"])
        * (1 - coefficients["f"] * egr)
    )


def _gulder_sign_violations(coefficients: Dict[str, float]) -> List[str]:
    # Used to reject fit_gulder()'s result and, unvalidated, by
    # validate()'s robust-fit loops.
    violations = []
    if coefficients["eta"] < -_GULDER_SIGN_TOLERANCE:
        violations.append(
            f"eta={coefficients['eta']:.4g} < 0 - physically, "
            "Su0=W*phi^eta*... must vanish (not diverge) as "
            "phi->0 (eta should be >= 0)"
        )
    if coefficients["xi"] < -_GULDER_SIGN_TOLERANCE:
        violations.append(
            f"xi={coefficients['xi']:.4g} < 0 - physically, "
            "flame speed should fall off away from "
            f"phi={_GULDER_PHI_PEAK:g}, not rise (xi should "
            "be >= 0)"
        )
    if coefficients["alpha"] < -_GULDER_SIGN_TOLERANCE:
        violations.append(
            f"alpha={coefficients['alpha']:.4g} < 0 - "
            "physically, flame speed should increase with "
            "unburnt temperature (alpha should be >= 0)"
        )
    if coefficients["beta"] > _GULDER_SIGN_TOLERANCE:
        violations.append(
            f"beta={coefficients['beta']:.4g} > 0 - "
            "physically, flame speed should decrease with "
            "pressure (beta should be <= 0)"
        )
    return violations


class FitsMixin:

    def fit_gulder(
        self, f_bounds=(0.5, 5.0),
        p=None, T=None, phi=None, egr=None,
        directory=None,
        require_plausible: bool = False,
        resolve: bool = True,
        resolve_num_points: int = _RESOLVE_NUM_POINTS,
        log: bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Fits the Gulder correlation (see `evaluate_gulder`) to the results.

        An axis constant across the results is excluded as unidentifiable.
        Prints a fit-quality summary.

        Args:
            f_bounds: Search range of `f` (EGR should not raise flame speed).
            p: Optional `(low, high)` restricting the fit to a pressure range.
            T: Same, for temperature.
            phi: Same, for equivalence ratio.
            egr: Same, for EGR.
            directory: If `self.results` is `None`, load `self.cache_filename`
                from here first.
            require_plausible: Exclude points `validate()` flags
                `MONOTONICITY`/`SUB_FIT`/`GLOBAL_OUTLIER` (running it first if
                needed). `NON_CONVERGED` (`sL=nan`) points are still floored
                and included: they are flammability-limit data.
            resolve: With `require_plausible`, re-solve flagged points with
                different numerics first (needs `self.mechanism`); only points
                still flagged are excluded.
            resolve_num_points: `num_points` for the re-solve.
            log: Print re-solve progress.

        Returns:
            `(coefficients, quality)`.

        Raises:
            ValueError: If there are no results, the coefficients are
                physically implausible (e.g. a sign violation), or (with
                `require_plausible`) all points in the restriction are flagged
                and unresolvable.
        """
        self._load_results_from_directory(directory, "fit_gulder")

        columns, array = self.to_array()
        T_all = array[:, columns.index("T")]
        p_all = array[:, columns.index("p")]
        egr_all = array[:, columns.index("egr")]
        phi_all = array[:, columns.index("phi")]
        sL_all = array[:, columns.index("sL")]

        restrict = np.ones(len(sL_all), dtype=bool)
        for name, values, bounds in (
            ("p", p_all, p), ("T", T_all, T),
            ("phi", phi_all, phi), ("egr", egr_all, egr),
        ):
            if bounds is not None:
                low, high = bounds
                restrict &= (values >= low) & (values <= high)
        if not np.any(restrict):
            raise ValueError(
                "fit_gulder(): the given p/T/phi/egr restriction(s) "
                f"exclude every point in self.results (p={p!r}, "
                f"T={T!r}, phi={phi!r}, egr={egr!r})"
            )

        if require_plausible:
            if self.plausibility is None:
                self.validate()
            flagged = (self.plausibility & int(_BAD_FLAGS)) != 0
            if resolve:
                candidates = np.flatnonzero(flagged & restrict)
                if candidates.size:
                    self._resolve_flagged(
                        candidates.tolist(), "fit_gulder",
                        num_points=resolve_num_points, log=log,
                    )
                    sL_all = np.array([r["sL"] for r in self.results])
            # A nan point's undefined relative error reads as infinite in
            # _global_outlier_flags(), so NON_CONVERGED always carries
            # GLOBAL_OUTLIER; that is no independent evidence, so
            # NON_CONVERGED wins.
            exclude = (
                (self.plausibility & int(_HARD_BAD_FLAGS)) != 0
            ) & ((self.plausibility & int(PlausibilityStatus.RESOLVED)) == 0
            ) & ((self.plausibility & int(PlausibilityStatus.NON_CONVERGED)) == 0)
            restrict &= ~exclude
            if not np.any(restrict):
                raise ValueError(
                    "fit_gulder(): every point passing the given "
                    "p/T/phi/egr restriction(s) is flagged implausible "
                    "(MONOTONICITY/SUB_FIT/GLOBAL_OUTLIER) and could "
                    "not be resolved - widen the restriction or "
                    "investigate the sweep"
                )

        fit = self._fit_gulder_raw(
            p_all[restrict], T_all[restrict], phi_all[restrict],
            egr_all[restrict], sL_all[restrict], f_bounds,
        )
        coefficients = fit["coefficients"]

        violations = _gulder_sign_violations(coefficients)
        if violations:
            note = ""
            feature_names = fit["feature_names"]
            eta_or_xi_violates = (
                "eta" in feature_names and "xi" in feature_names
                and (
                    coefficients["eta"] < -_GULDER_SIGN_TOLERANCE
                    or coefficients["xi"] < -_GULDER_SIGN_TOLERANCE
                )
            )
            if eta_or_xi_violates:
                # Diagnostic only: if shifting the assumed Gulder peak
                # barely changes predictions, eta/xi weren't separable.
                phi_values = fit["phi"]
                xi_idx = feature_names.index("xi")
                shifted_columns = list(fit["feature_columns"])
                shifted_columns[xi_idx] = -np.square(
                    phi_values
                    - (_GULDER_PHI_PEAK + _GULDER_PHI_PEAK_SHIFT_PROBE)
                )
                shifted_design = np.column_stack(shifted_columns)
                shifted_coef, *_ = np.linalg.lstsq(
                    shifted_design, fit["target"], rcond=None
                )
                max_diff = float(np.max(np.abs(
                    fit["design"] @ fit["coef"]
                    - shifted_design @ shifted_coef
                )))
                if max_diff < _GULDER_IDENTIFIABILITY_TOLERANCE:
                    note = (
                        " - likely cause: only "
                        f"{len(set(phi_values))} distinct equivalence "
                        "ratio value(s) in the fitted data, too "
                        "narrow/few to separate eta and xi (confirmed: "
                        "shifting the assumed Gulder peak location "
                        f"changes predictions by only {max_diff:.3g}, "
                        "not a real effect) - sweep a wider phi range "
                        "rather than fitting this data as-is"
                    )
            raise ValueError(
                "fit_gulder(): physically implausible fit - "
                + "; ".join(violations) + note
            )

        quality = fit["quality"]
        print(
            f"fit_gulder(): n={quality['n_points']} valid points\n"
            "  relative error: "
            f"mean={100*quality['relative_error_mean']:.3g}% "
            f"std={100*quality['relative_error_std']:.3g}% "
            f"max={100*quality['relative_error_max']:.3g}%\n"
            f"  R2={quality['r2']:.4g}  "
            f"adjusted R2={quality['r2_adjusted']:.4g}"
        )
        return coefficients, quality

    def _fit_gulder_raw(
        self, p, T, phi, egr, sL_actual, f_bounds,
    ) -> Dict[str, Any]:
        # No plausibility check or printing: reused by fit_gulder() and
        # validate()'s robust loops, which fit data not yet validated.
        valid = ~np.isnan(sL_actual)
        sL = np.where(valid, sL_actual, _GULDER_ZERO_FLOOR)
        log_sL = np.log(sL)

        feature_names = ["ln_W"]
        feature_columns = [np.ones_like(log_sL)]
        if not self._is_constant(phi):
            feature_names += ["eta", "xi"]
            feature_columns += [
                np.log(phi), -np.square(phi - _GULDER_PHI_PEAK),
            ]
        if not self._is_constant(T):
            feature_names.append("alpha")
            feature_columns.append(np.log(T / _GULDER_TREF))
        if not self._is_constant(p):
            feature_names.append("beta")
            feature_columns.append(np.log(p / _GULDER_PREF))
        design = np.column_stack(feature_columns)

        def solve_linear(target):
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
            return coef, float(np.sum(np.square(target - design @ coef)))

        if self._is_constant(egr):
            f = 0.0
            target = log_sL
        else:
            def residual(f):
                # +inf for an infeasible f (1-f*egr<=0), not nan -
                # np.argmin would otherwise pick the first nan.
                if np.any(1 - f * egr <= 0):
                    return float("inf")
                target = log_sL - np.log(1 - f * egr)
                return solve_linear(target)[1]

            f = self._grid_search_1d(residual, *f_bounds)
            if residual(f) == float("inf"):
                raise ValueError(
                    f"fit_gulder(): no f in f_bounds={f_bounds!r} keeps "
                    "1 - f*egr positive for every point (max egr in "
                    f"the data is {float(np.max(egr)):.4g}) - narrow "
                    "f_bounds to below "
                    f"{1.0 / float(np.max(egr)):.4g}"
                )
            target = log_sL - np.log(1 - f * egr)

        coef, _ = solve_linear(target)

        named = dict(zip(feature_names, coef))
        coefficients = {
            "W": float(np.exp(coef[0])),
            "eta": named.get("eta", 0.0),
            "xi": named.get("xi", 0.0),
            "alpha": named.get("alpha", 0.0),
            "beta": named.get("beta", 0.0),
            "f": f,
        }

        sL_pred = evaluate_gulder(coefficients, p, T, phi, egr)
        n_points = int(valid.sum())
        if n_points:
            relative_error = (
                (sL_actual[valid] - sL_pred[valid]) / sL_actual[valid]
            )
            residual_mean = float(np.mean(relative_error))
            residual_std = float(np.std(relative_error))
            residual_max = float(np.max(np.abs(relative_error)))
            ss_res = float(np.sum(
                np.square(sL_actual[valid] - sL_pred[valid])
            ))
            ss_tot = float(np.sum(np.square(
                sL_actual[valid] - np.mean(sL_actual[valid])
            )))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            n_params = design.shape[1] + (0 if self._is_constant(egr) else 1)
            denom = n_points - n_params - 1
            r2_adjusted = (
                1.0 - (1.0 - r2) * (n_points - 1) / denom
                if denom > 0 else float("nan")
            )
        else:
            residual_mean = residual_std = residual_max = float("nan")
            r2 = r2_adjusted = float("nan")

        quality = {
            "n_points": n_points,
            "relative_error_mean": residual_mean,
            "relative_error_std": residual_std,
            "relative_error_max": residual_max,
            "r2": r2,
            "r2_adjusted": r2_adjusted,
        }
        return {
            "coefficients": coefficients, "coef": coef, "quality": quality,
            "feature_names": feature_names,
            "feature_columns": feature_columns,
            "design": design, "target": target, "phi": phi,
        }

    def fit_piecewise_phi(
        self, egr_max: float = 0.10,
        phi_nodes: Optional[Sequence[float]] = None,
        pressures: Optional[Sequence[float]] = None,
        temperatures: Optional[Sequence[float]] = None,
        out_of_bounds: str = "clamp",
        directory=None,
        require_plausible: bool = False,
        resolve: bool = True,
        resolve_num_points: int = _RESOLVE_NUM_POINTS,
        log: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Builds a piecewise-in-phi flame speed table for an
        `unstrainedLaminarFlameSpeed` sub-dict: per phi node `Su0(p,Tu)` and a
        linear-in-egr slope `f(p,Tu)`.

        `Su0` is the sweep's exact `egr=0` value at each `(phi_node, p, T)`
        corner; only `f` is fit.

        Args:
            egr_max: Only points with `egr <= egr_max` enter the `f` fit.
            phi_nodes: Table phi axis; each must match a `phi` in
                `self.results` (default: every swept value).
            pressures: Table pressure axis, same rule.
            temperatures: Table temperature axis, same rule.
            out_of_bounds: `Function2s::NonUniformTable2` out-of-bounds
                behaviour.
            directory: As in `fit_gulder()`.
            require_plausible: A corner whose source points `validate()` flags
                is re-solved first (see `resolve`); if still flagged it is
                interpolated linearly from the nearest trustworthy corners in
                the phi node's `(p, T)` grid (as `nonUniformTable2` would at
                runtime), never extrapolated. A corner with no trustworthy
                bracket on either axis still raises.
            resolve: With `require_plausible`, re-solve (needs
                `self.mechanism`) before reconstructing.
            resolve_num_points: `num_points` for the re-solve.
            log: Print re-solve progress.

        Returns:
            `(table, quality)`; `table` is `{"phi0": {"Phi", "pTu", "f"}, ...}`
            per `phi_nodes` value. `quality` adds `"reconstructed_corners"`
            (empty unless `require_plausible`): `[(phi, p, T, "pTu"|"f"), ...]`
            for interpolated values, which are never put in `self.results`.

        Raises:
            ValueError: If there are no results, a `(phi_node, p, T)` corner
                can't be fit or (with `require_plausible`) reconstructed, or a
                fitted `f` is negative.
        """
        self._load_results_from_directory(directory, "fit_piecewise_phi")
        if self.results is None:
            raise ValueError("sweep() hasn't been run yet - no results")

        results = self.results
        phi_nodes = sorted(
            self._to_float_list(phi_nodes) if phi_nodes is not None
            else {r["phi"] for r in results}
        )
        pressures = sorted(
            self._to_float_list(pressures) if pressures is not None
            else {r["p"] for r in results}
        )
        temperatures = sorted(
            self._to_float_list(temperatures) if temperatures is not None
            else {r["T"] for r in results}
        )

        # One vectorised isclose per grid value; scanning every result
        # per corner is quadratic (minutes on a full sweep).
        def near(key, values):
            column = np.array([r[key] for r in results])
            return {v: np.isclose(column, v) for v in values}

        at_phi = near("phi", phi_nodes)
        at_p = near("p", pressures)
        at_T = near("T", temperatures)

        def corner(phi, p, T):
            return np.flatnonzero(at_phi[phi] & at_p[p] & at_T[T])

        if require_plausible:
            if self.plausibility is None:
                self.validate()
            candidates = set()
            for phi in phi_nodes:
                for p in pressures:
                    for T in temperatures:
                        for i in corner(phi, p, T):
                            egr = results[i]["egr"]
                            if not (
                                egr <= egr_max or np.isclose(egr, 0.0)
                            ):
                                continue
                            if self.plausibility[i] & int(_BAD_FLAGS):
                                candidates.add(int(i))
            if resolve and candidates:
                self._resolve_flagged(
                    sorted(candidates), "fit_piecewise_phi",
                    num_points=resolve_num_points, log=log,
                )
            trustworthy_id = {
                id(r): _is_trustworthy(int(self.plausibility[i]))
                for i, r in enumerate(results)
            }
        else:
            trustworthy_id = None

        # A pure egr=0 sweep is legitimate, not a data gap; checked
        # globally so a real gap elsewhere still raises below.
        has_egr_variation = any(r["egr"] > 0 for r in results)

        unfilled_baseline = []
        unfilled_slope = []
        negative_f = []
        relative_residuals = []
        nodes = []
        for node_idx, phi in enumerate(phi_nodes):
            pTu_grid = []
            f_grid = []
            for i, p in enumerate(pressures):
                pTu_row = []
                f_row = []
                for j, T in enumerate(temperatures):
                    rows = [results[k] for k in corner(phi, p, T)]
                    baseline = [
                        r for r in rows
                        if np.isclose(r["egr"], 0.0)
                        and np.isfinite(r["sL"]) and r["sL"] != 0.0
                        and (
                            trustworthy_id is None
                            or trustworthy_id[id(r)]
                        )
                    ]
                    if not baseline:
                        unfilled_baseline.append(
                            (node_idx, i, j, phi, p, T)
                        )
                        # A pure egr=0 sweep never needs f; a missing
                        # baseline must not drag f reconstruction (or a
                        # raise) into this irrelevant axis.
                        if has_egr_variation:
                            unfilled_slope.append(
                                (node_idx, i, j, phi, p, T)
                            )
                        pTu_row.append(float("nan"))
                        f_row.append(float("nan"))
                        continue
                    sL0 = baseline[0]["sL"]
                    window = [
                        r for r in rows
                        if r["egr"] <= egr_max and np.isfinite(r["sL"])
                        and (
                            trustworthy_id is None
                            or trustworthy_id[id(r)]
                        )
                    ]
                    if len(window) < 2:
                        pTu_row.append(sL0)
                        if has_egr_variation:
                            unfilled_slope.append(
                                (node_idx, i, j, phi, p, T)
                            )
                            f_row.append(float("nan"))
                        else:
                            # Nothing to fit f from - no EGR effect.
                            f_row.append(0.0)
                        continue
                    egr = np.array([r["egr"] for r in window])
                    y = np.array([r["sL"] for r in window]) / sL0
                    denom = float(np.sum(egr * egr))
                    f = (
                        float(np.sum(egr * (1 - y)) / denom)
                        if denom > 0 else float("nan")
                    )
                    if not np.isfinite(f):
                        unfilled_slope.append((node_idx, i, j, phi, p, T))
                    else:
                        if f < -_PIECEWISE_PHI_SIGN_TOLERANCE:
                            negative_f.append((phi, p, T, f))
                        relative_residuals.extend(
                            (y - (1 - f * egr)).tolist()
                        )
                    pTu_row.append(sL0)
                    f_row.append(f)
                pTu_grid.append(pTu_row)
                f_grid.append(f_row)
            nodes.append({"phi": phi, "pTu": pTu_grid, "f": f_grid})

        reconstructed_corners = []
        if require_plausible:
            filled_pTu, missing_baseline = self._fill_grid_gaps(
                nodes, "pTu", unfilled_baseline, pressures, temperatures,
            )
            filled_f, missing_slope = self._fill_grid_gaps(
                nodes, "f", unfilled_slope, pressures, temperatures,
            )
            reconstructed_corners = filled_pTu + filled_f
        else:
            missing_baseline = [
                (phi, p, T) for _, _, _, phi, p, T in unfilled_baseline
            ]
            missing_slope = [
                (phi, p, T) for _, _, _, phi, p, T in unfilled_slope
            ]

        def _format_corners(corners, limit=10):
            text = "; ".join(
                f"phi={c[0]:g} p={c[1]:.4g} T={c[2]:.4g}"
                for c in corners[:limit]
            )
            if len(corners) > limit:
                text += f" (+{len(corners) - limit} more)"
            return text

        if missing_baseline:
            raise ValueError(
                "fit_piecewise_phi(): no usable egr=0 baseline point "
                f"for {len(missing_baseline)} corner(s) - "
                + _format_corners(missing_baseline)
            )
        if missing_slope:
            raise ValueError(
                "fit_piecewise_phi(): fewer than 2 usable points with "
                f"egr <= {egr_max:g} to fit f from, for "
                f"{len(missing_slope)} corner(s) - "
                + _format_corners(missing_slope)
            )
        if negative_f:
            raise ValueError(
                "fit_piecewise_phi(): physically implausible fit - "
                "f < 0 (EGR should reduce, not increase, flame speed) "
                f"for {len(negative_f)} corner(s) - " + "; ".join(
                    f"phi={c[0]:g} p={c[1]:.4g} T={c[2]:.4g} "
                    f"f={c[3]:.4g}"
                    for c in negative_f[:10]
                )
            )

        residuals = np.array(relative_residuals)
        quality = {
            "n_corners": len(phi_nodes) * len(pressures) * len(temperatures),
            "n_points": int(len(residuals)),
            "relative_error_mean": (
                float(np.mean(residuals)) if residuals.size else 0.0
            ),
            "relative_error_std": (
                float(np.std(residuals)) if residuals.size else 0.0
            ),
            "relative_error_max": (
                float(np.max(np.abs(residuals))) if residuals.size else 0.0
            ),
            "reconstructed_corners": reconstructed_corners,
        }
        print(
            f"fit_piecewise_phi(): {len(phi_nodes)} phi node(s) x "
            f"{len(pressures)} pressure(s) x {len(temperatures)} "
            f"temperature(s) = {quality['n_corners']} corner(s), "
            f"{quality['n_points']} point(s) used for f\n"
            "  f relative error: "
            f"mean={100*quality['relative_error_mean']:.3g}% "
            f"std={100*quality['relative_error_std']:.3g}% "
            f"max={100*quality['relative_error_max']:.3g}%"
            + (
                f"\n  reconstructed {len(reconstructed_corners)} "
                "corner value(s) (not real solved data) - see "
                "quality['reconstructed_corners']"
                if reconstructed_corners else ""
            )
        )

        def _table2_dict(values):
            return {
                "type": "nonUniformTable2",
                "x": pressures,
                "y": temperatures,
                "outOfBounds": out_of_bounds,
                "values": values,
            }

        # Pure egr=0: a plain scalar 0, not a table full of zeros.
        return {
            f"phi{i}": {
                "Phi": node["phi"],
                "pTu": _table2_dict(node["pTu"]),
                "f": 0 if not has_egr_variation else _table2_dict(node["f"]),
            }
            for i, node in enumerate(nodes)
        }, quality
