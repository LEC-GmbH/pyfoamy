"""Shared types and tolerances of the laminar flame speed modules."""

import enum
from typing import Dict, TypedDict


class LFSPoint(TypedDict):
    """One solved (`sL=nan` if failed) laminar flame speed point."""
    p: float
    T: float
    phi: float
    egr: float
    sL: float
    width: float
    thickness: float
    maxT: float
    species: Dict[str, float]


# Proven refine-criteria/time-step defaults, not derived from theory.
_REFINE_CRITERIA = {"ratio": 2, "slope": 0.05, "curve": 0.1,
                    "prune": 0.001}
_TIME_STEPS = (1e-7, [2, 5, 5, 5, 5, 10, 10, 20])
_TRACE_THRESHOLD = 1e-6

# "regen": if cantera grew the domain past what we asked, retry from
# the geometric mean of the failing/succeeding bracket. Hitting
# cantera's 12-doubling cap means it never converged (sL=nan).
_REGEN_WIDTH_SHRINK = 0.5 ** 0.5
_MAX_INTERNAL_WIDTH_GROWTH = 2 ** 12
# Each failed solve doubles the domain; give up (sL=nan) after this.
_MAX_FAILED_SOLVES = 6

# Sanity floor ("did you mean 1e5, not 1"), not a flammability limit.
_MIN_PRESSURE = 1e4

# Gulder(p, Tu, phi, egr) = W * phi^eta * exp(-xi*(phi-_GULDER_PHI_PEAK)^2)
#   * (p/_GULDER_PREF)^beta * (Tu/_GULDER_TREF)^alpha * (1 - f*egr)
_GULDER_TREF = 300.0
_GULDER_PREF = 1.013e5
_GULDER_PHI_PEAK = 1.075
# Floors a non-propagating (sL=nan) point before the log-space fit
# (log(0) is -inf).
_GULDER_ZERO_FLOOR = 1e-6
# Floating-point-noise floor for the sign check; real coefficients are
# O(0.1) or larger.
_GULDER_SIGN_TOLERANCE = 1e-2
# Detects an underdetermined phi-shape (too few distinct phi to
# separate eta/xi) by refitting with a shifted peak; only annotates the
# sign-violation error.
_GULDER_PHI_PEAK_SHIFT_PROBE = 0.5
_GULDER_IDENTIFIABILITY_TOLERANCE = 1e-6

# Same floating-point-noise-floor reasoning as _GULDER_SIGN_TOLERANCE.
_PIECEWISE_PHI_SIGN_TOLERANCE = 1e-2

_AXES = ["p", "T", "phi", "egr"]
_FIXED_COLUMNS = _AXES + ["sL", "width", "thickness", "maxT"]

# validate()'s per-point diagnosis, one bit per check; a point can have
# several (e.g. MONOTONICITY | GLOBAL_OUTLIER).
_MONOTONICITY_TOLERANCE = 0.02
_SUB_FIT_MIN_POINTS = 3
_VALIDATE_EGR_MAX = 0.10
_TRIM_FRACTION = 0.01
_MAX_DISCARD_FRACTION = 0.2
_OUTLIER_THRESHOLD = 0.15

# A re-solve must use different numerics than the original or it
# reproduces the same wrong-branch result: start well below the width
# the flagged solve converged to and use more grid points.
_RESOLVE_WIDTH_SHRINK = 0.1
_RESOLVE_NUM_POINTS_SCALE = 1.4
_DEFAULT_NUM_POINTS = 40
_RESOLVE_NUM_POINTS = round(_DEFAULT_NUM_POINTS * _RESOLVE_NUM_POINTS_SCALE)


class PlausibilityStatus(enum.IntFlag):
    GOOD = 0
    NON_CONVERGED = enum.auto()
    MONOTONICITY = enum.auto()
    SUB_FIT = enum.auto()
    GLOBAL_OUTLIER = enum.auto()
    UNCHECKED = enum.auto()
    # Appended last to keep earlier bit values stable in old caches'
    # persisted `plausibility` arrays.
    RESOLVED = enum.auto()


# validate()'s four real "this point's sL is wrong" diagnoses -
# UNCHECKED/RESOLVED are deliberately excluded, see _is_trustworthy().
_BAD_FLAGS = (
    PlausibilityStatus.NON_CONVERGED | PlausibilityStatus.MONOTONICITY
    | PlausibilityStatus.SUB_FIT | PlausibilityStatus.GLOBAL_OUTLIER
)
# NON_CONVERGED is handled separately by fit_gulder() itself (floored
# into the fit as informative flammability-limit data, not excluded) -
# these three are the ones a still-bad point must be excluded for.
_HARD_BAD_FLAGS = (
    PlausibilityStatus.MONOTONICITY | PlausibilityStatus.SUB_FIT
    | PlausibilityStatus.GLOBAL_OUTLIER
)


def _is_trustworthy(status: int) -> bool:
    # GOOD; RESOLVED after a successful re-solve whatever it was first
    # flagged for (history is OR'd in, see _resolve_flagged); or
    # UNCHECKED (validate() could not run): no filtering rather than
    # losing data.
    return (
        status == int(PlausibilityStatus.GOOD)
        or bool(status & int(PlausibilityStatus.RESOLVED))
        or status == int(PlausibilityStatus.UNCHECKED)
    )
