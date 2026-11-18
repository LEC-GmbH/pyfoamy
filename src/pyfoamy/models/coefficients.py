"""Places optional model-coefficient overrides in a bare model sub-dict."""

from typing import Optional


def coefficients_entry(model, coefficients: Optional[dict]) -> dict:
    """Returns `{model: coefficients}`, or `{}` if there are none.

    Every model-choice dict built here is read via
    `dictionary::typeDict()`/`optionalTypeDict()`, which check for a bare
    sub-dict named after the model before `<model>Coeffs` or flat
    siblings, so this one form works everywhere.

    Args:
        model: Model name (`str` or enum member).
        coefficients: Overrides; `None` or empty places none.
    """
    if not coefficients:
        return {}
    return {str(model): dict(coefficients)}
