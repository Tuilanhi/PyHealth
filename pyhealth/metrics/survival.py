"""Discrimination metrics for right-censored survival predictions.

Survival models score each sample with a *risk* (higher means the event is
expected sooner), for example a Cox model's linear predictor. These metrics
compare risks with observed outcomes while respecting censoring.
"""

from typing import Any

import numpy as np

_TIED_RISK = {"half": 0.5, "zero": 0.0}
_BLOCK_ROWS = 1024


def _as_1d(values: Any, name: str) -> np.ndarray:
    """Converts a list, array, or tensor to a 1D float array.

    Args:
        values: Values of shape ``(N,)`` or ``(N, 1)``.
        name: Argument name, used in error messages.

    Returns:
        A float64 array of shape ``(N,)``.

    Raises:
        ValueError: If the values are not one per sample or not finite.
    """
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (N,) or (N, 1), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def concordance_index(
    risk: Any,
    duration: Any,
    event: Any,
    tied_risk: str = "half",
) -> float:
    """Harrell's concordance index (C-index) for right-censored data.

    A pair of samples ``(i, j)`` is comparable when ``i`` had the event and
    was observed for less time than ``j`` (``duration_i < duration_j``), or
    for the same time while ``j`` was censored. The pair is concordant when
    ``risk_i > risk_j``. The C-index is the fraction of comparable pairs that
    are concordant, with tied risks counted as ``0.5`` by default. These rules
    match ``lifelines.utils.concordance_index(duration, -risk, event)``.

    Reference:
        Harrell, F. E., Califf, R. M., Pryor, D. B., Lee, K. L., and
        Rosati, R. A. "Evaluating the Yield of Medical Tests." JAMA 247(18),
        1982.

    Args:
        risk: Predicted risk per sample, shape ``(N,)`` or ``(N, 1)``; higher
            means higher risk (for example ``y_prob`` from a Cox model).
        duration: Observed time per sample, shape ``(N,)`` or ``(N, 1)``.
        event: Event indicator per sample (1 observed, 0 censored).
        tied_risk: How a comparable pair with equal risks counts:
            ``"half"`` (0.5, the standard definition) or ``"zero"`` (0, as in
            the MAS authors' code).

    Returns:
        The C-index in ``[0, 1]``: 0.5 for random risks, 1 for a perfect
        ranking. NaN if no pair is comparable (for example, no events).

    Raises:
        ValueError: If the inputs have different lengths, are not finite, the
            events are not 0/1, or ``tied_risk`` is unknown.

    Examples:
        >>> concordance_index(risk=[3, 2, 1], duration=[1, 2, 3], event=[1, 1, 0])
        1.0
        >>> concordance_index(risk=[1, 2, 3], duration=[1, 2, 3], event=[1, 1, 0])
        0.0
    """
    if tied_risk not in _TIED_RISK:
        raise ValueError(
            f"tied_risk must be one of {list(_TIED_RISK)}, got {tied_risk!r}."
        )
    risk = _as_1d(risk, "risk")
    duration = _as_1d(duration, "duration")
    event = _as_1d(event, "event")
    if not (risk.size == duration.size == event.size):
        raise ValueError(
            f"risk, duration, and event lengths differ: {risk.size}, "
            f"{duration.size}, {event.size}."
        )
    if not np.all(np.isin(event, (0.0, 1.0))):
        raise ValueError("event must contain only 0 and 1.")

    tie_weight = _TIED_RISK[tied_risk]
    observed = event == 1.0
    concordant = 0.0
    comparable = 0
    # Rows are the samples with the event; compare them in blocks so memory
    # stays bounded for large N.
    rows = np.flatnonzero(observed)
    for begin in range(0, rows.size, _BLOCK_ROWS):
        block = rows[begin : begin + _BLOCK_ROWS]
        earlier = duration[block, None] < duration[None, :]
        same_time_censored = (duration[block, None] == duration[None, :]) & ~observed[
            None, :
        ]
        is_comparable = earlier | same_time_censored
        higher = risk[block, None] > risk[None, :]
        tied = risk[block, None] == risk[None, :]
        comparable += int(is_comparable.sum())
        concordant += float((is_comparable & higher).sum())
        concordant += tie_weight * float((is_comparable & tied).sum())
    if comparable == 0:
        return float("nan")
    return concordant / comparable
