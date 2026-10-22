"""Longitudinal liver survival samples for time-varying survival models.

Reproduces the sample construction of the Model for Allograft Survival (MAS):

    Gao, X., Cooper, M., Naghibzadeh, M., Azhie, A., Bhat, M., and
    Krishnan, R. G. "Predicting Long-Term Allograft Survival in Liver
    Transplant Recipients." Machine Learning for Healthcare, PMLR 252, 2024.
    https://proceedings.mlr.press/v252/gao24a.html

Each patient has repeated lab visits and one outcome. This module turns a
patient's visits into counting-process rows: visit ``k`` covers the interval
``(visit_k, visit_{k+1}]`` (the last visit covers ``(visit_K, end]``) and uses
the labs measured *at* visit ``k``, so a row never sees labs from a later
visit. The event flag is 1 only on the last row of a patient who had the event.

Input times follow the Mayo PBC ``pbcseq`` layout: days since enrollment, and a
status code of 0 (censored), 1 (transplant), or 2 (death). Output times are in
years.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
STATUS_CENSORED = 0
STATUS_TRANSPLANT = 1
STATUS_DEATH = 2
EVENT_DEFINITIONS = ("death", "death_or_transplant")


def forward_fill(features: Sequence[Sequence[float]]) -> np.ndarray:
    """Carries each feature's last observed value forward in time.

    Values are filled from earlier visits only. A feature that has not been
    observed yet stays NaN.

    Args:
        features: Visit-by-feature values with shape ``(K, F)``; NaN marks a
            missing value.

    Returns:
        A float64 array of shape ``(K, F)``.

    Examples:
        >>> forward_fill([[1.0, float("nan")], [float("nan"), 2.0]]).tolist()
        [[1.0, nan], [1.0, 2.0]]
    """
    filled = np.array(features, dtype=np.float64)
    if filled.ndim != 2:
        raise ValueError(f"features must have shape (K, F), got {filled.shape}.")
    for k in range(1, filled.shape[0]):
        missing = np.isnan(filled[k])
        filled[k, missing] = filled[k - 1, missing]
    return filled


def event_indicator(status: int, event: str = "death") -> int:
    """Maps a ``pbcseq`` status code to an event indicator.

    Args:
        status: 0 (censored), 1 (transplant), or 2 (death).
        event: ``"death"`` counts only deaths, so a transplant is censored at
            the transplant date. ``"death_or_transplant"`` counts both.

    Returns:
        1 if the status counts as an event, otherwise 0.

    Raises:
        ValueError: If ``status`` or ``event`` is not recognised.
    """
    if event not in EVENT_DEFINITIONS:
        raise ValueError(f"event must be one of {EVENT_DEFINITIONS}, got {event!r}.")
    if status not in (STATUS_CENSORED, STATUS_TRANSPLANT, STATUS_DEATH):
        raise ValueError(f"status must be 0, 1, or 2, got {status!r}.")
    if status == STATUS_DEATH:
        return 1
    if status == STATUS_TRANSPLANT and event == "death_or_transplant":
        return 1
    return 0


def build_counting_process_samples(
    patient_id: str,
    visit_days: Sequence[float],
    features: Sequence[Sequence[float]],
    futime: float,
    status: int,
    event: str = "death",
) -> list[dict[str, Any]]:
    """Builds one ``[start, stop, event]`` sample per visit of a patient.

    Args:
        patient_id: The patient identifier.
        visit_days: Visit times in days since enrollment, strictly increasing,
            starting at 0 or later and ending before ``futime``.
        features: Lab values per visit, shape ``(K, F)``. Missing values (NaN)
            are forward-filled from earlier visits; values never observed stay
            NaN for the feature processor to impute.
        futime: Days from enrollment to the event or censoring.
        status: 0 (censored), 1 (transplant), or 2 (death).
        event: Event definition passed to :func:`event_indicator`.

    Returns:
        A list of ``K`` sample dicts with keys ``patient_id``, ``visit_id``
        (``"{patient_id}_{k}"``), ``landmark_time`` (visit time in years),
        ``features`` (list of ``F`` floats), and ``survival_interval``
        (``[start, stop, event]`` in years).

    Raises:
        ValueError: If the visits are empty, non-finite, unsorted, negative,
            at or after ``futime``, or do not match ``features``.

    Examples:
        >>> rows = build_counting_process_samples(
        ...     "1", [0, 365.25], [[1.0], [2.0]], futime=730.5, status=2
        ... )
        >>> [row["survival_interval"] for row in rows]
        [[0.0, 1.0, 0], [1.0, 2.0, 1]]
    """
    days = np.asarray(visit_days, dtype=np.float64)
    if days.ndim != 1 or days.size == 0:
        raise ValueError(f"Patient {patient_id}: needs at least one visit.")
    if not math.isfinite(futime):
        raise ValueError(f"Patient {patient_id}: futime must be finite.")
    if not np.all(np.isfinite(days)):
        raise ValueError(f"Patient {patient_id}: visit days must be finite.")
    if days[0] < 0 or np.any(np.diff(days) <= 0):
        raise ValueError(
            f"Patient {patient_id}: visit days must be >= 0 and strictly increasing."
        )
    if days[-1] >= futime:
        raise ValueError(
            f"Patient {patient_id}: last visit (day {days[-1]}) is not before "
            f"futime (day {futime})."
        )
    filled = forward_fill(features)
    if filled.shape[0] != days.size:
        raise ValueError(
            f"Patient {patient_id}: {days.size} visits but {filled.shape[0]} "
            "feature rows."
        )
    final_event = event_indicator(status, event)
    stops = np.append(days[1:], futime)
    samples = []
    for k, (start, stop) in enumerate(zip(days, stops)):
        is_last = k == days.size - 1
        samples.append(
            {
                "patient_id": str(patient_id),
                "visit_id": f"{patient_id}_{k}",
                "landmark_time": float(start / DAYS_PER_YEAR),
                "features": filled[k].tolist(),
                "survival_interval": [
                    float(start / DAYS_PER_YEAR),
                    float(stop / DAYS_PER_YEAR),
                    final_event if is_last else 0,
                ],
            }
        )
    return samples


def counting_process_samples_from_frame(
    visits: pd.DataFrame,
    feature_names: Sequence[str],
    event: str = "death",
) -> list[dict[str, Any]]:
    """Builds counting-process samples for every patient in a visit table.

    Args:
        visits: One row per visit with the ``pbcseq`` columns ``id``,
            ``day``, ``futime``, and ``status``, plus one column per feature.
            ``futime`` and ``status`` must be constant within a patient.
        feature_names: Feature columns to use, in order.
        event: Event definition passed to :func:`event_indicator`.

    Returns:
        Samples from :func:`build_counting_process_samples`, ordered by
        patient id and visit day.

    Raises:
        ValueError: If a required column is missing, ``futime`` or
            ``status`` varies within a patient, or ``status`` is not an integer.
    """
    required = ["id", "day", "futime", "status", *feature_names]
    missing = [column for column in required if column not in visits.columns]
    if missing:
        raise ValueError(f"Visit table is missing columns {missing}.")
    samples: list[dict[str, Any]] = []
    ordered = visits.sort_values(["id", "day"], kind="stable")
    for patient_id, rows in ordered.groupby("id", sort=True):
        if rows["futime"].nunique() != 1 or rows["status"].nunique() != 1:
            raise ValueError(
                f"Patient {patient_id}: futime and status must be constant."
            )
        status = float(rows["status"].iloc[0])
        if not status.is_integer():
            raise ValueError(
                f"Patient {patient_id}: status must be 0, 1, or 2, got {status}."
            )
        samples.extend(
            build_counting_process_samples(
                patient_id=str(patient_id),
                visit_days=rows["day"].to_numpy(dtype=np.float64),
                features=rows[list(feature_names)].to_numpy(dtype=np.float64),
                futime=float(rows["futime"].iloc[0]),
                status=int(status),
                event=event,
            )
        )
    return samples
