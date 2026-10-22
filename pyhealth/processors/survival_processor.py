"""Processors for censored time-to-event (survival) data.

PyHealth's label processors cover binary, multiclass, multilabel, and
regression targets. Survival models need a censoring-aware label instead:
the time at risk plus whether the event was observed. This module adds:

- :class:`SurvivalIntervalProcessor` (``"survival_interval"``): a
  counting-process label ``[start, stop, event]`` for time-varying Cox models.
- :class:`SurvivalLabelProcessor` (``"survival"``): a ``[duration, event]``
  label for landmark samples and models such as DeepSurv.
- :class:`StandardizedTensorProcessor` (``"standardized_tensor"``): z-scores
  numeric features with statistics learned in ``fit()``, filling missing values
  with the fitted mean.

Survival labels are not one of ``BaseModel``'s built-in modes, so a model using
them has ``mode = None``: the ``Trainer`` then reports only the loss, and
survival metrics (for example :func:`pyhealth.metrics.concordance_index`) are
computed from ``Trainer.inference`` outputs.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor


def _as_float_list(value: Any, length: int, name: str) -> list[float]:
    """Converts a label to a list of floats of a fixed length.

    Args:
        value: A sequence or tensor holding the label.
        length: The required number of entries.
        name: The label layout, used in error messages.

    Returns:
        The label entries as Python floats.

    Raises:
        ValueError: If the label has the wrong length or a non-finite entry.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    values = [float(v) for v in value]
    if len(values) != length:
        raise ValueError(f"Expected {name} with {length} values, got {len(values)}.")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be finite, got {values}.")
    return values


def _check_event(event: float, name: str) -> None:
    """Checks that an event indicator is 0 (censored) or 1 (observed).

    Args:
        event: The event indicator.
        name: The label layout, used in error messages.

    Raises:
        ValueError: If the indicator is not 0 or 1.
    """
    if event not in (0.0, 1.0):
        raise ValueError(f"{name}: event must be 0 or 1, got {event}.")


@register_processor("survival_interval")
class SurvivalIntervalProcessor(FeatureProcessor):
    """Counting-process survival label ``[start, stop, event]``.

    Each sample is one at-risk interval ``(start, stop]`` of a patient.
    ``event`` is 1 if the event happened at ``stop`` and 0 otherwise, so only
    a patient's last interval can carry an event. This is the input format of
    time-varying Cox models such as ``lifelines.CoxTimeVaryingFitter``.

    Output:
        A float32 tensor of shape ``(3,)``; batches stack to ``(B, 3)``.

    Examples:
        >>> from pyhealth.processors import SurvivalIntervalProcessor
        >>> SurvivalIntervalProcessor().process([0.0, 0.5, 1])
        tensor([0.0000, 0.5000, 1.0000])
    """

    stores_tensor = True

    def process(self, value: Sequence[float]) -> torch.Tensor:
        """Validates one interval and converts it to a tensor.

        Args:
            value: ``[start, stop, event]`` with ``0 <= start < stop`` and
                ``event`` in ``{0, 1}``.

        Returns:
            A float32 tensor of shape ``(3,)``.

        Raises:
            ValueError: If the interval is malformed.
        """
        start, stop, event = _as_float_list(value, 3, "[start, stop, event]")
        if start < 0 or stop <= start:
            raise ValueError(
                f"Interval needs 0 <= start < stop, got start={start}, stop={stop}."
            )
        _check_event(event, "[start, stop, event]")
        return torch.tensor([start, stop, event], dtype=torch.float32)

    def size(self) -> int:
        """Returns the label width, 3."""
        return 3

    def is_token(self) -> bool:
        """Survival labels are continuous values, not token indices."""
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output is a 1D tensor of shape ``(3,)``."""
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def __repr__(self) -> str:
        return "SurvivalIntervalProcessor()"


@register_processor("survival")
class SurvivalLabelProcessor(FeatureProcessor):
    """Right-censored survival label ``[duration, event]``.

    ``duration`` is the time from the sample's reference point (for example a
    landmark visit) to the event or censoring, and ``event`` is 1 if the event
    was observed.

    Output:
        A float32 tensor of shape ``(2,)``; batches stack to ``(B, 2)``.

    Examples:
        >>> from pyhealth.processors import SurvivalLabelProcessor
        >>> SurvivalLabelProcessor().process([1.5, 0])
        tensor([1.5000, 0.0000])
    """

    stores_tensor = True

    def process(self, value: Sequence[float]) -> torch.Tensor:
        """Validates one label and converts it to a tensor.

        Args:
            value: ``[duration, event]`` with ``duration > 0`` and ``event``
                in ``{0, 1}``.

        Returns:
            A float32 tensor of shape ``(2,)``.

        Raises:
            ValueError: If the label is malformed.
        """
        duration, event = _as_float_list(value, 2, "[duration, event]")
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}.")
        _check_event(event, "[duration, event]")
        return torch.tensor([duration, event], dtype=torch.float32)

    def size(self) -> int:
        """Returns the label width, 2."""
        return 2

    def is_token(self) -> bool:
        """Survival labels are continuous values, not token indices."""
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output is a 1D tensor of shape ``(2,)``."""
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def __repr__(self) -> str:
        return "SurvivalLabelProcessor()"


@register_processor("standardized_tensor")
class StandardizedTensorProcessor(FeatureProcessor):
    """Numeric features z-scored with statistics learned in ``fit()``.

    ``fit()`` computes each feature's mean and standard deviation over the
    samples it sees, ignoring NaN. ``process()`` returns
    ``(x - mean) / std`` and replaces missing values (NaN) with 0, which is the
    fitted mean after standardization. A feature with zero or undefined spread
    is only centered.

    The processor learns statistics (``learns_statistics = True``), so fit it on
    the training split only, for example with
    ``set_task(task, split=PatientSplit(...))`` or by fitting it on the training
    samples and passing it to ``create_sample_dataset(input_processors=...)``.

    Input:
        A list of ``F`` floats (shape ``(F,)``) or a list of ``T`` such lists
        (shape ``(T, F)``). Every sample must use the same shape layout.

    Output:
        A float32 tensor with the input's shape.

    Examples:
        >>> from pyhealth.processors import StandardizedTensorProcessor
        >>> p = StandardizedTensorProcessor()
        >>> p.fit([{"x": [1.0, 10.0]}, {"x": [3.0, float("nan")]}], "x")
        >>> p.process([2.0, float("nan")])
        tensor([0., 0.])
    """

    learns_statistics = True
    stores_tensor = True

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self._n_dim: int | None = None

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Learns per-feature mean and standard deviation.

        Args:
            samples: Iterable of sample dictionaries.
            field: The feature key to read from each sample.

        Raises:
            ValueError: If no sample has the field, or the samples disagree on
                the number of features or dimensions.
        """
        rows = []
        for sample in samples:
            if field not in sample or sample[field] is None:
                continue
            array = np.asarray(sample[field], dtype=np.float64)
            if array.ndim not in (1, 2):
                raise ValueError(
                    f"{field}: expected shape (F,) or (T, F), got {array.shape}."
                )
            if self._n_dim is None:
                self._n_dim = array.ndim
            elif array.ndim != self._n_dim:
                raise ValueError(f"{field}: samples mix 1D and 2D feature values.")
            rows.append(array.reshape(-1, array.shape[-1]))
        if not rows:
            raise ValueError(f"No values found for field '{field}'.")
        widths = {row.shape[1] for row in rows}
        if len(widths) != 1:
            raise ValueError(
                f"{field}: samples have different feature counts {widths}."
            )
        stacked = np.concatenate(rows, axis=0)
        observed = ~np.isnan(stacked)
        counts = observed.sum(axis=0)
        filled = np.where(observed, stacked, 0.0)
        mean = np.divide(
            filled.sum(axis=0), counts, out=np.zeros(stacked.shape[1]), where=counts > 0
        )
        squared = np.where(observed, (stacked - mean) ** 2, 0.0)
        var = np.divide(
            squared.sum(axis=0),
            counts,
            out=np.zeros(stacked.shape[1]),
            where=counts > 0,
        )
        std = np.sqrt(var)
        self.mean = mean
        self.std = np.where(std > 0, std, 1.0)

    def process(self, value: Any) -> torch.Tensor:
        """Standardizes one sample's features.

        Args:
            value: Feature values with shape ``(F,)`` or ``(T, F)``.

        Returns:
            A float32 tensor with the same shape, NaN replaced by 0.

        Raises:
            RuntimeError: If called before ``fit()``.
            ValueError: If the feature count differs from the fitted one.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardizedTensorProcessor.fit must be called first.")
        array = np.asarray(value, dtype=np.float64)
        if array.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"Expected {self.mean.shape[0]} features, got {array.shape[-1]}."
            )
        standardized = np.nan_to_num((array - self.mean) / self.std, nan=0.0)
        return torch.tensor(standardized, dtype=torch.float32)

    def size(self) -> int | None:
        """Returns the number of features, or None before fitting."""
        return None if self.mean is None else int(self.mean.shape[0])

    def is_token(self) -> bool:
        """Standardized features are continuous values."""
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Returns ``(1,)`` for ``(F,)`` inputs and ``(2,)`` for ``(T, F)``.

        Raises:
            NotImplementedError: If called before ``fit()``.
        """
        if self._n_dim is None:
            raise NotImplementedError("Call fit() before dim().")
        return (self._n_dim,)

    def spatial(self) -> tuple[bool, ...]:
        """The time axis of ``(T, F)`` inputs is spatial; features are not.

        Raises:
            NotImplementedError: If called before ``fit()``.
        """
        if self._n_dim is None:
            raise NotImplementedError("Call fit() before spatial().")
        return (False,) if self._n_dim == 1 else (True, False)

    def __repr__(self) -> str:
        return f"StandardizedTensorProcessor(n_features={self.size()})"
