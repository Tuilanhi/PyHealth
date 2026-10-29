pyhealth.metrics.survival
=========================

Discrimination metrics for right-censored survival predictions. Pass a risk score
per sample (higher means the event is expected sooner), the observed time, and
the event indicator.

.. code-block:: python

    from pyhealth.metrics import concordance_index

    c = concordance_index(risk, duration, event)  # Harrell's C; NaN if no pair is comparable

.. currentmodule:: pyhealth.metrics.survival

.. autofunction:: concordance_index
