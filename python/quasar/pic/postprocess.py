"""Post-processing helpers for PIC diagnostics."""

from __future__ import annotations

import numpy as np


def rms(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))
