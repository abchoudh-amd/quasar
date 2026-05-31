"""Shared lazy-matplotlib import for the coil and pic postprocessing modules.

Both ``quasar.coil.postprocess`` and ``quasar.pic.postprocess`` import matplotlib
lazily (it is an optional dependency) and want the same friendly install hint when
it is missing. Keep that logic here so the two stay in sync.
"""

from __future__ import annotations


def require_pyplot(agg: bool = False):
    """Return ``matplotlib.pyplot`` or raise a clear ImportError if matplotlib is
    not installed. With ``agg=True`` the non-interactive Agg backend is selected
    (for headless figure-to-file rendering) before pyplot is imported."""
    try:
        import matplotlib
        if agg:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting; install it via "
            "`pip install matplotlib`"
        ) from e
    return plt
