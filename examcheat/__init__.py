import numpy as _np

if not hasattr(_np, "trapz"):
    _np.trapz = _np.trapezoid

__version__ = "1.0.0"
