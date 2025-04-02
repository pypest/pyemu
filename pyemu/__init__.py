"""pyEMU: python modules for Environmental Model Uncertainty analyses.  These
modules are designed to work directly and seamlessly with PEST and PEST++ model
independent interface.  pyEMU can also be used to setup this interface.

Several forms of uncertainty analyses are support including FOSM-based
analyses (pyemu.Schur and pyemu.ErrVar), data worth analyses and
high-dimensional ensemble generation.
"""

from .eds import EnDS
from .en import Ensemble, ObservationEnsemble, ParameterEnsemble
from .ev import ErrVar
from .la import LinearAnalysis
from .logger import Logger
# from .mc import MonteCarlo
# from .inf import Influence
from .mat import Cov, Jco, Matrix
from .plot import plot_utils
from .pst import Pst, pst_utils, Results
from .sc import Schur
from .utils import (geostats, gw_utils, helpers, metrics, optimization,
                    os_utils, pp_utils, smp_utils)

#from .prototypes import *
try:
    from .legacy import *
except (ModuleNotFoundError, ImportError) as e:
    import warnings
    warnings.warn("Failed to import legacy module. "
                  "May impact ability to access older methods."
                  f"{type(e).__name__} {e.msg}")

from ._version import get_versions

__version__ = get_versions()["version"]
__all__ = [
    "LinearAnalysis",
    "Schur",
    "ErrVar",
    "Ensemble",
    "ParameterEnsemble",
    "ObservationEnsemble",
    "Matrix",
    "Jco",
    "Cov",
    "Pst",
    "pst_utils",
    "helpers",
    "gw_utils",
    "geostats",
    "pp_utils",
    "os_utils",
    "smp_utils",
    "plot_utils",
    "metrics",
]
# del get_versions
