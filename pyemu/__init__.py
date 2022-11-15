"""pyEMU: python modules for Environmental Model Uncertainty analyses.  These
modules are designed to work directly and seamlessly with PEST and PEST++ model
independent interface.  pyEMU can also be used to setup this interface.

Several forms of uncertainty analyses are support including FOSM-based
analyses (pyemu.Schur and pyemu.ErrVar), data worth analyses and
high-dimensional ensemble generation.
"""

from .la import LinearAnalysis
from .sc import Schur
from .ev import ErrVar
from .en import Ensemble, ParameterEnsemble, ObservationEnsemble
from .eds import EnDS

# from .mc import MonteCarlo
# from .inf import Influence
from .mat import Matrix, Jco, Cov
from .pst import Pst, pst_utils
from .utils import (
    helpers,
    gw_utils,
    optimization,
    geostats,
    pp_utils,
    os_utils,
    smp_utils,
    metrics,
)
from .plot import plot_utils
from .logger import Logger

from .prototypes import *

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
