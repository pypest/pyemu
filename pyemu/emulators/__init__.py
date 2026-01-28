from .transformers import (
    BaseTransformer,
    Log10Transformer,
    RowWiseMinMaxScaler,
    #StandardScalerTransformer,
    NormalScoreTransformer,
    TransformerPipeline,
    AutobotsAssemble
)
import importlib.util
from .base import Emulator
from .dsi import DSI
#from .lpfa import LPFA
#from .gpr import GPR  


__all__ = [
    'Emulator', #base Emulator Class
    'DSI',  # DSI Emulator Class
#    'DSIAE',  # DSI Autoencoder Emulator Class
    'LPFA',
##    'GPR',  # GPR Emulator Class
    'BaseTransformer',
    'Log10Transformer',
    'RowWiseMinMaxScaler',
#    'StandardScalerTransformer',
    'NormalScoreTransformer',
    'TransformerPipeline',
    'AutobotsAssemble'
]

# Check sklearn availability
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

try:
    import tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


try:
    import tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


try:
    import tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


# Conditional imports
if HAS_SKLEARN:
    from .lpfa import LPFA
    from .gpr import GPR
    from .transformers import StandardScalerTransformer
    __all__.extend(['LPFA', 'GPR','StandardScalerTransformer'])
else:
    # Create placeholder classes that raise informative errors
    class LPFA:
        def __init__(self, *args, **kwargs):
            raise ImportError("LPFA emulator requires scikit-learn. Install with: pip install scikit-learn")
    
    class GPR:
        def __init__(self, *args, **kwargs):
            raise ImportError("GPR emulator requires scikit-learn. Install with: pip install scikit-learn")
    
    class StandardScalerTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("StandardScalerTransformer requires scikit-learn. Install with: pip install scikit-learn")

if HAS_TENSORFLOW and HAS_SKLEARN:
    from .dsiae import DSIAE
    __all__.append('DSIAE')
else:
    class DSIAE:
        def __init__(self, *args, **kwargs):
            raise ImportError("DSIAE emulator requires TensorFlow. Install with: pip install tensorflow")