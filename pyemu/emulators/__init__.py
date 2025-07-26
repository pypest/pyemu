from .transformers import (
    BaseTransformer,
    Log10Transformer,
    RowWiseMinMaxScaler,
    #StandardScalerTransformer,
    NormalScoreTransformer,
    TransformerPipeline,
    AutobotsAssemble
)
from .base import Emulator
from .dsi import DSI
#from .lpfa import LPFA
#from .gpr import GPR  


__all__ = [
    'Emulator', #base Emulator Class
    'DSI',  # DSI Emulator Class
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
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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
