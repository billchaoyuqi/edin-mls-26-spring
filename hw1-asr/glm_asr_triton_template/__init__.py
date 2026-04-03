import os, sys
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)
from . import layers
layers.Linear.BACKEND = "auto"    # Triton for prefill (M>=64), cuBLAS GEMV for decode (M=1)
layers.MLP.FUSED = True           # Enable fused SwiGLU kernel
layers.MLP._gate_weight_t = None  # Clear stale cache so weights rebuild correctly
layers.MLP._up_weight_t   = None
layers.EncoderMLP.FUSED = True
layers.EncoderMLP._fc1_weight_t = None
from . import model
from . import rope
from . import conv
from . import weight_loader