#!/bin/bash
# =============================================================================
# cuTile Hopper Hack - Run cuTile tutorials on non-Blackwell GPUs
# =============================================================================
#
# This script allows you to run cuTile tutorials on older GPUs (Ada Lovelace,
# Ampere, etc.) by injecting a compatibility layer that translates cuTile
# API calls to CuPy RawKernel.
#
# Usage:
#   ./hack.sh <python_script.py>
#   ./hack.sh 1-vectoradd/vectoradd.py
#   ./hack.sh 7-attention/attention.py
#
# Or source it to enable the hack in current shell:
#   source hack.sh
#   python 1-vectoradd/vectoradd.py
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HACK_DIR="${SCRIPT_DIR}/hack-hopper"

# Set CUDA environment variables
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Inject the compatibility layer by prepending to PYTHONPATH
export PYTHONPATH="${HACK_DIR}:${PYTHONPATH}"

# Function to check GPU compatibility
check_gpu() {
    python3 -c "
import cupy as cp
cc = cp.cuda.Device().compute_capability
major = int(cc[:-1])
if major >= 10:
    print('[cuTile] Blackwell GPU detected (sm_' + cc + ') - using native cuTile')
    exit(1)
else:
    print('[cuTile Compat] Non-Blackwell GPU detected (sm_' + cc + ') - using compatibility layer')
    exit(0)
" 2>/dev/null
    return $?
}

# If script is sourced, just set up environment
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "[hack.sh] Environment configured for cuTile compatibility layer"
    echo "          PYTHONPATH includes: ${HACK_DIR}"
    echo "          CUDA_HOME: ${CUDA_HOME}"
    check_gpu
    return 0
fi

# If script is executed with arguments, run the Python script
if [[ $# -gt 0 ]]; then
    # Check GPU first
    check_gpu
    USE_COMPAT=$?

    if [[ $USE_COMPAT -eq 1 ]]; then
        # Blackwell GPU - use native cuTile without hack
        echo "[hack.sh] Using native cuTile (Blackwell GPU detected)"
        PYTHONPATH="" python3 "$@"
    else
        # Non-Blackwell GPU - use compatibility layer
        echo "[hack.sh] Using compatibility layer"
        echo ""
        python3 "$@"
    fi
else
    echo "cuTile Hopper Hack - Run cuTile on non-Blackwell GPUs"
    echo ""
    echo "Usage:"
    echo "  $0 <script.py>        Run a Python script with the compatibility layer"
    echo "  source $0             Set up environment for current shell"
    echo ""
    echo "Examples:"
    echo "  $0 1-vectoradd/vectoradd.py"
    echo "  $0 2-execution-model/sigmoid_1d.py"
    echo "  $0 7-attention/attention.py"
    echo ""
    echo "Supported tutorials:"
    echo "  - 1-vectoradd/vectoradd.py"
    echo "  - 2-execution-model/sigmoid_1d.py"
    echo "  - 2-execution-model/grid_2d.py"
    echo "  - 3-data-model/data_types.py"
    echo "  - 6-performance-tuning/autotune_benchmark.py"
    echo "  - 7-attention/attention.py"
fi
