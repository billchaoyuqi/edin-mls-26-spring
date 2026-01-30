#!/bin/bash
# =============================================================================
# cuTile Hopper Hack - Run cuTile on non-Blackwell GPUs (Hopper, Ada, Ampere)
# =============================================================================
#
# This script sets up the environment to run cuTile tutorials on older GPUs
# by translating cuTile API calls to Triton kernels.
#
# First-time setup:
#   source hack.sh --install    # Install dependencies (triton, cupy)
#
# Usage:
#   source hack.sh              # Set up environment for current shell
#   python 1-vectoradd/vectoradd.py
#   python 2-execution-model/sigmoid_1d.py
#
# =============================================================================

# Get absolute path of this script (works even when sourced)
_HACK_SH_SOURCE="${BASH_SOURCE[0]:-$0}"
_HACK_SH_DIR="$(dirname "${_HACK_SH_SOURCE}")"
SCRIPT_DIR="$(cd "${_HACK_SH_DIR}" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
HACK_DIR="${SCRIPT_DIR}/hack-hopper"
unset _HACK_SH_SOURCE _HACK_SH_DIR

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# Install Dependencies
# =============================================================================
install_deps() {
    echo -e "${GREEN}[hack.sh] Installing dependencies...${NC}"

    # Check if pip is available
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}[hack.sh] pip not found. Please install pip first.${NC}"
        return 1
    fi

    # Install triton (OpenAI Triton for GPU kernels)
    echo -e "${YELLOW}[hack.sh] Installing triton...${NC}"
    pip install triton

    # Install cupy (CUDA array library)
    echo -e "${YELLOW}[hack.sh] Installing cupy...${NC}"
    # Try to detect CUDA version and install appropriate cupy
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        echo -e "${YELLOW}[hack.sh] Detected CUDA ${CUDA_VERSION}${NC}"

        case $CUDA_MAJOR in
            12)
                pip install cupy-cuda12x
                ;;
            11)
                pip install cupy-cuda11x
                ;;
            *)
                echo -e "${YELLOW}[hack.sh] Unknown CUDA version, trying generic cupy...${NC}"
                pip install cupy
                ;;
        esac
    else
        echo -e "${YELLOW}[hack.sh] nvcc not found, trying generic cupy...${NC}"
        pip install cupy
    fi

    # Install numpy
    pip install numpy

    # Install PyTorch (for ASR homework)
    echo -e "${YELLOW}[hack.sh] Installing PyTorch...${NC}"
    pip install torch --index-url https://download.pytorch.org/whl/cu121

    # Install transformers and related packages (for ASR homework)
    echo -e "${YELLOW}[hack.sh] Installing transformers and related packages...${NC}"
    pip install transformers safetensors soundfile librosa accelerate

    echo -e "${GREEN}[hack.sh] Dependencies installed successfully!${NC}"
    echo ""
    echo "Now run: source hack.sh"
}

# =============================================================================
# Check GPU and Dependencies
# =============================================================================
check_deps() {
    local missing=0

    # Check triton
    if ! python3 -c "import triton" 2>/dev/null; then
        echo -e "${RED}[hack.sh] triton not installed. Run: source hack.sh --install${NC}"
        missing=1
    fi

    # Check cupy
    if ! python3 -c "import cupy" 2>/dev/null; then
        echo -e "${RED}[hack.sh] cupy not installed. Run: source hack.sh --install${NC}"
        missing=1
    fi

    return $missing
}

check_gpu() {
    python3 -c "
import cupy as cp
cc = cp.cuda.Device().compute_capability
major = int(cc[:-1])
minor = int(cc[-1])
sm = f'sm_{major}{minor}'
if major >= 10:
    print(f'[hack.sh] Blackwell GPU detected ({sm}) - native cuTile should work')
elif major == 9:
    print(f'[hack.sh] Hopper GPU detected ({sm}) - using Triton backend')
elif major == 8:
    if minor >= 9:
        print(f'[hack.sh] Ada Lovelace GPU detected ({sm}) - using Triton backend')
    else:
        print(f'[hack.sh] Ampere GPU detected ({sm}) - using Triton backend')
else:
    print(f'[hack.sh] Older GPU detected ({sm}) - using Triton backend')
" 2>/dev/null
}

# =============================================================================
# Environment Setup
# =============================================================================
setup_env() {
    # Set CUDA environment variables
    export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

    # Inject the compatibility layer by prepending to PYTHONPATH
    export PYTHONPATH="${HACK_DIR}:${PYTHONPATH}"

    # Export project root
    export MLS_PROJECT_ROOT="${PROJECT_ROOT}"
}

# =============================================================================
# Main
# =============================================================================

# Handle --install flag
if [[ "$1" == "--install" ]] || [[ "$1" == "-i" ]]; then
    install_deps
    return 0 2>/dev/null || exit 0
fi

# Handle --help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "cuTile Hopper Hack - Run cuTile on non-Blackwell GPUs"
    echo ""
    echo "Usage:"
    echo "  source hack.sh --install    # First-time: install dependencies"
    echo "  source hack.sh              # Set up environment"
    echo ""
    echo "After sourcing, you can run cuTile tutorials directly:"
    echo "  python 1-vectoradd/vectoradd.py"
    echo "  python 2-execution-model/sigmoid_1d.py"
    echo "  python 2-execution-model/grid_2d.py"
    echo "  python 3-data-model/data_types.py"
    return 0 2>/dev/null || exit 0
fi

# If script is sourced, set up environment
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    setup_env

    echo -e "${GREEN}[hack.sh] Environment configured${NC}"
    echo "          PYTHONPATH: ${HACK_DIR}"
    echo "          CUDA_HOME:  ${CUDA_HOME}"
    echo ""

    # Check dependencies
    if check_deps; then
        check_gpu
        echo ""
        echo -e "${GREEN}Ready! Run your cuTile scripts with python.${NC}"
    else
        echo ""
        echo -e "${YELLOW}Run 'source hack.sh --install' to install missing dependencies.${NC}"
    fi

    return 0
fi

# If executed directly without sourcing
echo "Please source this script instead of executing it:"
echo ""
echo "  source hack.sh --install   # First time: install dependencies"
echo "  source hack.sh             # Set up environment"
