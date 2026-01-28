#!/usr/bin/env bash
set -eo pipefail

# =========================
# Config
# =========================
ENV_NAME="cutile"
PYTHON_VERSION="3.11"
CUDA_TAG="cuda13x"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALL_DIR="${HOME}/miniconda3"

# =========================
# Sanity hints (non-fatal)
# =========================
echo ">>> Assumptions:"
echo "    - NVIDIA driver >= r580"
echo "    - CUDA Toolkit >= 13.1"
echo "    - Blackwell GPU (CC 10.x / 12.x)"
echo

# =========================
# Check / Install conda
# =========================
if command -v conda >/dev/null 2>&1; then
	echo ">>> conda found: $(conda --version)"
	eval "$(conda shell.bash hook)"
elif [ -x /opt/conda/bin/conda ]; then
	echo ">>> conda found at /opt/conda/bin/conda"
	eval "$(/opt/conda/bin/conda shell.bash hook)"
else
	echo ">>> conda not found. Installing Miniconda..."

	MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
	curl -fsSL "${MINICONDA_URL}" -o "${MINICONDA_INSTALLER}"
	bash "${MINICONDA_INSTALLER}" -b -p "${MINICONDA_INSTALL_DIR}"
	rm -f "${MINICONDA_INSTALLER}"

	# Activate conda for current session
	eval "$("${MINICONDA_INSTALL_DIR}/bin/conda" shell.bash hook)"

	# Initialize conda for future shells
	conda init bash
	echo ">>> Miniconda installed at ${MINICONDA_INSTALL_DIR}"
fi

# =========================
# Accept conda Terms of Service
# =========================
echo ">>> Accepting conda channel Terms of Service"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# =========================
# Create conda environment
# =========================
if conda env list | grep -q "^${ENV_NAME} "; then
	echo ">>> Reusing existing conda environment: ${ENV_NAME}"
else
	echo ">>> Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
	conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" --override-channels -c conda-forge
fi

conda activate "${ENV_NAME}"

# =========================
# Install CUDA Toolkit
# =========================
echo ">>> Installing CUDA Toolkit from nvidia channel"
conda install -y nvidia::cuda

# =========================
# Core CUDA Python stack
# =========================
echo ">>> Installing CUDA Python stack (CUDA 13)"

# CuPy for CUDA 13
pip install "cupy-${CUDA_TAG}"

# NVIDIA CUDA Python bindings (driver/runtime API)
pip install cuda-python

# cuTile Python
pip install cuda-tile

# =========================
# Optional but recommended
# =========================
echo ">>> Installing optional tooling"

# NVML access (driver introspection, useful for debugging)
pip install pynvml

# NumPy (used by almost all examples)
pip install numpy

# =========================
# Freeze snapshot
# =========================
echo ">>> Writing lock snapshot (requirements.lock)"
conda list --export >requirements.lock

# =========================
# Done
# =========================
echo
echo "✅ cuTile Python environment is ready."
echo
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Installed key packages:"
echo "  - nvidia::cuda (via conda)"
echo "  - cupy-${CUDA_TAG}"
echo "  - cuda-python"
echo "  - cuda-tile"
echo
