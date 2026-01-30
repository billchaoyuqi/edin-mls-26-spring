# cuTile Tutorial

NVIDIA cuTile Python GPU programming tutorials.

## Environment Setup

### 1. Install Environment

```bash
# Run from project root
bash utils/setup-env.sh
```

This creates a conda environment named `mls` with:
- CUDA Toolkit (nvidia::cuda)
- CuPy, cuda-python, cuda-tile
- HuggingFace (transformers, datasets)
- PyTorch, Streamlit, etc.

### 2. Activate Environment and Configure hack.sh

```bash
conda activate mls
source utils/hack.sh  # Run from project root
```

## Running Tutorials

### Blackwell GPU (RTX 50 series, B100, B200)

Run directly:

```bash
python cutile-tutorial/1-vectoradd/vectoradd.py
```

### Non-Blackwell GPU (RTX 40/30 series, A100, H100, etc.)

**Recommended**: Source hack.sh then run (supports both cutile and hw1-asr):

```bash
# From project root
source utils/hack.sh

# Run cuTile tutorials
python cutile-tutorial/1-vectoradd/vectoradd.py
python cutile-tutorial/7-attention/attention.py

# Run hw1-asr
python hw1-asr/benchmark_student.py glm_asr_scratch
```

`hack.sh` automatically:
- Sets CUDA environment variables (CUDA_PATH, CUDA_HOME, LD_LIBRARY_PATH, CUPY_CUDA_PATH)
- Sets CuPy compilation include paths (CFLAGS, CXXFLAGS)
- Injects hack-hopper compatibility layer into PYTHONPATH
- Translates cuTile API to CuPy RawKernel implementation

## Tutorial Directories

| Directory | Content |
|-----------|---------|
| 0-environment | Environment check |
| 1-vectoradd | Vector addition (Hello World) |
| 2-execution-model | Execution model (1D/2D grid) |
| 3-data-model | Data types (FP16/FP32) |
| 4-transpose | Matrix transpose |
| 5-secret-notes | Advanced notes |
| 6-performance-tuning | Performance tuning |
| 7-attention | Attention mechanism |

## Supported GPUs

| GPU | Compute Capability | Support Method |
|-----|-------------------|----------------|
| RTX 5090/5080 | 12.x | Native support |
| B100/B200/GB200 | 10.x | Native support |
| RTX 4090/4080 | 8.9 | hack.sh compatibility layer |
| RTX 3090/3080 | 8.6 | hack.sh compatibility layer |
| A100/H100 | 8.0/9.0 | hack.sh compatibility layer |
