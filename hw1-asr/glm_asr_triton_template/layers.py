"""
Triton Neural Network Layers — Optimised
Optimisations applied (all relative to friend's original):
[L1]  gelu_kernel: inline tanh approx via tl.exp (cross-version safe)
[L2]  silu_kernel: inline sigmoid via tl.exp (cross-version safe)
[L3]  gelu / silu wrappers: BLOCK_SIZE 256 -> 512 (better memory throughput)
[L4]  linear_gelu_kernel: added @triton.autotune (was missing -- fixed tiles)
[L5]  linear_gelu_kernel: inline tanh approx via tl.exp
[L6]  swiglu_fused_kernel: added @triton.autotune (was missing -- fixed tiles)
[L7]  swiglu_fused_kernel: inline sigmoid via tl.exp (kept safe)
[L8]  MLP.__call__: grid tuple -> lambda meta (autotune now injects tiles)
[L9]  MLP.__call__: removed hard BLOCK_M/N/K kwargs (autotune decides)
[L10] FIX: Linear._forward_torch now converts weight to fp16 before M=1
      fast-path check -- previously weight stayed fp32 because
      _ensure_weight_prepared() was only called from _forward_triton,
      so the fp16 GEMV decode path never fired.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

import os as _os
_os.environ.setdefault("TRITON_CACHE_DIR", _os.path.expanduser("~/.triton/glm_asr_cache"))

# =============================================================================
# Helpers
# =============================================================================

def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None

def pad_to_multiple(size: int, multiple: int) -> int:
    """Pad size to be a multiple of the given value."""
    return ((size + multiple - 1) // multiple) * multiple

def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1

def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

# Largest BLOCK_M / BLOCK_N in the autotune config list below.
_LINEAR_MAX_BLOCK = 128

# =============================================================================
# Norm Kernels
# =============================================================================

@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    stride_x, stride_y,
    hidden_size, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm: x / RMS(x) * weight."""
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden_size
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + pid * stride_y + offs, x * tl.rsqrt(var + eps) * w, mask=mask)


@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    stride_x, stride_y,
    hidden_size, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias."""
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    x    = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden_size
    x_c  = x - mean
    var  = tl.sum(x_c * x_c, axis=0) / hidden_size
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + pid * stride_y + offs, x_c * tl.rsqrt(var + eps) * w + b, mask=mask)

# =============================================================================
# Pointwise Kernels
# =============================================================================

@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """[L1] GELU via inline tanh approx using tl.exp (cross-version safe)."""
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    tl.store(y_ptr + offs, x * 0.5 * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0)), mask=mask)


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """[L2] SiLU via inline sigmoid using tl.exp (cross-version safe)."""
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, x * (1.0 / (1.0 + tl.exp(-x))), mask=mask)

# =============================================================================
# Linear Kernel -- autotuned
# NOTE: grid at call site MUST be lambda meta: ... so autotune can inject tiles.
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32,  "BLOCK_K": 32}, num_warps=2, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_kernel_tf32(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """TF32-style matmul: output = A @ B."""
    pid_m  = tl.program_id(0)
    pid_n  = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
                    mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# =============================================================================
# Fused Linear+GELU -- autotuned
# [L4] Added @triton.autotune
# [L5] Inline tanh approx via tl.exp
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused Linear + GELU -- result stays in registers, no HBM round-trip."""
    pid_m  = tl.program_id(0)
    pid_n  = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
                    mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc * acc * acc)
    acc = acc * 0.5 * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0))  # [L5]
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# =============================================================================
# Fused SwiGLU -- autotuned
# [L6] Added @triton.autotune
# [L7] Inline sigmoid via tl.exp
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def swiglu_fused_kernel(
    a_ptr, gate_ptr, up_ptr, c_ptr,
    M, N, K,
    stride_am,  stride_ak,
    stride_gk,  stride_gn,
    stride_uk,  stride_un,
    stride_cm,  stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused SwiGLU: SiLU(x @ gate) * (x @ up). Two matmuls, one kernel."""
    pid_m  = tl.program_id(0)
    pid_n  = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a  = tl.load(a_ptr    + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
                     mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        gw = tl.load(gate_ptr + (k + offs_k[:, None]) * stride_gk + offs_n[None, :] * stride_gn,
                     mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        uw = tl.load(up_ptr   + (k + offs_k[:, None]) * stride_uk + offs_n[None, :] * stride_un,
                     mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        gate_acc += tl.dot(a, gw)
        up_acc   += tl.dot(a, uw)
    # [L7] inline sigmoid
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             gate_acc * (1.0 / (1.0 + tl.exp(-gate_acc))) * up_acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# =============================================================================
# Embedding Kernel
# =============================================================================

@triton.jit
def embedding_kernel(
    indices_ptr, weight_ptr, output_ptr,
    embedding_dim, stride_w0, stride_w1, stride_out0,
    BLOCK_SIZE: tl.constexpr,
):
    """Embedding lookup using gather."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    idx  = tl.load(indices_ptr + pid0)
    offs = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < embedding_dim
    w    = tl.load(weight_ptr + idx * stride_w0 + offs * stride_w1, mask=mask, other=0.0)
    tl.store(output_ptr + pid0 * stride_out0 + offs, w, mask=mask)

# =============================================================================
# Softmax Kernel
# =============================================================================

@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE: tl.constexpr):
    """Numerically stable softmax over last dimension."""
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x    = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float("inf"))
    x    = x - tl.max(x, axis=0)
    ex   = tl.exp(x)
    tl.store(y_ptr + row * stride_y + offs, ex / tl.sum(ex, axis=0), mask=mask)

# =============================================================================
# Attention Kernels (used by fallback path in attention.py)
# =============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr, k_ptr, scores_ptr,
    scale, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute attention scores: Q @ K^T * scale."""
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
                mask=offs_d < head_dim, other=0.0)
    k = tl.load(k_ptr + pid_bh * stride_k0 + offs_k[:, None] * stride_k1 + offs_d[None, :] * stride_k2,
                mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0)
    tl.store(scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
             tl.sum(k * q[None, :], axis=1) * scale, mask=offs_k < seq_k)


@triton.jit
def attention_output_kernel(
    weights_ptr, v_ptr, output_ptr,
    seq_k, head_dim,
    stride_w0, stride_w1, stride_w2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute attention output: weights @ V."""
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    w = tl.load(weights_ptr + pid_bh * stride_w0 + pid_q * stride_w1 + offs_k * stride_w2,
                mask=offs_k < seq_k, other=0.0)
    v = tl.load(v_ptr + pid_bh * stride_v0 + offs_k[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
                mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0)
    tl.store(output_ptr + pid_bh * stride_o0 + pid_q * stride_o1 + offs_d * stride_o2,
             tl.sum(v * w[:, None], axis=0), mask=offs_d < head_dim)


@triton.jit
def causal_mask_kernel(
    scores_ptr, seq_k, offset,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
):
    """Apply causal mask to attention scores."""
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    mask   = offs_k < seq_k
    scores = tl.load(scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
                     mask=mask, other=-1e9)
    tl.store(scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
             tl.where(offs_k > pid_q + offset, -1e9, scores), mask=mask)

# =============================================================================
# Layer Classes
# =============================================================================

class RMSNorm:
    """Root Mean Square Normalization using Triton with Torch fallback."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps         = eps
        self.weight      = torch.ones(hidden_size, dtype=torch.float32)
        self.use_triton  = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if self.use_triton and x.is_cuda:
            batch_size = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous().to(torch.float32)
            output  = torch.empty_like(x_flat)
            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)
            rmsnorm_kernel[(batch_size,)](
                x_flat, self.weight, output,
                x_flat.stride(0), output.stride(0),
                self.hidden_size, self.eps,
                BLOCK_SIZE=next_power_of_two(self.hidden_size), num_warps=4)
            return output.reshape(original_shape)
        x_float  = x.to(torch.float32)
        variance = torch.mean(x_float * x_float, dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        return (self.weight * x_normed).to(x.dtype)


class LayerNorm:
    """Layer Normalization using Triton with Torch fallback."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps         = eps
        self.weight      = torch.ones(hidden_size,  dtype=torch.float32)
        self.bias        = torch.zeros(hidden_size, dtype=torch.float32)
        self.use_triton  = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if self.use_triton and x.is_cuda:
            batch_size = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous().to(torch.float32)
            output  = torch.empty_like(x_flat)
            if self.weight.device != x.device: self.weight = self.weight.to(x.device)
            if self.bias.device   != x.device: self.bias   = self.bias.to(x.device)
            layernorm_kernel[(batch_size,)](
                x_flat, self.weight, self.bias, output,
                x_flat.stride(0), output.stride(0),
                self.hidden_size, self.eps,
                BLOCK_SIZE=next_power_of_two(self.hidden_size), num_warps=4)
            return output.reshape(original_shape)
        x_float  = x.to(torch.float32)
        mean     = torch.mean(x_float, dim=-1, keepdim=True)
        variance = torch.var(x_float,  dim=-1, keepdim=True, unbiased=False)
        x_normed = (x_float - mean) * torch.rsqrt(variance + self.eps)
        if self.weight.device != x.device: self.weight = self.weight.to(x.device)
        if self.bias.device   != x.device: self.bias   = self.bias.to(x.device)
        return (self.weight * x_normed + self.bias).to(x.dtype)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """[L3] GELU activation using Triton. BLOCK_SIZE 512 (was 256)."""
    original_shape = x.shape
    total  = int(np.prod(x.shape))
    block  = 512
    x_flat = x.reshape(-1).contiguous().to(torch.float32)
    output = torch.empty_like(x_flat)
    if x.is_cuda:
        gelu_kernel[(triton.cdiv(total, block),)](x_flat, output, total, BLOCK_SIZE=block, num_warps=4)
        return output[:total].reshape(original_shape).to(x.dtype)
    return torch.nn.functional.gelu(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    """[L3] SiLU activation using Triton. BLOCK_SIZE 512 (was 256)."""
    original_shape = x.shape
    total  = int(np.prod(x.shape))
    block  = 512
    x_flat = x.reshape(-1).contiguous().to(torch.float32)
    output = torch.empty_like(x_flat)
    if x.is_cuda:
        silu_kernel[(triton.cdiv(total, block),)](x_flat, output, total, BLOCK_SIZE=block, num_warps=4)
        return output[:total].reshape(original_shape).to(x.dtype)
    return torch.nn.functional.silu(x)


def get_activation(name: str):
    activations = {"gelu": gelu, "silu": silu}
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


class Linear:
    """Linear layer with switchable backend (torch or Triton)."""

    TILE_M  = 64
    TILE_N  = 64
    TILE_K  = 32
    BACKEND = "auto"   # "auto" -> Triton when M >= TILE_M on CUDA, else torch

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features  = in_features
        self.out_features = out_features
        self.has_bias     = bias
        self.weight       = torch.zeros((out_features, in_features), dtype=torch.float32)
        self.bias_param   = torch.zeros(out_features, dtype=torch.float32) if bias else None
        self._weight_t_padded = None
        self._K_padded        = None
        self._N_padded        = None

    def _ensure_weight_prepared(self, device=None):
        """Cache transposed padded weight for Triton kernel (also converts to fp16)."""
        if self._weight_t_padded is not None:
            return
        K = self.in_features
        N = self.out_features
        self._K_padded = pad_to_multiple(K, self.TILE_K)
        self._N_padded = pad_to_multiple(N, self.TILE_N)
        target_device  = device if device is not None else self.weight.device
        # Convert to fp16 once -- halves VRAM, enables cuBLAS tensor-core path
        self.weight    = self.weight.to(device=target_device, dtype=torch.float16)
        weight_t       = self.weight.t().contiguous()
        if self._K_padded > K or self._N_padded > N:
            weight_pad = torch.zeros(
                (self._K_padded, self._N_padded), dtype=torch.float32, device=target_device)
            weight_pad[:K, :N] = weight_t.float()
            self._weight_t_padded = weight_pad
        else:
            self._weight_t_padded = weight_t.float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if Linear.BACKEND in ("torch", "cublas"):
            return self._forward_torch(x)
        if Linear.BACKEND == "triton":
            return self._forward_triton(x)
        # "auto": Triton for large-M on CUDA, torch otherwise
        M = int(np.prod(x.shape[:-1]))
        if M >= self.TILE_M and x.is_cuda:
            return self._forward_triton(x)
        return self._forward_torch(x)

    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Torch matmul backend.
        [L10] FIX: convert weight to fp16 here so the M=1 fp16 GEMV fast path
        always fires during decode. Previously weight stayed fp32 because
        _ensure_weight_prepared() was only called from _forward_triton.
        """
        original_shape = x.shape
        batch_dims     = original_shape[:-1]
        M              = int(np.prod(batch_dims))

        # [L10] Ensure weight is fp16 (idempotent after first call)
        if self.weight.dtype != torch.float16:
            self.weight = self.weight.to(dtype=torch.float16)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)

        # M=1 decode step: fp16 GEMV via cuBLAS -- fastest path
        if M == 1:
            x_2d   = x.reshape(1, self.in_features).to(torch.float16)
            output = torch.nn.functional.linear(x_2d, self.weight)
            if self.has_bias and self.bias_param is not None:
                if self.bias_param.device != x.device:
                    self.bias_param = self.bias_param.to(x.device)
                output = output + self.bias_param.to(torch.float16)
            return output.reshape(*batch_dims, self.out_features).to(x.dtype)

        # Larger M: fp16 @ fp16 matmul with fp32 output
        x_2d   = x.reshape(M, self.in_features).to(torch.float16)
        output = (x_2d @ self.weight.t()).to(torch.float32)
        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
            output = output + self.bias_param.float()
        return output.reshape(*batch_dims, self.out_features)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Autotuned Triton matmul backend."""
        original_shape = x.shape
        batch_dims     = original_shape[:-1]
        M = int(np.prod(batch_dims))
        K = self.in_features
        N = self.out_features
        x_2d = x.reshape(M, K).to(torch.float32).contiguous()
        self._ensure_weight_prepared(x.device)
        M_padded  = pad_to_multiple(M, _LINEAR_MAX_BLOCK)
        x_padded  = torch.zeros((M_padded, self._K_padded), dtype=torch.float32, device=x.device)
        x_padded[:M, :K] = x_2d
        output    = torch.zeros((M_padded, self._N_padded), dtype=torch.float32, device=x.device)
        grid      = lambda meta: (
            triton.cdiv(M_padded, meta["BLOCK_M"]),
            triton.cdiv(self._N_padded, meta["BLOCK_N"]))
        linear_kernel_tf32[grid](
            x_padded, self._weight_t_padded, output,
            M_padded, self._N_padded, self._K_padded,
            x_padded.stride(0),           x_padded.stride(1),
            self._weight_t_padded.stride(0), self._weight_t_padded.stride(1),
            output.stride(0),             output.stride(1))
        output = output[:M, :N]
        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
            output = output + self.bias_param
        return output.reshape(*batch_dims, self.out_features)


class Embedding:
    """Embedding layer using Triton."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.weight         = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float32)

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        original_shape = input_ids.shape
        batch_size     = int(np.prod(original_shape))
        if self.weight.device != input_ids.device:
            self.weight = self.weight.to(input_ids.device)
        if not input_ids.is_cuda:
            flat   = input_ids.reshape(-1).to(torch.int64)
            output = self.weight.index_select(0, flat)
            return output.reshape(*original_shape, self.embedding_dim)
        indices_flat = input_ids.reshape(-1).to(torch.int32).contiguous()
        output = torch.empty((batch_size, self.embedding_dim), dtype=torch.float32,
                             device=indices_flat.device)
        block = 256
        embedding_kernel[(batch_size, triton.cdiv(self.embedding_dim, block))](
            indices_flat, self.weight, output,
            self.embedding_dim,
            self.weight.stride(0), self.weight.stride(1), output.stride(0),
            BLOCK_SIZE=block, num_warps=2)
        return output.reshape(*original_shape, self.embedding_dim)


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Softmax using Triton kernel."""
    if axis != -1 and axis != len(x.shape) - 1:
        x = torch.movedim(x, axis, -1)
    original_shape = x.shape
    batch_size     = int(np.prod(x.shape[:-1]))
    seq_len        = x.shape[-1]
    x_flat = x.reshape(batch_size, seq_len).to(torch.float32).contiguous()
    output = torch.empty_like(x_flat)
    if x.is_cuda:
        softmax_kernel[(batch_size,)](
            x_flat, output,
            x_flat.stride(0), output.stride(0),
            seq_len, BLOCK_SIZE=next_power_of_two(seq_len), num_warps=4)
        result = output.reshape(original_shape)
    else:
        result = torch.softmax(x, dim=-1)
    if axis != -1 and axis != len(original_shape) - 1:
        result = torch.movedim(result, -1, axis)
    return result


class MLP:
    """MLP with SwiGLU gating using Triton."""

    FUSED  = True
    TILE_M = 64
    TILE_N = 64
    TILE_K = 32

    def __init__(
        self,
        hidden_size:      int,
        intermediate_size: int,
        activation: str  = "silu",
        bias: bool       = False,
        use_gating: bool = True,
    ):
        self.use_gating        = use_gating
        self.act_fn            = get_activation(activation)
        self.hidden_size       = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled      = bias
        if use_gating:
            self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj   = Linear(hidden_size, intermediate_size, bias=bias)
        else:
            self.up_proj   = Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj     = Linear(intermediate_size, hidden_size, bias=bias)
        self._gate_weight_t = None
        self._up_weight_t   = None

    def _prepare_fused_weights(self, device):
        """Pre-transpose gate/up weights for fused SwiGLU kernel."""
        if self._gate_weight_t is not None and self._gate_weight_t.device == device:
            return
        K     = self.hidden_size
        N     = self.intermediate_size
        K_pad = pad_to_multiple(K, self.TILE_K)
        N_pad = pad_to_multiple(N, self.TILE_N)
        for attr, proj in [("_gate_weight_t", self.gate_proj), ("_up_weight_t", self.up_proj)]:
            proj.weight = proj.weight.to(device=device, dtype=torch.float16)
            w_t  = proj.weight.float().t().contiguous()
            w_pad = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=device)
            w_pad[:K, :N] = w_t
            setattr(self, attr, w_pad)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_gating:
            return self.down_proj(self.act_fn(self.up_proj(x)))
        if MLP.FUSED and x.is_cuda:
            orig_shape = x.shape
            M  = int(np.prod(x.shape[:-1]))
            K  = self.hidden_size
            N  = self.intermediate_size
            self._prepare_fused_weights(x.device)
            x_2d   = x.reshape(M, K).to(torch.float32).contiguous()
            M_pad  = pad_to_multiple(M, self.TILE_M)
            K_pad  = pad_to_multiple(K, self.TILE_K)
            N_pad  = pad_to_multiple(N, self.TILE_N)
            if M_pad > M or K_pad > K:
                x_padded = torch.zeros((M_pad, K_pad), dtype=torch.float32, device=x.device)
                x_padded[:M, :K] = x_2d
            else:
                x_padded = x_2d
            output = torch.zeros((M_pad, N_pad), dtype=torch.float32, device=x.device)
            # [L8] lambda grid -- autotune injects winning BLOCK_M / BLOCK_N
            # [L9] no hard BLOCK_M/N/K kwargs -- autotune decides
            grid = lambda meta: (
                triton.cdiv(M_pad, meta["BLOCK_M"]),
                triton.cdiv(N_pad, meta["BLOCK_N"]))
            swiglu_fused_kernel[grid](
                x_padded,
                self._gate_weight_t,
                self._up_weight_t,
                output,
                M_pad, N_pad, K_pad,
                x_padded.stride(0),          x_padded.stride(1),
                self._gate_weight_t.stride(0), self._gate_weight_t.stride(1),
                self._up_weight_t.stride(0),   self._up_weight_t.stride(1),
                output.stride(0),            output.stride(1))
            output = output[:M, :N].reshape(*orig_shape[:-1], self.intermediate_size)
            return self.down_proj(output)
        gate_out = self.act_fn(self.gate_proj(x))
        up_out   = self.up_proj(x)
        return self.down_proj(gate_out * up_out)



class EncoderMLP:
    """Fused Linear+GELU+Linear for audio encoder layers."""

    FUSED = True
    TILE_M = 64
    TILE_N = 64
    TILE_K = 32

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = True):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.fc1 = Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=bias)
        self._fc1_weight_t = None

    def _prepare_fused_weights(self, device):
        """Pre-transpose fc1 weight for fused linear+gelu kernel."""
        if self._fc1_weight_t is not None and self._fc1_weight_t.device == device:
            return
        K = self.hidden_size
        N = self.intermediate_size
        K_pad = pad_to_multiple(K, self.TILE_K)
        N_pad = pad_to_multiple(N, self.TILE_N)
        self.fc1.weight = self.fc1.weight.to(device=device, dtype=torch.float16)
        w_t = self.fc1.weight.float().t().contiguous()
        w_pad = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=device)
        w_pad[:K, :N] = w_t
        self._fc1_weight_t = w_pad

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if EncoderMLP.FUSED and x.is_cuda:
            orig_shape = x.shape
            M = int(np.prod(x.shape[:-1]))
            K = self.hidden_size
            N = self.intermediate_size
            self._prepare_fused_weights(x.device)
            x_2d = x.reshape(M, K).to(torch.float32).contiguous()
            M_pad = pad_to_multiple(M, self.TILE_M)
            K_pad = pad_to_multiple(K, self.TILE_K)
            N_pad = pad_to_multiple(N, self.TILE_N)
            if M_pad > M or K_pad > K:
                x_padded = torch.zeros((M_pad, K_pad), dtype=torch.float32, device=x.device)
                x_padded[:M, :K] = x_2d
            else:
                x_padded = x_2d
            output = torch.zeros((M_pad, N_pad), dtype=torch.float32, device=x.device)
            grid = lambda meta: (
                triton.cdiv(M_pad, meta["BLOCK_M"]),
                triton.cdiv(N_pad, meta["BLOCK_N"]))
            linear_gelu_kernel[grid](
                x_padded, self._fc1_weight_t, output,
                M_pad, N_pad, K_pad,
                x_padded.stride(0), x_padded.stride(1),
                self._fc1_weight_t.stride(0), self._fc1_weight_t.stride(1),
                output.stride(0), output.stride(1))
            output = output[:M, :N].reshape(*orig_shape[:-1], self.intermediate_size)
            return self.fc2(output)
        # fallback
        return self.fc2(gelu(self.fc1(x)))




class FusedQKVLinear:
    """Fuses q_proj + k_proj + v_proj into one GEMV at decode time."""
    
    def __init__(self, q_proj, k_proj, v_proj):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self._fused_weight = None

    def _prepare(self, device):
        if self._fused_weight is not None:
            return
        # Concatenate [W_q; W_k; W_v] along output dim → (q_out+k_out+v_out, hidden)
        wq = self.q_proj.weight.to(device=device, dtype=torch.float16)
        wk = self.k_proj.weight.to(device=device, dtype=torch.float16)
        wv = self.v_proj.weight.to(device=device, dtype=torch.float16)
        self._fused_weight = torch.cat([wq, wk, wv], dim=0)  # (2048+512+512, 2048)
        self._q_size = wq.shape[0]
        self._k_size = wk.shape[0]
        self._v_size = wv.shape[0]

    def __call__(self, x):
        self._prepare(x.device)
        # One single GEMV instead of three
        out = torch.nn.functional.linear(
            x.reshape(1, -1).to(torch.float16),
            self._fused_weight
        ).to(x.dtype).reshape(*x.shape[:-1], -1)
        q = out[..., :self._q_size]
        k = out[..., self._q_size:self._q_size + self._k_size]
        v = out[..., self._q_size + self._k_size:]
        return q, k, v


if __name__ == "__main__":
    print("Testing Triton Layers...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== RMSNorm ===")
    norm = RMSNorm(256)
    x    = torch.randn(2, 16, 256, device=device, dtype=torch.float32)
    y    = norm(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== Linear ===")
    linear = Linear(256, 512)
    y = linear(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== MLP ===")
    mlp = MLP(256, 512, activation="silu", use_gating=True)
    y   = mlp(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\nAll Triton layers working!")
