"""
Triton Rotary Position Embeddings (RoPE) — Optimised
Optimisations applied (all relative to friend's original rope.py):
  [R1] compute_freqs_kernel: added @triton.autotune on BLOCK size
       (friend used triton.next_power_of_2 but no autotune — fixed block)
  [R2] apply_rope_kernel: NEW Triton kernel — friend's _apply_rope_single()
       was pure Torch (no Triton at all). Now dispatches to a fused Triton
       kernel on CUDA; Torch fallback kept for CPU.
  [R3] apply_rotary_pos_emb: dispatches to Triton kernel on CUDA,
       Torch path retained for CPU / partial-dim cases needing torch.cat
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None

def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1

MAX_ROPE_DIM = 256

# ── Triton Kernels ────────────────────────────────────────────────────────────

# [R1] @triton.autotune — friend called kernel with a fixed triton.next_power_of_2 block
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 32},  num_warps=2),
        triton.Config({"BLOCK": 64},  num_warps=2),
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=4),
    ],
    key=["half_dim"],
)
@triton.jit
def compute_freqs_kernel(
    positions_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    half_dim,
    stride_pos,
    stride_inv,
    stride_cos0,
    stride_cos1,
    stride_sin0,
    stride_sin1,
    BLOCK: tl.constexpr,
):
    """
    Compute cos and sin for rotary embeddings.
    Grid: (seq_len,)
    Each program handles one sequence position, all half_dim frequencies.
    """
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < half_dim

    pos      = tl.load(positions_ptr + pid * stride_pos)
    inv      = tl.load(inv_freq_ptr  + offs * stride_inv, mask=mask, other=0.0)
    freqs    = pos * inv
    cos_half = tl.cos(freqs)
    sin_half = tl.sin(freqs)

    # Store first half
    tl.store(cos_ptr + pid * stride_cos0 + offs * stride_cos1,            cos_half, mask=mask)
    tl.store(sin_ptr + pid * stride_sin0 + offs * stride_sin1,            sin_half, mask=mask)
    # Store duplicated second half (full rotary_dim coverage)
    tl.store(cos_ptr + pid * stride_cos0 + (offs + half_dim) * stride_cos1, cos_half, mask=mask)
    tl.store(sin_ptr + pid * stride_sin0 + (offs + half_dim) * stride_sin1, sin_half, mask=mask)


# [R2] NEW: Fused RoPE application kernel — replaces friend's pure-Torch _apply_rope_single
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32},  num_warps=2),
        triton.Config({"BLOCK_D": 64},  num_warps=2),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
    ],
    key=["half_dim"],
)
@triton.jit
def apply_rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    num_heads, seq_len, half_dim, head_dim,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_cs, stride_cd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_D: tl.constexpr,
):
    """
    Fused RoPE: out[b,h,s,d] = x1*cos - x2*sin  (first half)
                              = x2*cos + x1*sin  (second half)
    Grid: (batch * num_heads * seq_len,)
    """
    pid    = tl.program_id(0)
    seq_id = pid % seq_len
    bh_id  = pid // seq_len
    b_id   = bh_id // num_heads
    h_id   = bh_id % num_heads

    offs_d = tl.arange(0, BLOCK_D)
    mask1  = offs_d < half_dim

    # Load x1 and x2 (the two halves being rotated)
    x1 = tl.load(x_ptr + b_id * stride_xb + h_id * stride_xh + seq_id * stride_xs + offs_d * stride_xd,
                 mask=mask1, other=0.0)
    x2 = tl.load(x_ptr + b_id * stride_xb + h_id * stride_xh + seq_id * stride_xs + (offs_d + half_dim) * stride_xd,
                 mask=mask1, other=0.0)

    cos_v = tl.load(cos_ptr + seq_id * stride_cs + offs_d * stride_cd, mask=mask1, other=1.0)
    sin_v = tl.load(sin_ptr + seq_id * stride_cs + offs_d * stride_cd, mask=mask1, other=0.0)

    # Apply rotation
    out1 = x1 * cos_v - x2 * sin_v
    out2 = x2 * cos_v + x1 * sin_v

    tl.store(out_ptr + b_id * stride_ob + h_id * stride_oh + seq_id * stride_os + offs_d * stride_od,
             out1, mask=mask1)
    tl.store(out_ptr + b_id * stride_ob + h_id * stride_oh + seq_id * stride_os + (offs_d + half_dim) * stride_od,
             out2, mask=mask1)

# ── RoPE Classes ──────────────────────────────────────────────────────────────

class RotaryEmbedding:
    """Rotary Position Embedding using Triton."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ):
        self.dim                     = dim
        self.max_position_embeddings = max_position_embeddings
        self.base                    = base
        self.partial_rotary_factor   = partial_rotary_factor

        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq
        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos and sin using Triton kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        positions  = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache  = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)
        sin_cache  = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)

        if device.type == "cuda":
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            # [R1] autotune picks best BLOCK for half_dim
            compute_freqs_kernel[(seq_len,)](
                positions,
                self.inv_freq,
                cos_cache,
                sin_cache,
                seq_len,
                half_dim,
                positions.stride(0),
                self.inv_freq.stride(0),
                cos_cache.stride(0),
                cos_cache.stride(1),
                sin_cache.stride(0),
                sin_cache.stride(1),
            )
        else:
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs    = positions[:, None] * self.inv_freq[None, :]
            cos_half = torch.cos(freqs)
            sin_half = torch.sin(freqs)
            cos_cache[:, :half_dim]            = cos_half
            cos_cache[:, half_dim:half_dim*2]  = cos_half
            sin_cache[:, :half_dim]            = sin_half
            sin_cache[:, half_dim:half_dim*2]  = sin_half

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for given positions."""
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self._update_cache(self.max_seq_len_cached, device=x.device)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]
                sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

        return cos, sin

# ── RoPE Application ──────────────────────────────────────────────────────────

def _apply_rope_single_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """[R2] Apply RoPE to a single tensor (Q or K) using fused Triton kernel."""
    batch, num_heads, seq_len, _ = x.shape
    x_f   = x.to(torch.float32).contiguous()
    cos_f = cos[:seq_len].to(torch.float32).contiguous()
    sin_f = sin[:seq_len].to(torch.float32).contiguous()
    out   = torch.empty_like(x_f)

    grid = (batch * num_heads * seq_len,)
    apply_rope_kernel[grid](
        x_f, cos_f, sin_f, out,
        num_heads, seq_len, half_dim, head_dim,
        x_f.stride(0),   x_f.stride(1),   x_f.stride(2),   x_f.stride(3),
        cos_f.stride(0), cos_f.stride(1),
        out.stride(0),   out.stride(1),   out.stride(2),   out.stride(3),
    )

    # Passthrough for dimensions beyond rotary_dim
    if head_dim > half_dim * 2:
        x_pass = x_f[..., half_dim * 2:]
        out    = torch.cat([out[..., :half_dim * 2], x_pass], dim=-1)

    return out


def _apply_rope_single_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply RoPE to a single tensor (Q or K) using Torch — CPU fallback."""
    cos = cos[:x.shape[2]]
    sin = sin[:x.shape[2]]
    cos_exp = cos[None, None, :, :]
    sin_exp = sin[None, None, :, :]
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:half_dim * 2]
    x1_rot = x1 * cos_exp - x2 * sin_exp
    x2_rot = x2 * cos_exp + x1 * sin_exp
    if head_dim > half_dim * 2:
        return torch.cat([x1_rot, x2_rot, x[..., half_dim * 2:]], dim=-1)
    return torch.cat([x1_rot, x2_rot], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K.
    [R3] Dispatches to fused Triton kernel on CUDA; Torch fallback for CPU.
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _                 = k.shape

    if rotary_dim is None:
        rotary_dim = head_dim

    half_dim = rotary_dim // 2

    if cos.shape[1] > half_dim:
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]

    cos = cos.to(torch.float32).contiguous()
    sin = sin.to(torch.float32).contiguous()

    if q.is_cuda and half_dim <= MAX_ROPE_DIM:
        # [R3] Fused Triton kernel path
        q_out = _apply_rope_single_triton(q, cos, sin, half_dim, head_dim)
        k_out = _apply_rope_single_triton(k, cos, sin, half_dim, head_dim)
    else:
        # Torch fallback (CPU or very large head_dim)
        q_out = _apply_rope_single_torch(q, cos, sin, half_dim, head_dim)
        k_out = _apply_rope_single_torch(k, cos, sin, half_dim, head_dim)

    return q_out.to(q.dtype), k_out.to(k.dtype)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


if __name__ == "__main__":
    print("Testing Triton RoPE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads  = 4
    seq_len    = 16
    head_dim   = 64

    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    cos, sin = rope(q)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Q rotated shape: {q_rot.shape}")
    print(f"K rotated shape: {k_rot.shape}")

    print("\nTesting partial RoPE (50%):")
    rope_partial = RotaryEmbedding(dim=head_dim, partial_rotary_factor=0.5)
    cos_p, sin_p = rope_partial(q)
    q_rot_p, k_rot_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, head_dim // 2)
    print(f"Q rotated (partial) shape: {q_rot_p.shape}")

    print("\nTriton RoPE working!")
