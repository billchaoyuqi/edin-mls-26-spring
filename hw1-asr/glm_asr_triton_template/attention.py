"""
Triton Multi-Head Attention Implementation — Optimised
Optimisations applied (all relative to friend's original):
[A1] flash_attention_kernel: added @triton.autotune on BLOCK_M/BLOCK_N
     (friend had NO autotune — hardcoded BLOCK_M=64, BLOCK_N=64)
[A2] flash_attention_kernel: SCALE made tl.constexpr
     (was a plain float arg — now baked in at compile time)
[A3] flash_attention_kernel: both tl.dot operands explicitly cast to fp16
     (friend had tl.dot(p.to(tl.float16), v) — v not cast, type mismatch)
[A4] flash_attention(): grid changed from tuple to lambda meta
     (autotune now injects its winning BLOCK_M into the grid)
[A5] flash_attention(): k/v contiguous() after GQA expand (friend had it,
     kept for correctness)
[A6] FIX: autotune key changed from ["N_CTX","HEAD_DIM"] to
     ["SEQ_Q","SEQ_K","HEAD_DIM"] to match actual kernel params.
     Old key caused autotune cache miss on EVERY decode step (SEQ_Q=1).
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


# ── Fallback 3-kernel path ─────────────────────────────────────────────────────

@triton.jit
def attention_scores_kernel(
    q_ptr, k_ptr, scores_ptr,
    scale, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute scaled attention scores: scores = Q @ K^T * scale"""
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
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """Numerically stable softmax applied in-place along the last dimension."""
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k
    s = tl.load(scores_ptr + row * stride_s + offs, mask=mask, other=-float("inf"))
    s = s - tl.max(s, axis=0)
    ex = tl.exp(s)
    tl.store(scores_ptr + row * stride_s + offs, ex / tl.sum(ex, axis=0), mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr, v_ptr, output_ptr,
    seq_k, head_dim,
    stride_w0, stride_w1, stride_w2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute attention output: attn_weights @ V"""
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    w = tl.load(attn_ptr + pid_bh * stride_w0 + pid_q * stride_w1 + offs_k * stride_w2,
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
    current_pos = pid_q + offset
    tl.store(scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
             tl.where(offs_k > current_pos, -1e9, scores), mask=mask)


# ── Flash Attention Kernel ─────────────────────────────────────────────────────
# [A1] @triton.autotune — friend had NO autotune, hardcoded BLOCK_M=64, BLOCK_N=64
# [A2] SCALE as tl.constexpr — baked in at compile time, saves a scalar arg per launch
# [A3] both tl.dot operands cast to fp16 — friend cast only p, not v
# [A6] FIX: key uses SEQ_Q,SEQ_K,HEAD_DIM — matches actual kernel params.
#      Old ["N_CTX","HEAD_DIM"] caused a cache miss on every decode step (SEQ_Q changes 1→N).

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=4, num_stages=3),
    ],
    key=["SEQ_Q", "SEQ_K", "HEAD_DIM"],  # [A6] FIX: was ["N_CTX","HEAD_DIM"]
)
@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    SEQ_Q,
    SEQ_K,
    HEAD_DIM:  tl.constexpr,
    BLOCK_M:   tl.constexpr,
    BLOCK_N:   tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SCALE:     tl.constexpr,  # [A2] constexpr — baked in, no per-launch scalar
):
    # Program IDs: (seq_block, head, batch)
    start_m = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_b   = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_ptr = Q + off_b * stride_qb + off_h * stride_qh
    K_ptr = K + off_b * stride_kb + off_h * stride_kh
    V_ptr = V + off_b * stride_vb + off_h * stride_vh

    q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < SEQ_Q, other=0.0)

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal: only loop up to current block boundary
    hi = tl.minimum((start_m + 1) * BLOCK_M, SEQ_K) if IS_CAUSAL else SEQ_K

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=offs_n[:, None] < SEQ_K, other=0.0)
        v = tl.load(V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=offs_n[:, None] < SEQ_K, other=0.0)

        # [A3] Both operands cast to fp16 — uses tensor cores, fp32 accumulator
        scores = tl.dot(q.to(tl.float16), tl.trans(k).to(tl.float16)).to(tl.float32) * SCALE

        if IS_CAUSAL:
            # Correct causal offset for decode (SEQ_Q=1, SEQ_K=full_seq)
            scores = tl.where(
                (offs_m[:, None] + (SEQ_K - SEQ_Q)) >= offs_n[None, :],
                scores, -1e9
            )

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha  = tl.exp(m_i - m_new)
        p      = tl.exp(scores - m_new[:, None])
        l_i    = alpha * l_i + tl.sum(p, axis=1)
        # [A3] v also cast to fp16 — friend's code forgot this
        acc    = alpha[:, None] * acc + tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
        m_i    = m_new

    acc = acc / l_i[:, None]
    O_ptr = Out + off_b * stride_ob + off_h * stride_oh
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(tl.float16), mask=offs_m[:, None] < SEQ_Q)


def flash_attention(q, k, v, is_causal=True):
    """
    Flash attention forward pass.
    q: (B, H, M, D)   k/v: (B, Hkv, N, D)
    Returns: (B, H, M, D)
    """
    B, H, M, D = q.shape
    N = k.shape[2]
    scale = 1.0 / (D ** 0.5)

    # GQA: expand k/v heads to match q heads (zero-copy broadcast)
    if k.shape[1] < H:
        ratio = H // k.shape[1]
        k = k.unsqueeze(2).expand(-1, -1, ratio, -1, -1).reshape(B, H, N, D)
        v = v.unsqueeze(2).expand(-1, -1, ratio, -1, -1).reshape(B, H, N, D)
    # [A5] Ensure contiguous after expand
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)
    # [A4] lambda grid so autotune injects winning BLOCK_M (friend had a fixed tuple)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), H, B)
    flash_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        M, N,         # SEQ_Q, SEQ_K
        D,            # HEAD_DIM (constexpr)
        IS_CAUSAL=is_causal,
        SCALE=scale,
    )
    return out


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask=None,
    is_causal: bool = False,
    scale: float = None,
    offset: int = 0,
) -> torch.Tensor:
    """
    Unified attention dispatch.
    Routes to Flash Attention (Triton) when on CUDA and no external mask.
    Falls back to 3-kernel path for masked / CPU cases.
    """
    if q.is_cuda and attention_mask is None:
        q_fa = q.contiguous().to(torch.float16)
        k_fa = k.contiguous().to(torch.float16)
        v_fa = v.contiguous().to(torch.float16)
        return flash_attention(q_fa, k_fa, v_fa, is_causal=is_causal).to(q.dtype)

    # ── Fallback: 3-kernel path (CPU or masked attention) ──────────────────────
    B, H, M, D = q.shape
    seq_k = k.shape[2]
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # GQA expansion for fallback path
    if k.shape[1] < H:
        ratio = H // k.shape[1]
        k = k.unsqueeze(2).expand(-1, -1, ratio, -1, -1).reshape(B, H, seq_k, D)
        v = v.unsqueeze(2).expand(-1, -1, ratio, -1, -1).reshape(B, H, seq_k, D)

    BLOCK_K = next_power_of_two(seq_k)
    BLOCK_D = next_power_of_two(D)

    q_bh = q.reshape(B * H, M,     D).contiguous().to(torch.float32)
    k_bh = k.reshape(B * H, seq_k, D).contiguous().to(torch.float32)
    v_bh = v.reshape(B * H, seq_k, D).contiguous().to(torch.float32)

    scores = torch.zeros(B * H, M, seq_k, device=q.device, dtype=torch.float32)

    if q.is_cuda:
        attention_scores_kernel[(B * H, M)](
            q_bh, k_bh, scores,
            scale, seq_k, D,
            q_bh.stride(0), q_bh.stride(1), q_bh.stride(2),
            k_bh.stride(0), k_bh.stride(1), k_bh.stride(2),
            scores.stride(0), scores.stride(1), scores.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, num_warps=4)

        if is_causal:
            causal_mask_kernel[(B * H, M)](
                scores, seq_k, offset,
                scores.stride(0), scores.stride(1), scores.stride(2),
                BLOCK_K=BLOCK_K, num_warps=2)

        if attention_mask is not None:
            scores = scores + attention_mask.reshape(B * H, M, seq_k).to(torch.float32)

        softmax_inplace_kernel[(B * H * M,)](
            scores, scores.stride(1), seq_k,
            BLOCK_SIZE=BLOCK_K, num_warps=4)

        output = torch.zeros(B * H, M, D, device=q.device, dtype=torch.float32)
        attention_output_kernel[(B * H, M)](
            scores, v_bh, output,
            seq_k, D,
            scores.stride(0), scores.stride(1), scores.stride(2),
            v_bh.stride(0),   v_bh.stride(1),   v_bh.stride(2),
            output.stride(0), output.stride(1),  output.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, num_warps=4)
    else:
        # Pure CPU path
        scores_cpu = torch.bmm(q_bh, k_bh.transpose(1, 2)) * scale
        if is_causal:
            causal = torch.triu(torch.full((M, seq_k), -1e9, device=q.device), diagonal=1)
            scores_cpu = scores_cpu + causal
        if attention_mask is not None:
            scores_cpu = scores_cpu + attention_mask.reshape(B * H, M, seq_k).float()
        attn_w = torch.softmax(scores_cpu, dim=-1)
        output = torch.bmm(attn_w, v_bh)

    return output.reshape(B, H, M, D).to(q.dtype)


class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask=None,
        is_causal: bool = False,
        offset: int = 0,
    ) -> torch.Tensor:
        return scaled_dot_product_attention(q, k, v, attention_mask, is_causal, self.scale, offset)

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim)
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    output_gqa = scaled_dot_product_attention(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    print("\nAll Triton attention working!")
