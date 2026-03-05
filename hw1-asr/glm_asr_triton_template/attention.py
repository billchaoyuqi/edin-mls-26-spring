"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
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


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # 偏移量定义
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    # Step 1: 加载当前查询位置的 Query 向量 (1, head_dim)
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    )

    # Step 2: 加载该 Head 下的所有 Keys (seq_k, head_dim)
    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    # Step 3: 计算点积分数并缩放 (Score = Q @ K^T * scale)
    # k * q[None, :] 会进行广播乘法，然后在 head_dim 维度求和
    scores = tl.sum(k * q[None, :], axis=1) * scale

    # Step 4: 存储分数
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k

    # Step 1: 加载一行分数，mask 掉越界部分
    s = tl.load(scores_ptr + row * stride_s + offs, mask=mask, other=-float("inf"))

    # Step 2: 减去最大值以保证数值稳定性 (防止 exp 溢出)
    s = s - tl.max(s, axis=0)

    # Step 3: 计算 exp 和归一化
    exp_s = tl.exp(s)
    denom = tl.sum(exp_s, axis=0)
    out = exp_s / denom

    # Step 4: 写回原处
    tl.store(scores_ptr + row * stride_s + offs, out, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_w0,
    stride_w1,
    stride_w2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    # Step 1: 加载该 Query 对应的所有 Attention 权重 (seq_k,)
    w = tl.load(
        attn_ptr
        + pid_bh * stride_w0
        + pid_q * stride_w1
        + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    )

    # Step 2: 加载该 Head 下的所有 Values (seq_k, head_dim)
    v = tl.load(
        v_ptr
        + pid_bh * stride_v0
        + offs_k[:, None] * stride_v1
        + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    # Step 3: 计算加权和 (Output = Weights @ V)
    # v * w[:, None] 将权重应用到每一行 Value 上，然后在 seq_k 维度求和
    out = tl.sum(v * w[:, None], axis=0)

    # Step 4: 存储输出
    tl.store(
        output_ptr
        + pid_bh * stride_o0
        + pid_q * stride_o1
        + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


@triton.jit
def flash_attn_fused_kernel(
        Q, K, V, Out,
        L, M,  # 用于 Online Softmax 的中间变量 (可选，如果只在 Kernel 内计算可不存)
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        Batch, Head, Seq_Q, Seq_K, Head_Dim,
        is_causal: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # 获取当前 Batch 和 Head 的索引
    bh_id = tl.program_id(0)
    # 获取当前 Q 块的索引
    m_id = tl.program_id(1)

    # 偏移量
    offs_m = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # 计算 Q 的指针位移并加载 (BLOCK_M, BLOCK_D)
    q_ptrs = Q + bh_id * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < Seq_Q) & (offs_d[None, :] < Head_Dim), other=0.0)

    # 初始化 Online Softmax 统计量
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 遍历 K, V 的分块 (BLOCK_N 步长)
    # FlashAttention 外层循环遍历 K, V 块，内层在 SRAM 处理
    for start_n in range(0, Seq_K, BLOCK_N):
        # 加载 K (BLOCK_N, BLOCK_D)
        k_ptrs = K + bh_id * stride_kb + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=((start_n + offs_n)[:, None] < Seq_K) & (offs_d[None, :] < Head_Dim), other=0.0)

        # 计算 qk^t: (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, tl.trans(k))
        qk *= sm_scale

        # 应用 Causal Mask
        if is_causal:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, -1e9)

        # --- Online Softmax 核心逻辑 ---
        m_ij = tl.max(qk, 1)
        m_next = tl.maximum(m_i, m_ij)

        p = tl.exp(qk - m_next[:, None])
        l_ij = tl.sum(p, 1)

        # 对旧累加值进行重缩放
        alpha = tl.exp(m_i - m_next)
        acc = acc * alpha[:, None]

        # 加载 V (BLOCK_N, BLOCK_D)
        v_ptrs = V + bh_id * stride_vb + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=((start_n + offs_n)[:, None] < Seq_K) & (offs_d[None, :] < Head_Dim), other=0.0)

        # 累加输出: p @ v
        acc += tl.dot(p.to(v.dtype), v)

        # 更新统计量
        l_i = l_i * alpha + l_ij
        m_i = m_next

    # 最终归一化
    acc = acc / l_i[:, None]

    # 写回输出
    out_ptrs = Out + bh_id * stride_ob + offs_m[:, None] * stride_oh + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < Seq_Q) & (offs_d[None, :] < Head_Dim))


# ============================================================================
# Attention Classes
# ============================================================================


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
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using Triton kernels.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
        # 确保张量是连续的且为 float32
        q = q.to(torch.float32).contiguous()
        k = k.to(torch.float32).contiguous()
        v = v.to(torch.float32).contiguous()

        # 输出张量
        output = torch.empty_like(q)

        # 如果有 attention_mask 且不是简单的 Causal，可能需要回退到基础实现
        # 但如果是 ASR 常见的对齐 Mask，你可以尝试将其融合进 Kernel
        if attention_mask is not None:
            # 暂时回退到通用实现或保持原逻辑
            # 这里为了演示核心 FlashAttn，我们主要针对 Causal 和无 Mask 情况优化
            use_triton = False
        else:
            use_triton = q.is_cuda

        if use_triton:
            # 根据 GPU 显存和 Head_Dim 自动调整分块
            # 这是第三阶段“调优”优化的体现
            BLOCK_M = 64 if head_dim <= 64 else 32
            BLOCK_N = 64 if head_dim <= 64 else 32

            # Grid: (Batch * Heads, Q 分块数)
            grid = (batch * num_heads, triton.cdiv(seq_q, BLOCK_M))

            flash_attn_fused_kernel[grid](
                q, k, v, output,
                None, None,  # L, M 统计量这里直接在 Kernel 内部处理
                float(scale),
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                batch, num_heads, seq_q, seq_k, head_dim,
                is_causal=is_causal,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=triton.next_power_of_2(head_dim),
                num_warps=8,
                num_stages=2
            )
            return output

        # --- Fallback 逻辑 (PyTorch 原生) ---
        scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
        if is_causal:
            mask = torch.triu(torch.ones((seq_q, seq_k), device=q.device), diagonal=1) * -1e9
            scores += mask
        if attention_mask is not None:
            scores += attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bnqk,bnkd->bnqd", attn_weights, v).to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nTriton Attention working!")
