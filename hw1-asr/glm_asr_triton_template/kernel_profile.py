"""
profile.py — Per-operator profiling for GLM-ASR Triton implementation.

Produces data for report Sections 3.3, 4.1, 5.1, 5.2, 6.3.

Usage (from inside glm_asr_triton_template/):
    python profile.py                     # full report  (§3.3 + §4.1)
    python profile.py --block-sweep       # §5.1: tile size sweep
    python profile.py --fusion-compare    # §5.2: fused vs unfused SwiGLU
    python profile.py --compare-example   # §6.3: vs example baseline
    python profile.py --all               # everything at once

Run on cluster:
    srun -p Teaching -w saxa --gres gpu:1 --pty bash
    cd hw1-asr/glm_asr_triton_template
    python profile.py --all 2>&1 | tee profile_results.txt
"""

import sys
import os
import argparse
import importlib
from typing import Dict, List, Tuple

import numpy as np
import torch
import triton
import triton.language as tl
import triton.testing

# ── Import our kernels ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layers    import (RMSNorm, LayerNorm, Linear, Embedding, MLP,
                       gelu, silu, softmax,
                       next_power_of_two, pad_to_multiple)
from attention import scaled_dot_product_attention
from rope      import RotaryEmbedding, apply_rotary_pos_emb

# ── GPU / hardware info ──────────────────────────────────────────────────────
DEVICE           = torch.device("cuda")
GPU_NAME         = ""          # filled in main()
GPU_TFLOPS_FP32  = None
GPU_BW_GBs       = None


def get_gpu_specs():
    """Estimate peak FP32 TFLOPS and memory bandwidth from device properties."""
    global GPU_NAME
    GPU_NAME = torch.cuda.get_device_name(0)
    props    = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    clock = getattr(props, "clock_rate", None)
    if clock is None:
        known_clocks = {"A100": 1410, "H100": 1830, "A40": 1740, "A6000": 1800,
                    "V100": 1380, "3090": 1700, "4090": 2520, "A10": 1695}
        clock_mhz = next((v for k, v in known_clocks.items() if k in GPU_NAME), 1500)
        clock = clock_mhz * 1e6
    else:
        clock = clock * 1e3  # kHz → Hz

    # ~128 FP32 CUDA cores per SM (Ampere / Hopper); 2 FLOPs per core per cycle
    tflops = sm_count * 128 * 2 * clock / 1e12

    known_bw = {
        "A100":     2000, "H100": 3350, "H200": 4800,
        "A6000":     768, "RTX 3090": 936, "RTX 4090": 1008,
        "V100":      900, "GTX 1080": 320,
    }
    mem_bw = next(
        (bw for name, bw in known_bw.items() if name in GPU_NAME),
        800,    # conservative fallback
    )
    return round(tflops, 1), mem_bw


# ── Timing helpers ───────────────────────────────────────────────────────────
def bench_ms(fn, warmup=25, rep=100):
    """Return median execution time in ms using triton.testing.do_bench."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


# ── Arithmetic intensity helpers ─────────────────────────────────────────────
BYTES_F32 = 4

def ai_rmsnorm(N, H):
    flops  = N * (2 * H + 1 + H)
    bytes_ = BYTES_F32 * (N * H + H + N * H)
    return flops / bytes_

def ai_layernorm(N, H):
    flops  = N * (4 * H + 2 + H)
    bytes_ = BYTES_F32 * (N * H + 2 * H + N * H)
    return flops / bytes_

def ai_gelu(N):
    return (N * 8) / (BYTES_F32 * 2 * N)

def ai_silu(N):
    return (N * 4) / (BYTES_F32 * 2 * N)

def ai_softmax(N, S):
    flops  = N * (S * 4)
    bytes_ = BYTES_F32 * 2 * N * S
    return flops / bytes_

def ai_linear(M, N, K):
    flops  = 2 * M * N * K
    bytes_ = BYTES_F32 * (M * K + K * N + M * N)
    return flops / bytes_

def ai_attention_scores(B, H, Sq, Sk, D):
    flops  = B * H * Sq * (2 * D * Sk)
    bytes_ = BYTES_F32 * B * H * (Sq * D + Sk * D + Sq * Sk)
    return flops / bytes_


# ============================================================================
# SECTION 3.3 + 4.1  —  Per-kernel latency and arithmetic intensity
# ============================================================================

def benchmark_kernels():
    """Benchmark all key kernels at GLM-ASR realistic dimensions."""
    print("\n" + "=" * 72)
    print("SECTION 3.3 + 4.1 — Per-Kernel Timing and Arithmetic Intensity")
    print("=" * 72)
    print(f"GPU             : {GPU_NAME}")
    print(f"Peak FP32       : {GPU_TFLOPS_FP32} TFLOPS (estimated)")
    print(f"Memory BW       : {GPU_BW_GBs} GB/s (estimated)")

    ridge = GPU_TFLOPS_FP32 * 1e12 / (GPU_BW_GBs * 1e9)
    print(f"Ridge point     : {ridge:.0f} FLOPs/Byte")
    print("  Kernels with AI < ridge → memory-bound")
    print("  Kernels with AI > ridge → compute-bound\n")

    rows = []

    # 1. rmsnorm_kernel — Text Decoder (56 rows, hidden=3584)
    B, H = 56, 3584
    x    = torch.randn(B, H, device=DEVICE)
    norm = RMSNorm(H); norm.weight = norm.weight.to(DEVICE)
    ms   = bench_ms(lambda: norm(x))
    rows.append(("rmsnorm_kernel",               ms, ai_rmsnorm(B, H),      f"B={B},H={H}"))

    # 2. layernorm_kernel — Audio Encoder (512 rows, hidden=1280)
    B, H = 512, 1280
    x    = torch.randn(B, H, device=DEVICE)
    ln   = LayerNorm(H); ln.weight = ln.weight.to(DEVICE); ln.bias = ln.bias.to(DEVICE)
    ms   = bench_ms(lambda: ln(x))
    rows.append(("layernorm_kernel",              ms, ai_layernorm(B, H),    f"B={B},H={H}"))

    # 3. gelu_kernel — Projector FFN (128 × 5120)
    N  = 128 * 5120
    xg = torch.randn(N, device=DEVICE)
    ms = bench_ms(lambda: gelu(xg))
    rows.append(("gelu_kernel",                   ms, ai_gelu(N),            f"N={N}"))

    # 4. silu_kernel — Text Decoder MLP (56 × 18944)
    N  = 56 * 18944
    xs = torch.randn(N, device=DEVICE)
    ms = bench_ms(lambda: silu(xs))
    rows.append(("silu_kernel",                   ms, ai_silu(N),            f"N={N}"))

    # 5. softmax_kernel — Text Decoder vocab logits (56 × 32000)
    BR, S = 56, 32000
    xsm   = torch.randn(BR, S, device=DEVICE)
    ms    = bench_ms(lambda: softmax(xsm))
    rows.append(("softmax_kernel",                ms, ai_softmax(BR, S),     f"{BR}×{S}"))

    # 6a. linear_kernel_tf32 small M — Text Decoder QKV (56×3584×3584)
    M, K, N = 56, 3584, 3584
    lin1    = Linear(K, N, bias=False)
    lin1.weight = torch.randn(N, K, device=DEVICE)
    x1 = torch.randn(M, K, device=DEVICE)
    ms = bench_ms(lambda: lin1(x1))
    rows.append(("linear_tf32 (small M=56)",      ms, ai_linear(M, N, K),   f"{M}×{K}×{N}"))

    # 6b. linear_kernel_tf32 large M — Audio Encoder FFN (512×1280×5120)
    M2, K2, N2 = 512, 1280, 5120
    lin2        = Linear(K2, N2, bias=False)
    lin2.weight = torch.randn(N2, K2, device=DEVICE)
    x2 = torch.randn(M2, K2, device=DEVICE)
    ms = bench_ms(lambda: lin2(x2))
    rows.append(("linear_tf32 (large M=512)",     ms, ai_linear(M2, N2, K2), f"{M2}×{K2}×{N2}"))

    # 7a. attention kernels — Audio Encoder (B=1,H=20,Sq=Sk=128,D=64)
    B, nH, Sq, Sk, D = 1, 20, 128, 128, 64
    qa = torch.randn(B, nH, Sq, D, device=DEVICE)
    ka = torch.randn(B, nH, Sk, D, device=DEVICE)
    va = torch.randn(B, nH, Sk, D, device=DEVICE)
    ms = bench_ms(lambda: scaled_dot_product_attention(qa, ka, va))
    rows.append(("attention (Encoder)",            ms, ai_attention_scores(B, nH, Sq, Sk, D), f"H={nH},Sq={Sq}"))

    # 7b. attention GQA — Text Decoder (B=1,H=28,KH=4,Sq=56,D=128)
    B, nH, kH, Sq, D = 1, 28, 4, 56, 128
    qd = torch.randn(B, nH, Sq, D, device=DEVICE)
    kd = torch.randn(B,  kH, Sq, D, device=DEVICE)
    vd = torch.randn(B,  kH, Sq, D, device=DEVICE)
    ms = bench_ms(lambda: scaled_dot_product_attention(qd, kd, vd))
    rows.append(("attention GQA (Decoder)",        ms, ai_attention_scores(B, nH, Sq, Sq, D), f"H={nH}/KH={kH}"))

    # 8. compute_freqs_kernel — RoPE for Text Decoder (S=56, D=128)
    # Force cache rebuild every call so we time the kernel, not a dict lookup.
    rope_bench = RotaryEmbedding(dim=128, max_position_embeddings=64, base=500000.0)
    x_r = torch.randn(1, 28, 57, 128, device=DEVICE)  # seq > cache → triggers rebuild
    ms  = bench_ms(lambda: rope_bench._update_cache(57, device=DEVICE))
    rows.append(("compute_freqs_kernel",           ms, 0.25,                 "S=57,D=128"))

    # Print table
    print(f"{'Kernel':<36} {'ms':>7}  {'AI (F/B)':>10}  Bound         Dims")
    print("-" * 80)
    for name, ms, ai, dims in rows:
        bound = "Compute ★" if ai > ridge else "Memory  "
        print(f"{name:<36} {ms:>7.4f}  {ai:>10.2f}  {bound}  {dims}")

    print("\n★ = compute-bound (AI > ridge point)")
    print("→ Paste 'ms' column into §3.3 and 'AI + Bound' columns into §4.1.")
    return rows


# ============================================================================
# SECTION 5.1  —  Tile size sweep
# ============================================================================

# NOTE: This is a LOCAL, non-autotuned copy of the matmul kernel used ONLY
# for the sweep. We cannot use linear_kernel_tf32 from layers.py because it
# is wrapped with @triton.autotune, which forbids passing BLOCK_M/N/K
# explicitly at the call site.
@triton.jit
def _sweep_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0,
        )
        b = tl.load(
            b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        )
        acc += tl.dot(a, b)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def block_size_sweep():
    """
    §5.1 Hypothesis: Larger BLOCK_M × BLOCK_N increases Tensor Core utilisation
    and amortises launch overhead — up to the point where register pressure or
    shared-memory capacity becomes the bottleneck.
    """
    print("\n" + "=" * 72)
    print("SECTION 5.1 — Tile Size Sweep for linear_kernel_tf32")
    print("=" * 72)
    print("Hypothesis: Larger BLOCK_M × BLOCK_N improves Tensor Core utilisation")
    print("  by increasing data reuse in SRAM, up to register/SRAM limits.\n")
    print("Shape: A(512×1280) @ B(1280×5120) — Audio Encoder FFN\n")

    M, K, N = 512, 1280, 5120
    a = torch.randn(M, K, device=DEVICE).contiguous()
    b = torch.randn(K, N, device=DEVICE).contiguous()

    configs = [
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
        ( 32,  32, 32, 2, 2),
        ( 64,  64, 32, 4, 2),
        ( 64,  64, 64, 4, 3),
        (128,  64, 32, 4, 3),
        ( 64, 128, 32, 4, 4),
        (128, 128, 32, 8, 3),
        (128, 128, 64, 8, 4),
    ]

    print(f"{'BM':>4} {'BN':>4} {'BK':>4} {'nW':>3} {'nS':>3} │ {'ms':>7} │ {'TFLOPS':>7} │ {'Speedup':>7}  Note")
    print("─" * 60)

    ref_ms   = None
    best_ms  = float("inf")
    best_cfg = None

    for bm, bn, bk, nw, ns in configs:
        Mp = pad_to_multiple(M, bm)
        Np = pad_to_multiple(N, bn)
        Kp = pad_to_multiple(K, bk)

        ap = torch.zeros(Mp, Kp, device=DEVICE); ap[:M, :K] = a
        bp = torch.zeros(Kp, Np, device=DEVICE); bp[:K, :N] = b
        cp = torch.zeros(Mp, Np, device=DEVICE)

        grid = (triton.cdiv(Mp, bm), triton.cdiv(Np, bn))

        # Capture loop variables explicitly to avoid Python closure issues
        def make_fn(ap_, bp_, cp_, bm_, bn_, bk_, nw_, ns_, Mp_, Np_, Kp_, g_):
            def fn():
                _sweep_matmul_kernel[g_](
                    ap_, bp_, cp_,
                    Mp_, Np_, Kp_,
                    ap_.stride(0), ap_.stride(1),
                    bp_.stride(0), bp_.stride(1),
                    cp_.stride(0), cp_.stride(1),
                    BLOCK_M=bm_, BLOCK_N=bn_, BLOCK_K=bk_,
                    num_warps=nw_, num_stages=ns_,
                )
            return fn

        try:
            ms = bench_ms(make_fn(ap, bp, cp, bm, bn, bk, nw, ns, Mp, Np, Kp, grid))
        except Exception as e:
            print(f"{bm:>4} {bn:>4} {bk:>4} {nw:>3} {ns:>3} │ {'ERR':>7} │ {'N/A':>7} │ {str(e)[:25]}")
            continue

        if ref_ms is None:
            ref_ms = ms

        tflops  = 2 * M * N * K / (ms * 1e-3) / 1e12
        speedup = ref_ms / ms
        note    = " ← BEST" if ms < best_ms else ""
        print(f"{bm:>4} {bn:>4} {bk:>4} {nw:>3} {ns:>3} │ {ms:>7.3f} │ {tflops:>7.3f} │ {speedup:>6.2f}x{note}")

        if ms < best_ms:
            best_ms  = ms
            best_cfg = (bm, bn, bk, nw, ns)

    if best_cfg:
        bm, bn, bk, nw, ns = best_cfg
        print(f"\nResult: Best config → BLOCK_M={bm}, BLOCK_N={bn}, BLOCK_K={bk}, "
              f"num_warps={nw}, num_stages={ns}")
        print(f"        Speedup over baseline (32×32×32): {ref_ms / best_ms:.2f}x")
    print("\n→ Paste this table into §5.1  (Hypothesis → Change → Result)")


# ============================================================================
# SECTION 5.2  —  Fused vs unfused SwiGLU
# ============================================================================

def fusion_comparison():
    """
    §5.2 Hypothesis: Fusing SiLU(gate_proj(x)) * up_proj(x) into a single
    kernel eliminates one global-memory round-trip for the gate activation,
    reducing latency.
    """
    print("\n" + "=" * 72)
    print("SECTION 5.2 — Kernel Fusion: Fused vs Unfused SwiGLU")
    print("=" * 72)
    print("Hypothesis: Fused SwiGLU saves 1 HBM write + 1 HBM read for the")
    print("  intermediate gate activation, reducing end-to-end MLP latency.\n")

    B, H, I = 56, 3584, 18944   # Text Decoder MLP realistic dimensions
    x = torch.randn(B, H, device=DEVICE)

    def make_mlp(fused: bool):
        MLP.FUSED = fused
        m = MLP(H, I, activation="silu", use_gating=True)
        m.gate_proj.weight = torch.randn(I, H, device=DEVICE)
        m.up_proj.weight   = torch.randn(I, H, device=DEVICE)
        m.down_proj.weight = torch.randn(H, I, device=DEVICE)
        # Invalidate the fused-weight cache so _prepare_fused_weights runs fresh
        m._gate_weight_t = None
        m._up_weight_t   = None
        return m

    mlp_unfused = make_mlp(False)
    mlp_fused   = make_mlp(True)

    MLP.FUSED = False
    ms_u = bench_ms(lambda: mlp_unfused(x))
    MLP.FUSED = True
    ms_f = bench_ms(lambda: mlp_fused(x))

    saved_mb = B * I * BYTES_F32 / 1e6   # one gate activation tensor in HBM

    print(f"{'Mode':<22} {'Latency (ms)':>14} {'Speedup':>10}")
    print("─" * 50)
    print(f"{'Unfused SwiGLU':<22} {ms_u:>14.4f} {'1.00x':>10}")
    print(f"{'Fused SwiGLU':<22} {ms_f:>14.4f} {ms_u / ms_f:>9.2f}x")
    print(f"\nHBM traffic saved  : ~{saved_mb:.1f} MB  (gate activation not written to HBM)")
    print("Result: (fill in — speedup observed, whether hypothesis confirmed)")
    print("\n→ Paste into §5.2  (Hypothesis → Change → Result)")

    MLP.FUSED = True   # restore default


# ============================================================================
# SECTION 6.3  —  Per-operator comparison vs example
# ============================================================================

def compare_vs_example():
    """
    §6.3 Compare template implementation against glm_asr_triton_example.
    Uses importlib so the example modules do not shadow our already-imported
    template modules.
    """
    print("\n" + "=" * 72)
    print("SECTION 6.3 — Per-Operator Comparison: Template vs Example Baseline")
    print("=" * 72)

    example_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm_asr_triton_example")
    )
    if not os.path.isdir(example_dir):
        print(f"⚠  Example directory not found:\n   {example_dir}")
        print("   Run from hw1-asr/glm_asr_triton_template/ — the sibling folder must exist.")
        return

    # Use importlib so we don't clobber the 'layers' / 'attention' already imported
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)
    try:
        ex_lay = importlib.import_module("layers")    if "layers_ex"    not in sys.modules else sys.modules["layers_ex"]
        ex_att = importlib.import_module("attention") if "attention_ex" not in sys.modules else sys.modules["attention_ex"]
    except ImportError as e:
        print(f"⚠  Import error: {e}")
        return

    cases = []

    # RMSNorm (Text Decoder: 56 × 3584)
    xn   = torch.randn(56, 3584, device=DEVICE)
    nt   = RMSNorm(3584);       nt.weight  = nt.weight.to(DEVICE)
    ne   = ex_lay.RMSNorm(3584); ne.weight = ne.weight.to(DEVICE)
    cases.append(("RMSNorm (H=3584)",
                  bench_ms(lambda: nt(xn)),
                  bench_ms(lambda: ne(xn))))

    # LayerNorm (Audio Encoder: 512 × 1280)
    xl   = torch.randn(512, 1280, device=DEVICE)
    lt   = LayerNorm(1280);        lt.weight  = lt.weight.to(DEVICE);  lt.bias  = lt.bias.to(DEVICE)
    le   = ex_lay.LayerNorm(1280); le.weight  = le.weight.to(DEVICE);  le.bias  = le.bias.to(DEVICE)
    cases.append(("LayerNorm (H=1280)",
                  bench_ms(lambda: lt(xl)),
                  bench_ms(lambda: le(xl))))

    # GELU (Projector FFN)
    xg = torch.randn(128 * 5120, device=DEVICE)
    cases.append(("GELU",
                  bench_ms(lambda: gelu(xg)),
                  bench_ms(lambda: ex_lay.gelu(xg))))

    # SiLU (Text Decoder MLP)
    xs = torch.randn(56 * 18944, device=DEVICE)
    cases.append(("SiLU",
                  bench_ms(lambda: silu(xs)),
                  bench_ms(lambda: ex_lay.silu(xs))))

    # Attention — Audio Encoder (MHA, no GQA needed for comparison)
    qa = torch.randn(1, 20, 128, 64, device=DEVICE)
    ka = torch.randn(1, 20, 128, 64, device=DEVICE)
    va = torch.randn(1, 20, 128, 64, device=DEVICE)
    cases.append(("Attention (Encoder)",
                  bench_ms(lambda: scaled_dot_product_attention(qa, ka, va)),
                  bench_ms(lambda: ex_att.scaled_dot_product_attention(qa, ka, va))))

    # Attention — Text Decoder (standard MHA shape; GQA tested separately)
    qd = torch.randn(1, 28, 56, 128, device=DEVICE)
    kd = torch.randn(1, 28, 56, 128, device=DEVICE)
    vd = torch.randn(1, 28, 56, 128, device=DEVICE)
    cases.append(("Attention (Decoder MHA)",
                  bench_ms(lambda: scaled_dot_product_attention(qd, kd, vd)),
                  bench_ms(lambda: ex_att.scaled_dot_product_attention(qd, kd, vd))))

    print(f"{'Operator':<28} {'Template (ms)':>14} {'Example (ms)':>14} {'Ratio T/E':>10}  Status")
    print("─" * 75)
    for name, ms_t, ms_e in cases:
        ratio  = ms_t / ms_e
        status = "✓ faster" if ratio < 0.95 else ("≈ parity" if ratio < 1.05 else "↑ slower")
        print(f"{name:<28} {ms_t:>14.4f} {ms_e:>14.4f} {ratio:>10.3f}  {status}")

    print("\n→ Paste into §6.3. Explain differences in §6.4 (e.g. autotune overhead,")
    print("  fusion benefit, GQA expansion cost).")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GLM-ASR Triton Profiler")
    parser.add_argument("--block-sweep",     action="store_true", help="§5.1: tile size sweep")
    parser.add_argument("--fusion-compare",  action="store_true", help="§5.2: fused vs unfused SwiGLU")
    parser.add_argument("--compare-example", action="store_true", help="§6.3: vs example baseline")
    parser.add_argument("--all",             action="store_true", help="Run every analysis")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU found. Run on a cluster node:")
        print("  srun -p Teaching -w saxa --gres gpu:1 --pty bash")
        sys.exit(1)

    global GPU_TFLOPS_FP32, GPU_BW_GBs
    GPU_TFLOPS_FP32, GPU_BW_GBs = get_gpu_specs()

    # §3.3 + §4.1 — always run
    benchmark_kernels()

    if args.block_sweep    or args.all: block_size_sweep()
    if args.fusion_compare or args.all: fusion_comparison()
    if args.compare_example or args.all: compare_vs_example()

    print("\n" + "=" * 72)
    print("Profiling complete. Copy the tables into §3.3, §4.1, §5.1, §5.2, §6.3.")
    print("=" * 72)


if __name__ == "__main__":
    main()
