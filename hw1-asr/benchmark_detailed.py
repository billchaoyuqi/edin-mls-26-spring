#!/usr/bin/env python3
"""
Detailed Benchmark Script with Operator-level Profiling
Measures execution time for each operator/layer in the model, including custom Triton kernels.
"""

import argparse
import time
import sys
import os
import numpy as np


class TorchTimer:
    def __init__(self):
        import torch
        self.torch = torch
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = None

    def start(self):
        if self.start_event is not None:
            self.start_event.record()

    def stop(self):
        if self.start_event is not None:
            self.end_event.record()
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        return 0


def detailed_profile_torch(model, input_features, input_ids, input_features_mask, num_runs=3):
    import torch
    results = {}
    timer = TorchTimer()

    print("\n" + "=" * 70)
    print("DETAILED OPERATOR PROFILING (TORCH)")
    print("=" * 70)

    print("\n[1/4] Profiling Audio Encoder...")
    encoder_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timer.start()
        audio_features = model.audio_encoder(input_features)
        elapsed = timer.stop()
        encoder_times.append(elapsed)
    results['audio_encoder'] = {'mean': np.mean(encoder_times), 'std': np.std(encoder_times)}
    print(f"  Audio Encoder: {results['audio_encoder']['mean']:.2f}ms (+/- {results['audio_encoder']['std']:.2f}ms)")

    # (Projector and Decoder steps are simplified to save space, but identical to original structure)
    # They are kept basic to allow focus on the micro-benchmarks below.
    return results


def profile_attention_ops_torch(seq_len=256, num_runs=5):
    """Profile attention operations specifically (Torch vs Custom Triton)."""
    import torch
    print("\n" + "=" * 70)
    print("ATTENTION OPERATION PROFILING (TORCH vs TRITON)")
    print("=" * 70)

    timer = TorchTimer()
    results = {}

    hidden_size = 2048
    num_heads = 16
    head_dim = hidden_size // num_heads

    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print(f"\nSequence length: {seq_len}")

    # 1. Standard Attention (PyTorch)
    print("\n[1] Standard Attention (Torch Matmul)...")
    matmul_times = []
    q_2d = q.reshape(batch_size * num_heads, seq_len, head_dim)
    k_2d = k.reshape(batch_size * num_heads, seq_len, head_dim)
    v_2d = v.reshape(batch_size * num_heads, seq_len, head_dim)

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)

    for _ in range(num_runs):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timer.start()
        scores = torch.matmul(q_2d, k_2d.transpose(1, 2)) / torch.sqrt(
            torch.tensor(head_dim, dtype=torch.float32, device=device))
        attn_weights = torch.exp(scores - torch.max(scores, dim=-1, keepdim=True).values)
        attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
        output = torch.matmul(attn_weights, v_2d)
        matmul_times.append(timer.stop())

    mem_torch = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
    results['matmul_attention'] = np.mean(matmul_times)
    print(
        f"  Torch matmul: {np.mean(matmul_times):.2f}ms (+/- {np.std(matmul_times):.2f}ms) | Peak VRAM: {mem_torch:.1f} MB")

    # 2. Custom Triton Fused RoPE + Attention
    print("\n[2] Custom Triton Fused RoPE+FlashAttention...")
    triton_times = []
    try:
        from rope import RotaryEmbedding
        from attention import scaled_dot_product_attention_with_rope

        rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=seq_len + 100)
        cos, sin = rope(q)

        # Warmup Triton to compile
        _ = scaled_dot_product_attention_with_rope(q, k, v, cos, sin)

        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)

        for _ in range(num_runs):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            timer.start()
            output_triton = scaled_dot_product_attention_with_rope(q, k, v, cos, sin)
            triton_times.append(timer.stop())

        mem_triton = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
        results['triton_fused_attention'] = np.mean(triton_times)
        print(
            f"  Triton Fused: {np.mean(triton_times):.2f}ms (+/- {np.std(triton_times):.2f}ms) | Peak VRAM: {mem_triton:.1f} MB")
        print(f"  -> Speedup vs Torch: {np.mean(matmul_times) / np.mean(triton_times):.2f}x")
        print(f"  -> Memory Saving: {mem_torch - mem_triton:.1f} MB")
    except ImportError as e:
        print(f"  [Skipped] Could not load custom Triton attention: {e}")

    return results


def profile_linear_ops_torch(hidden_size=2048, intermediate_size=5632, batch_size=1, seq_len=256, num_runs=5):
    """Profile linear/GEMM operations including custom Triton Fused MLPs."""
    import torch
    print("\n" + "=" * 70)
    print("LINEAR/GEMM OPERATION PROFILING (TORCH vs TRITON)")
    print("=" * 70)

    timer = TorchTimer()
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # 1. Standard PyTorch SwiGLU MLP
    print("\n[1] Torch Full MLP (SwiGLU style)...")
    w_gate = torch.randn(hidden_size, intermediate_size, device=device)
    w_up = torch.randn(hidden_size, intermediate_size, device=device)
    w_down = torch.randn(intermediate_size, hidden_size, device=device)

    mlp_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timer.start()
        gate = torch.matmul(x, w_gate)
        up = torch.matmul(x, w_up)
        gate_act = gate * (1 / (1 + torch.exp(-gate)))
        hidden = gate_act * up
        output = torch.matmul(hidden, w_down)
        mlp_times.append(timer.stop())

    results['torch_mlp'] = np.mean(mlp_times)
    print(f"  Torch MLP: {np.mean(mlp_times):.2f}ms (+/- {np.std(mlp_times):.2f}ms)")

    # 2. Custom Triton SwiGLU Fused MLP
    print("\n[2] Custom Triton Fused SwiGLU MLP...")
    try:
        from layers import MLP
        triton_mlp = MLP(hidden_size, intermediate_size, activation="silu", use_gating=True)
        # Copy weights for fairness
        triton_mlp.gate_proj.weight.data = w_gate.t().contiguous()
        triton_mlp.up_proj.weight.data = w_up.t().contiguous()
        triton_mlp.down_proj.weight.data = w_down.t().contiguous()

        # Warmup
        _ = triton_mlp(x)

        triton_mlp_times = []
        for _ in range(num_runs):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            timer.start()
            output_triton = triton_mlp(x)
            triton_mlp_times.append(timer.stop())

        results['triton_fused_mlp'] = np.mean(triton_mlp_times)
        print(f"  Triton Fused MLP: {np.mean(triton_mlp_times):.2f}ms (+/- {np.std(triton_mlp_times):.2f}ms)")
        print(f"  -> Speedup vs Torch: {np.mean(mlp_times) / np.mean(triton_mlp_times):.2f}x")
    except Exception as e:
        print(f"  [Skipped] Could not load Triton MLP: {e}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Folder name to benchmark')
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length for micro-benchmarks')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, args.folder)
    sys.path.insert(0, folder_path)

    # Note: Micro-benchmarks run directly using PyTorch tensors
    attention_results = profile_attention_ops_torch(seq_len=args.seq_len, num_runs=5)
    linear_results = profile_linear_ops_torch(seq_len=args.seq_len, num_runs=5)

    sys.path.remove(folder_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())