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
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
=======
>>>>>>> Stashed changes


class CUDATimer:
    """CUDA event-based timer for accurate GPU timing."""

    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.start_event = cp.cuda.Event()
        self.end_event = cp.cuda.Event()

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        self.end_event.synchronize()
        # CuPy uses get_elapsed_time instead of elapsed_time
        return self.cp.cuda.get_elapsed_time(self.start_event, self.end_event)
>>>>>>> 5888348787a1449fd9e4bbb15e5df01bb113caf5


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
<<<<<<< HEAD
        return 0
=======
        elapsed = (time.perf_counter() - self._start_time) * 1000
        return elapsed



def detailed_profile(model, input_features, input_ids, input_features_mask, num_runs=3):
    """Detailed profiling of model components."""
    import cupy as cp

    results = {}
    timer = CUDATimer()

    print("\n" + "="*70)
    print("DETAILED OPERATOR PROFILING")
    print("="*70)

    # 1. Profile Audio Encoder
    print("\n[1/4] Profiling Audio Encoder...")
    encoder_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        audio_features = model.audio_encoder(input_features)
        elapsed = timer.stop()
        encoder_times.append(elapsed)
    results['audio_encoder'] = {
        'mean': np.mean(encoder_times),
        'std': np.std(encoder_times),
        'min': np.min(encoder_times),
        'max': np.max(encoder_times)
    }
    print(f"  Audio Encoder: {results['audio_encoder']['mean']:.2f}ms (+/- {results['audio_encoder']['std']:.2f}ms)")

    # 2. Profile Multi-modal Projector
    print("\n[2/4] Profiling Multi-modal Projector...")
    projector_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        projected = model.multi_modal_projector(audio_features)
        elapsed = timer.stop()
        projector_times.append(elapsed)
    results['projector'] = {
        'mean': np.mean(projector_times),
        'std': np.std(projector_times),
        'min': np.min(projector_times),
        'max': np.max(projector_times)
    }
    print(f"  Projector: {results['projector']['mean']:.2f}ms (+/- {results['projector']['std']:.2f}ms)")

    # 3. Profile Text Decoder (prefill phase)
    print("\n[3/4] Profiling Text Decoder (Prefill)...")

    # Build input embeddings
    embed_tokens = model.text_decoder.embed_tokens
    text_embeds = embed_tokens(input_ids)

    # Find audio token positions
    audio_token_id = 59260
    audio_mask = (input_ids == audio_token_id)

    # Create combined embeddings
    combined_embeds = text_embeds.copy()
    if cp.any(audio_mask):
        audio_positions = cp.where(audio_mask[0])[0]
        num_audio_tokens = len(audio_positions)
        if num_audio_tokens <= projected.shape[1]:
            combined_embeds[0, audio_positions[:projected.shape[1]]] = projected[0, :num_audio_tokens]

    prefill_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        # Call with inputs_embeds argument
        hidden_states = model.text_decoder(inputs_embeds=combined_embeds)
        elapsed = timer.stop()
        prefill_times.append(elapsed)
    results['decoder_prefill'] = {
        'mean': np.mean(prefill_times),
        'std': np.std(prefill_times),
        'min': np.min(prefill_times),
        'max': np.max(prefill_times)
    }
    print(f"  Decoder Prefill: {results['decoder_prefill']['mean']:.2f}ms (+/- {results['decoder_prefill']['std']:.2f}ms)")

    # 4. Profile Decode Steps (autoregressive)
    print("\n[4/4] Profiling Decode Steps...")
    decode_times = []
    num_decode_steps = 10

    # Get logits and sample first token
    logits = model.lm_head(hidden_states[:, -1:, :])
    next_token = cp.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    for step in range(num_decode_steps):
        cp.cuda.Device().synchronize()
        timer.start()

        # Single decode step
        next_embed = embed_tokens(next_token)
        step_hidden = model.text_decoder(inputs_embeds=next_embed)
        step_logits = model.lm_head(step_hidden)
        next_token = cp.argmax(step_logits[:, -1, :], axis=-1, keepdims=True)

        elapsed = timer.stop()
        decode_times.append(elapsed)

    results['decode_step'] = {
        'mean': np.mean(decode_times),
        'std': np.std(decode_times),
        'min': np.min(decode_times),
        'max': np.max(decode_times)
    }
    print(f"  Decode Step (avg): {results['decode_step']['mean']:.2f}ms (+/- {results['decode_step']['std']:.2f}ms)")

    # 5. Profile individual layers in decoder
    print("\n[5] Profiling Individual Decoder Layers...")
    layer_times = []

    try:
        test_input = combined_embeds
        seq_len = test_input.shape[1]

        # Try to get layers - different model versions have different structures
        if hasattr(model.text_decoder, 'layers'):
            layers = model.text_decoder.layers[:5]
        else:
            layers = []

        for i, layer in enumerate(layers):
            times = []
            for _ in range(num_runs):
                cp.cuda.Device().synchronize()
                timer.start()
                # Try calling with position_ids if needed
                try:
                    test_output = layer(test_input)
                except TypeError:
                    position_ids = cp.arange(seq_len, dtype=cp.int64).reshape(1, -1)
                    test_output = layer(test_input, position_ids=position_ids)
                elapsed = timer.stop()
                times.append(elapsed)

            layer_times.append({
                'name': f'layer_{i}',
                'mean': np.mean(times),
                'std': np.std(times)
            })
            print(f"  Layer {i}: {np.mean(times):.2f}ms (+/- {np.std(times):.2f}ms)")
            test_input = test_output
    except Exception as e:
        print(f"  Layer profiling skipped: {e}")

    results['layers'] = layer_times

    return results
>>>>>>> 5888348787a1449fd9e4bbb15e5df01bb113caf5


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

<<<<<<< HEAD
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
=======
    print("\n[2/4] Profiling Multi-modal Projector...")
    projector_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        projected = model.multi_modal_projector(audio_features)
        elapsed = timer.stop()
        projector_times.append(elapsed)
    results['projector'] = {
        'mean': np.mean(projector_times),
        'std': np.std(projector_times),
        'min': np.min(projector_times),
        'max': np.max(projector_times)
    }
    print(f"  Projector: {results['projector']['mean']:.2f}ms (+/- {results['projector']['std']:.2f}ms)")

    print("\n[3/4] Profiling Text Decoder (Prefill)...")
    embed_tokens = model.text_decoder.embed_tokens
    text_embeds = embed_tokens(input_ids)

    audio_token_id = 59260
    audio_mask = (input_ids == audio_token_id)

    combined_embeds = text_embeds.clone()
    if torch.any(audio_mask):
        audio_positions = torch.where(audio_mask[0])[0]
        num_audio_tokens = int(audio_positions.numel())
        if num_audio_tokens <= projected.shape[1]:
            combined_embeds[0, audio_positions[:projected.shape[1]]] = projected[0, :num_audio_tokens]

    prefill_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        hidden_states = model.text_decoder(inputs_embeds=combined_embeds)
        elapsed = timer.stop()
        prefill_times.append(elapsed)
    results['decoder_prefill'] = {
        'mean': np.mean(prefill_times),
        'std': np.std(prefill_times),
        'min': np.min(prefill_times),
        'max': np.max(prefill_times)
    }
    print(f"  Decoder Prefill: {results['decoder_prefill']['mean']:.2f}ms (+/- {results['decoder_prefill']['std']:.2f}ms)")

    print("\n[4/4] Profiling Decode Steps...")
    decode_times = []
    num_decode_steps = 10

    logits = model.lm_head(hidden_states[:, -1:, :])
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    for _ in range(num_decode_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        next_embed = embed_tokens(next_token)
        step_hidden = model.text_decoder(inputs_embeds=next_embed)
        step_logits = model.lm_head(step_hidden)
        next_token = torch.argmax(step_logits[:, -1, :], dim=-1, keepdim=True)
        elapsed = timer.stop()
        decode_times.append(elapsed)

    results['decode_step'] = {
        'mean': np.mean(decode_times),
        'std': np.std(decode_times),
        'min': np.min(decode_times),
        'max': np.max(decode_times)
    }
    print(f"  Decode Step (avg): {results['decode_step']['mean']:.2f}ms (+/- {results['decode_step']['std']:.2f}ms)")

    print("\n[5] Profiling Individual Decoder Layers...")
    layer_times = []

    try:
        test_input = combined_embeds
        seq_len = test_input.shape[1]

        if hasattr(model.text_decoder, 'layers'):
            layers = model.text_decoder.layers[:5]
        else:
            layers = []

        for i, layer in enumerate(layers):
            times = []
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timer.start()
                try:
                    test_output = layer(test_input)
                except TypeError:
                    position_ids = torch.arange(seq_len, dtype=torch.int64, device=test_input.device).reshape(1, -1)
                    test_output = layer(test_input, position_ids=position_ids)
                elapsed = timer.stop()
                times.append(elapsed)

            layer_times.append({
                'name': f'layer_{i}',
                'mean': np.mean(times),
                'std': np.std(times)
            })
            print(f"  Layer {i}: {np.mean(times):.2f}ms (+/- {np.std(times):.2f}ms)")
            test_input = test_output
    except Exception as e:
        print(f"  Layer profiling skipped: {e}")

    results['layers'] = layer_times
>>>>>>> 5888348787a1449fd9e4bbb15e5df01bb113caf5

    return results


<<<<<<< Updated upstream
=======
<<<<<<< HEAD
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Folder name to benchmark')
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length for micro-benchmarks')
=======
>>>>>>> Stashed changes

def print_summary(component_results):
    """Print a summary table of all profiling results."""
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)

    print("\n{:<35} {:>12} {:>12}".format("Component", "Time (ms)", "% of Total"))
    print("-"*60)

    # Calculate total time
    total = 0
    if component_results:
        for key in ['audio_encoder', 'projector', 'decoder_prefill']:
            if key in component_results:
                total += component_results[key]['mean']
        # Add estimated decode time (50 steps)
        if 'decode_step' in component_results:
            total += component_results['decode_step']['mean'] * 50

    if component_results:
        for key, label in [
            ('audio_encoder', 'Audio Encoder'),
            ('projector', 'Multi-modal Projector'),
            ('decoder_prefill', 'Decoder (Prefill)'),
        ]:
            if key in component_results:
                t = component_results[key]['mean']
                pct = (t / total * 100) if total > 0 else 0
                print(f"{label:<35} {t:>10.2f}ms {pct:>10.1f}%")

        if 'decode_step' in component_results:
            t = component_results['decode_step']['mean'] * 50
            pct = (t / total * 100) if total > 0 else 0
            print(f"{'Decoder (50 decode steps)':<35} {t:>10.2f}ms {pct:>10.1f}%")

    print("-"*60)
    print(f"{'TOTAL (estimated for 50 tokens)':<35} {total:>10.2f}ms")


def run_nsys_profile(folder, audio_path=None, runs=1):
    """Run Nsight Systems profiling."""
    import subprocess

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_name = f"profile_{folder}"

    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        "--output", output_name,
        "--force-overwrite", "true",
        "python", os.path.join(script_dir, "benchmark_student.py"),
        folder, "--warmup", "1", "--runs", str(runs)
    ]

    if audio_path:
        cmd.extend(["--audio", audio_path])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=script_dir)
    print(f"\nProfile saved to: {output_name}.nsys-rep")
    print("Open with: nsys-ui " + output_name + ".nsys-rep")


def main():
    parser = argparse.ArgumentParser(description='Detailed operator profiling')
    parser.add_argument('folder', type=str, nargs='?', default='glm_asr_cutile_example',
                       help='Folder name to benchmark')
    parser.add_argument('--audio', type=str, help='Path to test audio file')
    parser.add_argument('--runs', type=int, default=3, help='Number of profiling runs')
    parser.add_argument('--nsys', action='store_true', help='Run Nsight Systems profiling')
<<<<<<< Updated upstream
=======
>>>>>>> 5888348787a1449fd9e4bbb15e5df01bb113caf5
>>>>>>> Stashed changes
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, args.folder)
    sys.path.insert(0, folder_path)

<<<<<<< HEAD
    # Note: Micro-benchmarks run directly using PyTorch tensors
    attention_results = profile_attention_ops_torch(seq_len=args.seq_len, num_runs=5)
    linear_results = profile_linear_ops_torch(seq_len=args.seq_len, num_runs=5)
=======
    # Clear cached modules
    for mod_name in list(sys.modules.keys()):
        if mod_name in ['weight_loader', 'model', 'layers', 'attention', 'rope', 'conv']:
            del sys.modules[mod_name]

    print(f"\nLoading model from {args.folder}...")
    from weight_loader import load_model_from_hf
    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    if use_torch_backend:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(processor, 'apply_transcription_request'):
            inputs = processor.apply_transcription_request(audio_array)
            input_features = inputs.input_features.to(device=device, dtype=torch.float32)
            input_ids = inputs.input_ids.to(device=device, dtype=torch.int64)
            input_features_mask = None
            if hasattr(inputs, 'input_features_mask') and inputs.input_features_mask is not None:
                input_features_mask = inputs.input_features_mask.to(device=device, dtype=torch.float32)
        else:
            features = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="max_length")
            input_features = features['input_features'].to(device=device, dtype=torch.float32)
            input_ids = torch.tensor([[59253, 10, 59261] + [59260] * 100 + [59262, 59253, 10, 9249, 70891, 419, 7122, 1119, 1467, 59254, 10]],
                                     dtype=torch.int64, device=device)
            input_features_mask = None

        print(f"Input features shape: {input_features.shape}")
        print(f"Input IDs shape: {input_ids.shape}")

        component_results = detailed_profile_torch(model, input_features, input_ids, input_features_mask, num_runs=args.runs)
    else:
        import cupy as cp
        if hasattr(processor, 'apply_transcription_request'):
            inputs = processor.apply_transcription_request(audio_array)
            input_features = cp.asarray(inputs.input_features.numpy(), dtype=cp.float32)
            input_ids = cp.asarray(inputs.input_ids.numpy(), dtype=cp.int64)
            input_features_mask = None
            if hasattr(inputs, 'input_features_mask') and inputs.input_features_mask is not None:
                input_features_mask = cp.asarray(inputs.input_features_mask.numpy(), dtype=cp.float32)
        else:
            features = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="max_length")
            input_features = cp.asarray(features['input_features'].numpy(), dtype=cp.float32)
            input_ids = cp.array([[59253, 10, 59261] + [59260] * 100 + [59262, 59253, 10, 9249, 70891, 419, 7122, 1119, 1467, 59254, 10]], dtype=cp.int64)
            input_features_mask = None

        print(f"Input features shape: {input_features.shape}")
        print(f"Input IDs shape: {input_ids.shape}")

        component_results = detailed_profile(model, input_features, input_ids, input_features_mask, num_runs=args.runs)

    # Print summary
    print_summary(component_results)
<<<<<<< Updated upstream
=======
>>>>>>> 5888348787a1449fd9e4bbb15e5df01bb113caf5
>>>>>>> Stashed changes

    sys.path.remove(folder_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())