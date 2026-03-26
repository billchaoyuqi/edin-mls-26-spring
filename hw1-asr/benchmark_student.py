#!/usr/bin/env python3
"""
Student Version Benchmark Script
Tests student implementations against expected output.

Usage:
    python benchmark_student.py <folder_name>
    python benchmark_student.py glm_asr_triton_example --duration 15
    python benchmark_student.py glm_asr_triton_example --profile
"""

import argparse
import time
import sys
import os
import numpy as np
import importlib

# Expected transcription for the test audio
EXPECTED_TEXT = "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"


def download_librispeech_sample():
    import urllib.request
    cache_dir = os.path.expanduser("~/.cache/glm_asr")
    os.makedirs(cache_dir, exist_ok=True)
    audio_path = os.path.join(cache_dir, "test_audio.flac")
    if os.path.exists(audio_path):
        return audio_path
    print("Downloading LibriSpeech sample...")
    url = "https://www.openslr.org/resources/12/test-clean/61/70968/61-70968-0000.flac"
    try:
        urllib.request.urlretrieve(url, audio_path)
        return audio_path
    except:
        return None


def load_test_audio(audio_path=None, force_duration=None):
    import wave
    import struct

    def read_wav(filepath):
        with wave.open(filepath, 'rb') as wav:
            sr = wav.getframerate()
            n_channels = wav.getnchannels()
            n_frames = wav.getnframes()
            sample_width = wav.getsampwidth()
            raw_data = wav.readframes(n_frames)
            if sample_width == 2:
                fmt = f'<{n_frames * n_channels}h'
                audio = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
            elif sample_width == 4:
                fmt = f'<{n_frames * n_channels}i'
                audio = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
            return audio, sr

    audio_array = None
    sr = 16000

    if force_duration is None:
        if audio_path and os.path.exists(audio_path):
            audio_paths = [audio_path]
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            audio_paths = [
                os.path.join(script_dir, "test_audio.wav"),
                "/tmp/test_audio.wav",
                os.path.expanduser("~/.cache/glm_asr/test_audio.wav"),
                os.path.expanduser("~/.cache/glm_asr/test_audio.flac"),
                "../test_audio.wav",
            ]

        for path in audio_paths:
            if os.path.exists(path):
                try:
                    audio_array, sr = read_wav(path)
                    print(f"Loaded audio from {path}")
                    break
                except Exception as e:
                    continue

    if audio_array is None or force_duration is not None:
        duration = force_duration if force_duration is not None else 5.0
        print(f"Using synthetic test audio ({duration}s) for scaling test")
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        # 混合不同频率，防止被模型轻易优化掉
        audio_array = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
        return audio_array.astype(np.float32), "[synthetic]", duration

    target_sr = 16000
    if sr != target_sr:
        old_indices = np.arange(len(audio_array))
        new_length = int(len(audio_array) * target_sr / sr)
        new_indices = np.linspace(0, len(audio_array) - 1, new_length)
        audio_array = np.interp(new_indices, old_indices, audio_array)

    duration = len(audio_array) / target_sr
    return audio_array.astype(np.float32), EXPECTED_TEXT, duration


def benchmark_triton_folder(folder_name, audio_array, num_warmup=1, num_runs=3, profile=False):
    import torch

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, folder_name)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    sys.path.insert(0, folder_path)

    for mod_name in list(sys.modules.keys()):
        if mod_name in ['weight_loader', 'model', 'layers', 'attention', 'rope', 'conv', 'decode_attention']:
            del sys.modules[mod_name]

    print(f"Loading model from {folder_name}...")
    from weight_loader import load_model_from_hf
    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_features, input_ids, input_features_mask = prepare_inputs_torch(audio_array, processor, device)

    generate_fn = model.generate
    for fn_name in ['generate_v8b', 'generate_v8', 'generate_v6']:
        if hasattr(model, fn_name):
            generate_fn = getattr(model, fn_name)
            break

    print(f"Using generate function: {generate_fn.__name__}")

    print(f"Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            try:
                _ = generate_fn(input_features, input_ids=input_ids, input_features_mask=input_features_mask,
                                max_new_tokens=100, temperature=1.0, top_k=1)
            except TypeError:
                _ = generate_fn(input_features, input_ids=input_ids, max_new_tokens=100, temperature=1.0, top_k=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    peak_memories = []

    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()

        is_profiling = profile and (i == num_runs - 1)
        if is_profiling:
            from torch.profiler import profile as torch_profile, record_function, ProfilerActivity
            print("  [PyTorch Profiler Active for this run...]")
            prof = torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                                 with_stack=False)
            prof.__enter__()

        with torch.no_grad():
            try:
                output = generate_fn(input_features, input_ids=input_ids, input_features_mask=input_features_mask,
                                     max_new_tokens=100, temperature=1.0, top_k=1)
            except TypeError:
                output = generate_fn(input_features, input_ids=input_ids, max_new_tokens=100, temperature=1.0, top_k=1)

        if is_profiling:
            prof.__exit__(None, None, None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("\n" + "-" * 40)
            print("OPERATOR-LEVEL BREAKDOWN (Top 15)")
            print("-" * 40)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
            print("-" * 40 + "\n")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            peak_memories.append(peak_mem_mb)

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        tokens = output.shape[1] - input_ids.shape[1]

        mem_str = f" | Peak VRAM: {peak_mem_mb:.1f} MB" if torch.cuda.is_available() else ""
        print(f"  Run {i + 1}: {elapsed:.1f}ms ({tokens} tokens){mem_str}")

    generated_np = output.detach().cpu().numpy()
    transcription = decode_output(generated_np, processor)

    sys.path.remove(folder_path)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'transcription': transcription,
        'tokens': tokens,
        'mean_mem': np.mean(peak_memories) if peak_memories else 0
    }


def prepare_inputs_torch(audio_array, processor, device):
    import torch
    if hasattr(processor, 'apply_transcription_request'):
        inputs = processor.apply_transcription_request(audio_array)
        input_features = inputs.input_features.to(device=device, dtype=torch.float32)
        input_ids = inputs.input_ids.to(device=device, dtype=torch.int64)
        input_features_mask = None
    else:
        features = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="max_length")
        input_features = features['input_features'].to(device=device, dtype=torch.float32)
        mel_frames = input_features.shape[-1]
        num_audio_tokens = max(1, mel_frames // 2 // 4)
        input_ids_list = [59253, 10, 59261] + [59260] * num_audio_tokens + [59262, 59253, 10] + [9249, 70891, 419, 7122,
                                                                                                 1119, 1467] + [59254,
                                                                                                                10]
        input_ids = torch.tensor([input_ids_list], dtype=torch.int64, device=device)
        input_features_mask = None
    return input_features, input_ids, input_features_mask


def decode_output(generated_np, processor):
    try:
        if hasattr(processor, 'tokenizer'):
            transcription = processor.tokenizer.decode(generated_np[0], skip_special_tokens=True)
        else:
            transcription = processor.decode(generated_np[0], skip_special_tokens=True)
        if "Please transcribe this audio into text" in transcription:
            transcription = transcription.split("Please transcribe this audio into text")[-1].strip()
        return transcription
    except Exception as e:
        return f"[decode error: {e}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Folder name to benchmark')
    parser.add_argument('--audio', type=str, help='Path to test audio file')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs')
    parser.add_argument('--duration', type=float, default=None, help='Force synthetic audio length (sec)')
    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiler on last run')
    args = parser.parse_args()

    print("=" * 70)
    print("GLM-ASR Student Version Benchmark")
    print("=" * 70)

    audio_array, expected, duration = load_test_audio(args.audio, args.duration)
    print(f"Audio duration: {duration:.2f}s")

    print("\n" + "=" * 70)
    print(f"Testing: {args.folder}")
    print("=" * 70)

    try:
        results = benchmark_triton_folder(args.folder, audio_array, args.warmup, args.runs, args.profile)

        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"Time:       {results['mean']:.1f}ms (+/- {results['std']:.1f}ms)")
        print(f"Peak VRAM:  {results.get('mean_mem', 0):.1f} MB")
        print(f"Speed:      {results['mean'] / results['tokens']:.2f}ms/token")
        print(f"\nTranscription: {results['transcription']}")
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())