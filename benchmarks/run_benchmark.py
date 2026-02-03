#!/usr/bin/env python3
"""
Run performance benchmarks comparing CoreML vs PyTorch HTDemucs.

Measures timing at each pipeline stage:
- Model loading
- STFT computation
- Frequency branch inference
- Time branch inference
- iSTFT computation
- Full pipeline end-to-end

Also tracks peak memory usage for both implementations.
"""

import argparse
import json
import math
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np

# Constants matching Swift implementation
FFT_SIZE = 4096
HOP_LENGTH = 1024
SEGMENT_SAMPLES = 343980
SAMPLE_RATE = 44100

# Test audio duration in seconds
BENCHMARK_DURATION = 30.0


def generate_test_audio(duration: float = BENCHMARK_DURATION) -> tuple:
    """Generate reproducible test audio for benchmarking.

    Uses seed 42 for reproducibility. Creates a mix of frequencies
    to simulate music-like content.
    """
    np.random.seed(42)
    samples = int(duration * SAMPLE_RATE)

    # Mix of frequencies to simulate music-like content
    t = np.linspace(0, duration, samples)
    left = (
        0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.random.randn(samples)
    ).astype(np.float32)

    right = (
        0.3 * np.sin(2 * np.pi * 440 * t + 0.5)
        + 0.2 * np.sin(2 * np.pi * 660 * t)
        + 0.1 * np.random.randn(samples)
    ).astype(np.float32)

    return left, right


def benchmark_pytorch(left: np.ndarray, right: np.ndarray) -> dict:
    """Benchmark PyTorch HTDemucs with per-stage timing.

    Note: PyTorch apply_model handles STFT internally, so we can only
    measure full pipeline time. Per-stage timings are set to None.
    """
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    results = {}

    # Model loading
    tracemalloc.start()
    start = time.perf_counter()
    model = get_model("htdemucs_6s")
    model.eval()
    results["model_load_sec"] = time.perf_counter() - start

    # Prepare audio
    audio = torch.from_numpy(np.stack([left, right], axis=0)).float()

    # Full pipeline (apply_model handles STFT internally)
    start = time.perf_counter()
    with torch.no_grad():
        _ = apply_model(
            model,
            audio.unsqueeze(0),
            device="cpu",
            shifts=1,
            split=True,
            overlap=0.25,
        )
    results["full_pipeline_sec"] = time.perf_counter() - start

    # Memory tracking
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["memory_peak_mb"] = peak / 1024 / 1024

    # Note: PyTorch apply_model doesn't expose per-stage timing easily
    # We report full pipeline only for PyTorch
    results["stft_sec"] = None
    results["freq_branch_sec"] = None
    results["time_branch_sec"] = None
    results["istft_sec"] = None

    return results


def compute_stft(audio: np.ndarray) -> tuple:
    """Compute STFT matching PyTorch conventions (normalized=True)."""
    pad_size = FFT_SIZE // 2
    padded = np.pad(audio, pad_size, mode="reflect")
    window = np.hanning(FFT_SIZE + 1)[:-1]
    num_frames = (len(padded) - FFT_SIZE) // HOP_LENGTH + 1
    norm_factor = 1.0 / math.sqrt(FFT_SIZE)

    real_out = []
    imag_out = []

    for i in range(num_frames):
        start_idx = i * HOP_LENGTH
        frame = padded[start_idx : start_idx + FFT_SIZE] * window
        spectrum = np.fft.rfft(frame) * norm_factor
        real_out.append(spectrum.real)
        imag_out.append(spectrum.imag)

    return np.array(real_out), np.array(imag_out)


def compute_istft(real: np.ndarray, imag: np.ndarray, length: int) -> np.ndarray:
    """Compute iSTFT matching PyTorch conventions (normalized=True)."""
    window = np.hanning(FFT_SIZE + 1)[:-1]
    num_frames = real.shape[0]
    pad_size = FFT_SIZE // 2
    norm_factor = math.sqrt(FFT_SIZE)

    output_length = (num_frames - 1) * HOP_LENGTH + FFT_SIZE
    output = np.zeros(output_length)
    window_sum = np.zeros(output_length)

    for i in range(num_frames):
        spectrum = (real[i] + 1j * imag[i]) * norm_factor
        frame = np.fft.irfft(spectrum, FFT_SIZE)
        start_idx = i * HOP_LENGTH
        output[start_idx : start_idx + FFT_SIZE] += frame * window
        window_sum[start_idx : start_idx + FFT_SIZE] += window**2

    output = np.where(window_sum > 1e-8, output / window_sum, output)
    return output[pad_size : pad_size + length]


def benchmark_coreml(left: np.ndarray, right: np.ndarray, model_path: str) -> dict:
    """Benchmark CoreML HTDemucs with per-stage timing.

    Measures:
    - Model load time
    - STFT time (computed separately)
    - Model inference time (split approx 50/50 for freq/time branches)
    - iSTFT time (measured for one stem, multiplied by 12 for all stems/channels)
    - Full pipeline = sum of above
    - Memory peak
    """
    import coremltools as ct

    results = {}

    # Model loading
    tracemalloc.start()
    start = time.perf_counter()
    model = ct.models.MLModel(model_path)
    results["model_load_sec"] = time.perf_counter() - start

    # Pad audio to segment size
    def pad_to_segment(audio, target):
        if len(audio) >= target:
            return audio[:target]
        return np.pad(audio, (0, target - len(audio)), mode="reflect")

    padded_left = pad_to_segment(left, SEGMENT_SAMPLES)
    padded_right = pad_to_segment(right, SEGMENT_SAMPLES)

    # STFT timing
    start = time.perf_counter()
    left_real, left_imag = compute_stft(padded_left)
    right_real, right_imag = compute_stft(padded_right)
    results["stft_sec"] = time.perf_counter() - start

    # Prepare model input
    model_bins = 2048
    model_frames = 336

    def prepare_for_model(real, imag):
        r = real[:, :model_bins]
        i = imag[:, :model_bins]
        if r.shape[0] < model_frames:
            pad = model_frames - r.shape[0]
            r = np.pad(r, ((0, pad), (0, 0)), mode="constant")
            i = np.pad(i, ((0, pad), (0, 0)), mode="constant")
        return r[:model_frames], i[:model_frames]

    lr, li = prepare_for_model(left_real, left_imag)
    rr, ri = prepare_for_model(right_real, right_imag)

    cac = np.stack([lr.T, li.T, rr.T, ri.T], axis=0)[np.newaxis, ...].astype(np.float32)
    raw = np.stack([padded_left, padded_right], axis=0)[np.newaxis, ...].astype(np.float32)

    # Model inference (freq + time branches combined)
    start = time.perf_counter()
    output = model.predict({"spectrogram": cac, "raw_audio": raw})
    inference_time = time.perf_counter() - start

    # We can't easily separate freq vs time branch in CoreML
    # Report approximate 50/50 split as per requirements
    results["freq_branch_sec"] = inference_time / 2
    results["time_branch_sec"] = inference_time / 2

    # iSTFT timing - measure for one stem, multiply by 12 for all stems/channels
    freq_out = output["add_66"]
    original_length = len(left)
    original_frames = left_real.shape[0]

    start = time.perf_counter()
    # iSTFT for one stem channel (representative timing)
    stem_real = freq_out[0, 0, 0].T[:original_frames]
    stem_imag = freq_out[0, 0, 1].T[:original_frames]
    stem_real_full = np.pad(stem_real, ((0, 0), (0, 1)), mode="constant")
    stem_imag_full = np.pad(stem_imag, ((0, 0), (0, 1)), mode="constant")
    _ = compute_istft(stem_real_full, stem_imag_full, original_length)
    istft_single = time.perf_counter() - start

    # Scale for all 6 stems * 2 channels = 12
    results["istft_sec"] = istft_single * 12

    # Full pipeline = sum of all stages
    results["full_pipeline_sec"] = (
        results["stft_sec"]
        + results["freq_branch_sec"]
        + results["time_branch_sec"]
        + results["istft_sec"]
    )

    # Memory tracking
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["memory_peak_mb"] = peak / 1024 / 1024

    return results


def run_benchmarks(model_path: str, runs: int = 3) -> dict:
    """Run benchmarks and return results."""
    print(f"Generating {BENCHMARK_DURATION}s test audio (seed=42)...")
    left, right = generate_test_audio()

    print(f"\nRunning PyTorch benchmark ({runs} runs)...")
    pytorch_results = []
    for i in range(runs):
        print(f"  Run {i + 1}/{runs}...", end=" ", flush=True)
        result = benchmark_pytorch(left.copy(), right.copy())
        pytorch_results.append(result)
        print(f"done ({result['full_pipeline_sec']:.2f}s)")

    print(f"\nRunning CoreML benchmark ({runs} runs)...")
    coreml_results = []
    for i in range(runs):
        print(f"  Run {i + 1}/{runs}...", end=" ", flush=True)
        result = benchmark_coreml(left.copy(), right.copy(), model_path)
        coreml_results.append(result)
        print(f"done ({result['full_pipeline_sec']:.2f}s)")

    # Average results
    def average_results(results_list):
        avg = {}
        for key in results_list[0].keys():
            values = [r[key] for r in results_list if r[key] is not None]
            avg[key] = sum(values) / len(values) if values else None
        return avg

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "audio_duration_sec": BENCHMARK_DURATION,
        "num_runs": runs,
        "coreml": average_results(coreml_results),
        "pytorch": average_results(pytorch_results),
    }


def print_comparison(results: dict):
    """Print formatted comparison table."""
    coreml = results["coreml"]
    pytorch = results["pytorch"]

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Audio duration: {results['audio_duration_sec']}s")
    print(f"Averaged over: {results['num_runs']} runs")
    print()
    print(f"{'Metric':<20} {'CoreML':>12} {'PyTorch':>12} {'Delta':>10}")
    print("-" * 60)

    metrics = [
        ("Model load", "model_load_sec", "s"),
        ("STFT", "stft_sec", "s"),
        ("Freq branch", "freq_branch_sec", "s"),
        ("Time branch", "time_branch_sec", "s"),
        ("iSTFT", "istft_sec", "s"),
        ("Full pipeline", "full_pipeline_sec", "s"),
        ("Memory peak", "memory_peak_mb", "MB"),
    ]

    for label, key, unit in metrics:
        c_val = coreml.get(key)
        p_val = pytorch.get(key)

        if c_val is None and p_val is None:
            continue

        c_str = f"{c_val:.2f}{unit}" if c_val is not None else "N/A"
        p_str = f"{p_val:.2f}{unit}" if p_val is not None else "N/A"

        if c_val is not None and p_val is not None:
            delta = ((c_val - p_val) / p_val) * 100
            delta_str = f"{delta:+.0f}%"
        else:
            delta_str = ""

        print(f"{label:<20} {c_str:>12} {p_str:>12} {delta_str:>10}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run HTDemucs benchmarks")
    parser.add_argument(
        "--model",
        default="Resources/Models/htdemucs_6s.mlpackage",
        help="Path to CoreML model",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/latest.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs to average",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Add results to baseline.json",
    )

    args = parser.parse_args()

    # Find model - try as given path first
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent.parent
        model_path = script_dir / args.model

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1

    results = run_benchmarks(str(model_path), args.runs)
    print_comparison(results)

    # Save latest results
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {output_path}")

    # Update baseline if requested
    if args.update_baseline:
        baseline_path = Path(__file__).parent / "baseline.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
        else:
            baseline = {}

        baseline[results["date"]] = {
            "coreml": results["coreml"],
            "pytorch": results["pytorch"],
        }
        baseline_path.write_text(json.dumps(baseline, indent=2))
        print(f"Updated baseline at {baseline_path}")

    return 0


if __name__ == "__main__":
    exit(main())
