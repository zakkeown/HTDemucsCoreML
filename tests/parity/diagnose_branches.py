#!/usr/bin/env python3
"""
Branch isolation diagnostic: Compare CoreML vs PyTorch for each branch separately.

This script:
1. Runs PyTorch HTDemucs and extracts freq_output and time_output separately
2. Runs CoreML model and extracts both outputs
3. Compares them numerically (stats, correlation, SNR)
4. Saves audio files for listening to each branch in isolation

Usage:
    python diagnose_branches.py [input_audio.wav]

If no input is provided, generates a synthetic test signal.
"""

import numpy as np
import torch
import soundfile as sf
from pathlib import Path
import argparse
import json
import coremltools as ct

# Constants matching both pipelines
SAMPLE_RATE = 44100
SEGMENT_SAMPLES = 343980
FFT_SIZE = 4096
HOP_LENGTH = 1024
NUM_BINS = FFT_SIZE // 2 + 1
MODEL_BINS = 2048
MODEL_FRAMES = 336

STEM_NAMES = ["drums", "bass", "other", "vocals", "guitar", "piano"]


def compute_stft(audio: np.ndarray) -> tuple:
    """Compute STFT matching HTDemucs._spec exactly (normalized=True, specific padding)."""
    import math

    # HTDemucs padding scheme (different from standard center padding!)
    # pad = hop_length // 2 * 3 = 1536
    # le = ceil(length / hop_length)
    # pad_right = pad + le * hop_length - length
    length = len(audio)
    le = math.ceil(length / HOP_LENGTH)
    pad_left = HOP_LENGTH // 2 * 3  # 1536
    pad_right = pad_left + le * HOP_LENGTH - length

    padded = np.pad(audio, (pad_left, pad_right), mode='reflect')

    window = np.hanning(FFT_SIZE + 1)[:-1]

    # PyTorch normalized=True divides by sqrt(N)
    norm_factor = 1.0 / math.sqrt(FFT_SIZE)

    # Compute all frames
    num_frames_raw = (len(padded) - FFT_SIZE) // HOP_LENGTH + 1

    real_out = []
    imag_out = []

    for i in range(num_frames_raw):
        start = i * HOP_LENGTH
        frame = padded[start:start + FFT_SIZE] * window
        spectrum = np.fft.rfft(frame) * norm_factor
        # Drop Nyquist bin to match HTDemucs [:-1]
        real_out.append(spectrum.real[:-1])
        imag_out.append(spectrum.imag[:-1])

    # HTDemucs trims: z[..., 2: 2 + le] - skip first 2 frames, keep le frames
    real_out = np.array(real_out)[2:2 + le]
    imag_out = np.array(imag_out)[2:2 + le]

    return real_out, imag_out


def compute_istft(real: np.ndarray, imag: np.ndarray, length: int) -> np.ndarray:
    """Compute iSTFT matching PyTorch conventions (normalized=True)."""
    import math
    window = np.hanning(FFT_SIZE + 1)[:-1]
    num_frames = real.shape[0]
    pad_size = FFT_SIZE // 2

    # Undo the 1/sqrt(N) normalization from forward STFT
    norm_factor = math.sqrt(FFT_SIZE)

    output_length = (num_frames - 1) * HOP_LENGTH + FFT_SIZE
    output = np.zeros(output_length)
    window_sum = np.zeros(output_length)

    for i in range(num_frames):
        spectrum = (real[i] + 1j * imag[i]) * norm_factor
        frame = np.fft.irfft(spectrum, FFT_SIZE)

        start = i * HOP_LENGTH
        output[start:start + FFT_SIZE] += frame * window
        window_sum[start:start + FFT_SIZE] += window ** 2

    output = np.where(window_sum > 1e-8, output / window_sum, output)
    output = output[pad_size:pad_size + length]
    return output


def prepare_cac_input(left: np.ndarray, right: np.ndarray) -> tuple:
    """Prepare Complex-as-Channels input for model."""
    # Compute STFT for both channels
    left_real, left_imag = compute_stft(left)
    right_real, right_imag = compute_stft(right)

    original_frames = left_real.shape[0]

    def prepare_for_model(real, imag):
        r = real[:, :MODEL_BINS]
        i = imag[:, :MODEL_BINS]
        if r.shape[0] < MODEL_FRAMES:
            pad_frames = MODEL_FRAMES - r.shape[0]
            r = np.pad(r, ((0, pad_frames), (0, 0)), mode='constant')
            i = np.pad(i, ((0, pad_frames), (0, 0)), mode='constant')
        else:
            r = r[:MODEL_FRAMES]
            i = i[:MODEL_FRAMES]
        return r, i

    left_real_m, left_imag_m = prepare_for_model(left_real, left_imag)
    right_real_m, right_imag_m = prepare_for_model(right_real, right_imag)

    # CaC format: [L_real, L_imag, R_real, R_imag]
    cac = np.stack([
        left_real_m.T,
        left_imag_m.T,
        right_real_m.T,
        right_imag_m.T,
    ], axis=0)[np.newaxis, ...]

    return cac.astype(np.float32), original_frames


def run_pytorch_htdemucs(left: np.ndarray, right: np.ndarray):
    """Run PyTorch HTDemucs and extract both branch outputs."""
    print("\n=== Running PyTorch HTDemucs ===")

    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # Load model
    model = get_model('htdemucs_6s')
    model.eval()
    htdemucs = model.models[0]

    # Prepare audio
    audio = np.stack([left, right], axis=0)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, 2, samples]

    # We need to hook into the model to capture intermediate outputs
    # The key is to capture the freq and time branch outputs BEFORE they're combined

    freq_output = None
    time_output = None

    # Hook into the forward pass to capture outputs
    # In HTDemucs, the final combination happens in the forward() method
    # We need to run the decoder and capture both branches

    # Actually, let's use our FullHybridHTDemucs wrapper for this
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from htdemucs_coreml.model_surgery import FullHybridHTDemucs

    hybrid = FullHybridHTDemucs(htdemucs)
    hybrid.eval()

    # Compute STFT using PyTorch
    with torch.no_grad():
        # Get PyTorch's STFT output
        z = htdemucs._spec(audio_tensor)  # Complex [1, 2, 2049, frames]

        # Convert to CaC format
        real = z.real  # [1, 2, 2049, frames]
        imag = z.imag

        # Interleave as [L_real, L_imag, R_real, R_imag]
        cac = torch.cat([
            real[:, 0:1],  # L_real [1, 1, 2049, frames]
            imag[:, 0:1],  # L_imag
            real[:, 1:2],  # R_real
            imag[:, 1:2],  # R_imag
        ], dim=1)  # [1, 4, 2049, frames]

        # Trim to model dimensions
        cac_trimmed = cac[:, :, :MODEL_BINS, :MODEL_FRAMES]

        print(f"  PyTorch STFT shape: {z.shape}")
        print(f"  CaC shape: {cac_trimmed.shape}")
        print(f"  Audio tensor shape: {audio_tensor.shape}")

        # Run hybrid model
        freq_out, time_out = hybrid(cac_trimmed, audio_tensor)

        print(f"  freq_output shape: {freq_out.shape}")
        print(f"  time_output shape: {time_out.shape}")

        freq_output = freq_out.numpy()
        time_output = time_out.numpy()

    return {
        'freq': freq_output,  # [1, 6, 4, 2048, 336]
        'time': time_output,  # [1, 6, 2, samples]
        'stft_frames': z.shape[-1],
        'cac_tensor': cac_trimmed,  # For passing to CoreML
        'audio_tensor': audio_tensor,  # For passing to CoreML
    }


def run_coreml(left: np.ndarray, right: np.ndarray, model_path: str, pytorch_stft_cac=None, audio_tensor=None):
    """Run CoreML model using PyTorch STFT for identical input comparison."""
    print("\n=== Running CoreML ===")

    model = ct.models.MLModel(model_path)

    if pytorch_stft_cac is not None and audio_tensor is not None:
        # Use the exact same STFT from PyTorch for fair comparison
        cac = pytorch_stft_cac.numpy().astype(np.float32)
        raw_audio = audio_tensor.numpy().astype(np.float32)
        original_frames = cac.shape[-1]
        print("  Using PyTorch STFT output for CoreML input (fair comparison)")
    else:
        # Fallback to numpy STFT (may have differences)
        padded_left = left if len(left) >= SEGMENT_SAMPLES else np.pad(left, (0, SEGMENT_SAMPLES - len(left)), mode='reflect')
        padded_right = right if len(right) >= SEGMENT_SAMPLES else np.pad(right, (0, SEGMENT_SAMPLES - len(right)), mode='reflect')
        padded_left = padded_left[:SEGMENT_SAMPLES]
        padded_right = padded_right[:SEGMENT_SAMPLES]

        cac, original_frames = prepare_cac_input(padded_left, padded_right)
        raw_audio = np.stack([padded_left, padded_right], axis=0)[np.newaxis, ...].astype(np.float32)
        print("  Using numpy STFT (may differ from PyTorch)")

    print(f"  spectrogram input shape: {cac.shape}")
    print(f"  raw_audio input shape: {raw_audio.shape}")
    print(f"  spectrogram stats: min={cac.min():.4f}, max={cac.max():.4f}, mean={cac.mean():.4f}")
    print(f"  raw_audio stats: min={raw_audio.min():.4f}, max={raw_audio.max():.4f}, mean={raw_audio.mean():.4f}")

    # Run model
    output = model.predict({
        "spectrogram": cac,
        "raw_audio": raw_audio
    })

    freq_out = output["add_66"]
    time_out = output["add_67"]

    print(f"  freq_output shape: {freq_out.shape}")
    print(f"  time_output shape: {time_out.shape}")

    return {
        'freq': freq_out,  # [1, 6, 4, 2048, 336]
        'time': time_out,  # [1, 6, 2, samples]
        'original_frames': original_frames,
    }


def compare_tensors(name: str, pytorch: np.ndarray, coreml: np.ndarray):
    """Compare two tensors and print diagnostic stats."""
    print(f"\n--- {name} ---")
    print(f"  Shape: PyTorch={pytorch.shape}, CoreML={coreml.shape}")

    # Basic stats
    print(f"  PyTorch: min={pytorch.min():.6f}, max={pytorch.max():.6f}, mean={pytorch.mean():.6f}, std={pytorch.std():.6f}")
    print(f"  CoreML:  min={coreml.min():.6f}, max={coreml.max():.6f}, mean={coreml.mean():.6f}, std={coreml.std():.6f}")

    # Ensure same shape for comparison
    min_shape = tuple(min(p, c) for p, c in zip(pytorch.shape, coreml.shape))
    pt_slice = pytorch[tuple(slice(0, s) for s in min_shape)]
    cm_slice = coreml[tuple(slice(0, s) for s in min_shape)]

    # Difference stats
    diff = pt_slice - cm_slice
    print(f"  Diff: min={diff.min():.6f}, max={diff.max():.6f}, mean={diff.mean():.6f}, std={diff.std():.6f}")
    print(f"  Max abs diff: {np.abs(diff).max():.6f}")
    print(f"  Mean abs diff: {np.abs(diff).mean():.6f}")

    # Correlation
    pt_flat = pt_slice.flatten()
    cm_flat = cm_slice.flatten()
    if pt_flat.std() > 1e-10 and cm_flat.std() > 1e-10:
        correlation = np.corrcoef(pt_flat, cm_flat)[0, 1]
        print(f"  Correlation: {correlation:.6f}")
    else:
        print(f"  Correlation: N/A (constant signal)")

    # SNR (treating PyTorch as reference)
    signal_power = np.mean(pt_flat ** 2)
    noise_power = np.mean(diff.flatten() ** 2)
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
        print(f"  SNR: {snr:.2f} dB")
    else:
        print(f"  SNR: inf dB (perfect match)")

    return {
        'correlation': float(correlation) if 'correlation' in dir() else None,
        'max_abs_diff': float(np.abs(diff).max()),
        'mean_abs_diff': float(np.abs(diff).mean()),
        'snr_db': float(snr) if 'snr' in dir() else float('inf'),
    }


def freq_branch_to_audio(freq_out: np.ndarray, stem_idx: int, original_length: int, original_frames: int):
    """Convert frequency branch output to audio via iSTFT."""
    stereo = []
    for ch in range(2):
        if ch == 0:
            real = freq_out[0, stem_idx, 0].T[:original_frames]  # [frames, bins]
            imag = freq_out[0, stem_idx, 1].T[:original_frames]
        else:
            real = freq_out[0, stem_idx, 2].T[:original_frames]
            imag = freq_out[0, stem_idx, 3].T[:original_frames]

        # Add Nyquist bin
        real_full = np.pad(real, ((0, 0), (0, 1)), mode='constant')
        imag_full = np.pad(imag, ((0, 0), (0, 1)), mode='constant')

        audio = compute_istft(real_full, imag_full, original_length)
        stereo.append(audio)

    return np.array(stereo).T  # [samples, 2]


def time_branch_to_audio(time_out: np.ndarray, stem_idx: int, original_length: int):
    """Extract time branch output as audio."""
    left = time_out[0, stem_idx, 0, :original_length]
    right = time_out[0, stem_idx, 1, :original_length]
    return np.stack([left, right], axis=1)  # [samples, 2]


def main():
    parser = argparse.ArgumentParser(description="Diagnose branch parity")
    parser.add_argument("input", nargs="?", help="Input audio file (optional)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--model", default="/Users/zakkeown/Code/HTDemucsCoreML/Resources/Models/htdemucs_6s.mlpackage")
    parser.add_argument("--stem", type=int, default=0, help="Stem index to save audio for (0=drums)")

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path(__file__).parent / "outputs" / "branch_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BRANCH ISOLATION DIAGNOSTIC")
    print("=" * 70)

    # Load or generate audio
    if args.input:
        print(f"\nLoading audio: {args.input}")
        audio, sr = sf.read(args.input)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=1)
        left = audio[:, 0].astype(np.float32)
        right = audio[:, 1].astype(np.float32)
    else:
        print("\nGenerating synthetic test signal...")
        t = np.linspace(0, SEGMENT_SAMPLES / SAMPLE_RATE, SEGMENT_SAMPLES, dtype=np.float32)
        # Multi-frequency signal that should activate different stems
        left = (0.3 * np.sin(2 * np.pi * 80 * t) +   # Bass-like
                0.2 * np.sin(2 * np.pi * 440 * t) +   # Mid
                0.1 * np.sin(2 * np.pi * 2000 * t))   # High
        right = (0.3 * np.sin(2 * np.pi * 100 * t) +
                 0.2 * np.sin(2 * np.pi * 550 * t) +
                 0.1 * np.sin(2 * np.pi * 2500 * t))

    original_length = min(len(left), SEGMENT_SAMPLES)
    left = left[:SEGMENT_SAMPLES] if len(left) >= SEGMENT_SAMPLES else np.pad(left, (0, SEGMENT_SAMPLES - len(left)), mode='reflect')
    right = right[:SEGMENT_SAMPLES] if len(right) >= SEGMENT_SAMPLES else np.pad(right, (0, SEGMENT_SAMPLES - len(right)), mode='reflect')

    print(f"Audio length: {original_length} samples ({original_length/SAMPLE_RATE:.2f}s)")

    # Save input for reference
    sf.write(str(output_dir / "input.wav"), np.stack([left, right], axis=1)[:original_length], SAMPLE_RATE)

    # Run both engines
    pytorch_results = run_pytorch_htdemucs(left[:original_length], right[:original_length])
    # Use PyTorch's exact STFT output for CoreML to ensure fair comparison
    coreml_results = run_coreml(
        left, right, args.model,
        pytorch_stft_cac=pytorch_results.get('cac_tensor'),
        audio_tensor=pytorch_results.get('audio_tensor')
    )

    # Compare raw branch outputs
    print("\n" + "=" * 70)
    print("NUMERICAL COMPARISON")
    print("=" * 70)

    freq_stats = compare_tensors("Frequency Branch (add_66)", pytorch_results['freq'], coreml_results['freq'])
    time_stats = compare_tensors("Time Branch (add_67)", pytorch_results['time'], coreml_results['time'])

    # Per-stem comparison
    print("\n" + "=" * 70)
    print("PER-STEM COMPARISON (Frequency Branch)")
    print("=" * 70)

    stem_stats = {}
    for stem_idx, stem_name in enumerate(STEM_NAMES):
        pt_stem = pytorch_results['freq'][0, stem_idx]
        cm_stem = coreml_results['freq'][0, stem_idx]
        stem_stats[stem_name] = compare_tensors(f"Stem: {stem_name}", pt_stem, cm_stem)

    # Save audio for selected stem
    stem_idx = args.stem
    stem_name = STEM_NAMES[stem_idx]
    print(f"\n" + "=" * 70)
    print(f"SAVING AUDIO FOR STEM: {stem_name}")
    print("=" * 70)

    original_frames = coreml_results['original_frames']

    # PyTorch freq branch audio
    pt_freq_audio = freq_branch_to_audio(pytorch_results['freq'], stem_idx, original_length, pytorch_results['stft_frames'])
    sf.write(str(output_dir / f"{stem_name}_pytorch_freq.wav"), pt_freq_audio, SAMPLE_RATE)
    print(f"  Saved: {stem_name}_pytorch_freq.wav (RMS: {np.sqrt(np.mean(pt_freq_audio**2)):.6f})")

    # PyTorch time branch audio
    pt_time_audio = time_branch_to_audio(pytorch_results['time'], stem_idx, original_length)
    sf.write(str(output_dir / f"{stem_name}_pytorch_time.wav"), pt_time_audio, SAMPLE_RATE)
    print(f"  Saved: {stem_name}_pytorch_time.wav (RMS: {np.sqrt(np.mean(pt_time_audio**2)):.6f})")

    # PyTorch combined
    pt_combined = pt_freq_audio + pt_time_audio
    sf.write(str(output_dir / f"{stem_name}_pytorch_combined.wav"), pt_combined, SAMPLE_RATE)
    print(f"  Saved: {stem_name}_pytorch_combined.wav (RMS: {np.sqrt(np.mean(pt_combined**2)):.6f})")

    # CoreML freq branch audio
    cm_freq_audio = freq_branch_to_audio(coreml_results['freq'], stem_idx, original_length, original_frames)
    sf.write(str(output_dir / f"{stem_name}_coreml_freq.wav"), cm_freq_audio, SAMPLE_RATE)
    print(f"  Saved: {stem_name}_coreml_freq.wav (RMS: {np.sqrt(np.mean(cm_freq_audio**2)):.6f})")

    # CoreML time branch audio
    cm_time_audio = time_branch_to_audio(coreml_results['time'], stem_idx, original_length)
    sf.write(str(output_dir / f"{stem_name}_coreml_time.wav"), cm_time_audio, SAMPLE_RATE)
    print(f"  Saved: {stem_name}_coreml_time.wav (RMS: {np.sqrt(np.mean(cm_time_audio**2)):.6f})")

    # CoreML combined
    cm_combined = cm_freq_audio + cm_time_audio
    sf.write(str(output_dir / f"{stem_name}_coreml_combined.wav"), cm_combined, SAMPLE_RATE)
    print(f"  Saved: {stem_name}_coreml_combined.wav (RMS: {np.sqrt(np.mean(cm_combined**2)):.6f})")

    # Save summary
    summary = {
        'freq_branch': freq_stats,
        'time_branch': time_stats,
        'per_stem': stem_stats,
        'audio_length': original_length,
        'pytorch_stft_frames': int(pytorch_results['stft_frames']),
        'coreml_original_frames': int(original_frames),
    }

    with open(output_dir / "diagnosis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"\nKey findings:")
    print(f"  Freq branch correlation: {freq_stats.get('correlation', 'N/A')}")
    print(f"  Time branch correlation: {time_stats.get('correlation', 'N/A')}")
    print(f"  Freq branch SNR: {freq_stats.get('snr_db', 'N/A'):.1f} dB")
    print(f"  Time branch SNR: {time_stats.get('snr_db', 'N/A'):.1f} dB")

    if freq_stats.get('correlation', 1) < 0.9:
        print("\n⚠️  WARNING: Frequency branch has low correlation!")
        print("    Possible causes:")
        print("    - STFT input formatting mismatch (CaC channel order)")
        print("    - Model conversion issue in frequency encoder/decoder")
        print("    - Normalization difference")

    if time_stats.get('correlation', 1) < 0.9:
        print("\n⚠️  WARNING: Time branch has low correlation!")
        print("    Possible causes:")
        print("    - Raw audio input formatting mismatch")
        print("    - Model conversion issue in time encoder/decoder")
        print("    - Padding/length handling difference")

    print("\nListen to the audio files to determine which branch is broken.")
    print("Compare *_pytorch_freq.wav vs *_coreml_freq.wav")
    print("Compare *_pytorch_time.wav vs *_coreml_time.wav")


if __name__ == "__main__":
    main()
