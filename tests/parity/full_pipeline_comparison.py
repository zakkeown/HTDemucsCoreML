#!/usr/bin/env python3
"""
Full pipeline comparison: PyTorch HTDemucs vs CoreML Swift pipeline.

Generates ground truth for the entire separation pipeline on a real audio sample.
"""

import numpy as np
from pathlib import Path
import json
import coremltools as ct
import soundfile as sf

# Constants
SAMPLE_RATE = 44100
SEGMENT_SAMPLES = 343980
FFT_SIZE = 4096
HOP_LENGTH = 1024


def compute_stft_pytorch_compatible(audio: np.ndarray) -> tuple:
    """
    Compute STFT matching PyTorch conventions (normalized=True).

    Returns:
        (real, imag): Each [num_frames, num_bins=2049]
    """
    import math

    # Center padding (reflect mode)
    pad_size = FFT_SIZE // 2
    padded = np.pad(audio, pad_size, mode='reflect')

    # Hann window (periodic)
    window = np.hanning(FFT_SIZE + 1)[:-1]

    # PyTorch normalized=True divides by sqrt(N)
    norm_factor = 1.0 / math.sqrt(FFT_SIZE)

    # Compute frames
    num_frames = (len(padded) - FFT_SIZE) // HOP_LENGTH + 1

    real_out = []
    imag_out = []

    for i in range(num_frames):
        start = i * HOP_LENGTH
        frame = padded[start:start + FFT_SIZE] * window
        spectrum = np.fft.rfft(frame) * norm_factor
        real_out.append(spectrum.real)
        imag_out.append(spectrum.imag)

    return np.array(real_out), np.array(imag_out)


def compute_istft_pytorch_compatible(real: np.ndarray, imag: np.ndarray, length: int) -> np.ndarray:
    """
    Compute iSTFT matching PyTorch conventions (normalized=True).
    """
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


def pad_audio_to_segment(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad audio to target length using reflect mode."""
    if len(audio) >= target_length:
        return audio[:target_length]

    pad_needed = target_length - len(audio)
    return np.pad(audio, (0, pad_needed), mode='reflect')


def save_array(arr: np.ndarray, path: Path, name: str):
    """Save array as binary float32 and metadata."""
    arr = arr.astype(np.float32)
    bin_path = path / f"{name}.bin"
    meta_path = path / f"{name}.json"

    arr.flatten().tofile(bin_path)

    with open(meta_path, 'w') as f:
        json.dump({
            "shape": list(arr.shape),
            "dtype": "float32",
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }, f, indent=2)

    print(f"  Saved {name}: shape={arr.shape}, range=[{arr.min():.6f}, {arr.max():.6f}]")


def main():
    output_dir = Path("/tmp/full_pipeline_comparison")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Full Pipeline Comparison: Python Ground Truth")
    print("=" * 60)

    # Load test audio (use simple_mix if available, otherwise generate)
    fixtures_dir = Path(__file__).parent / "fixtures"
    test_audio_path = fixtures_dir / "simple_mix.wav"

    if test_audio_path.exists():
        print(f"\n1. Loading test audio from {test_audio_path}...")
        audio, sr = sf.read(str(test_audio_path))
        if len(audio.shape) == 1:
            # Mono to stereo
            audio = np.stack([audio, audio], axis=1)
        left = audio[:, 0].astype(np.float32)
        right = audio[:, 1].astype(np.float32)
        print(f"   Loaded {len(left)} samples at {sr} Hz")
    else:
        print("\n1. Generating synthetic test audio...")
        duration = SEGMENT_SAMPLES / SAMPLE_RATE
        t = np.linspace(0, duration, SEGMENT_SAMPLES, dtype=np.float32)

        # Multi-frequency stereo signal
        left = 0.3 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 1000 * t)
        right = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.sin(2 * np.pi * 550 * t) + 0.1 * np.sin(2 * np.pi * 1200 * t)

        left = left.astype(np.float32)
        right = right.astype(np.float32)
        print(f"   Generated {len(left)} samples")

    # Truncate to segment size
    original_length = min(len(left), SEGMENT_SAMPLES)
    left = left[:original_length]
    right = right[:original_length]

    save_array(left, output_dir, "input_left")
    save_array(right, output_dir, "input_right")
    print(f"   Original length: {original_length}")

    # Pad to segment size
    print("\n2. Padding audio to segment size...")
    padded_left = pad_audio_to_segment(left, SEGMENT_SAMPLES)
    padded_right = pad_audio_to_segment(right, SEGMENT_SAMPLES)

    save_array(padded_left, output_dir, "padded_left")
    save_array(padded_right, output_dir, "padded_right")

    # Compute STFT
    print("\n3. Computing STFT on padded audio...")
    left_real, left_imag = compute_stft_pytorch_compatible(padded_left)
    right_real, right_imag = compute_stft_pytorch_compatible(padded_right)

    print(f"   STFT shape: [{left_real.shape[0]}, {left_real.shape[1]}]")
    save_array(left_real, output_dir, "stft_left_real")
    save_array(left_imag, output_dir, "stft_left_imag")
    save_array(right_real, output_dir, "stft_right_real")
    save_array(right_imag, output_dir, "stft_right_imag")

    # Prepare model input
    print("\n4. Preparing model input (CaC format)...")
    model_bins = 2048
    model_frames = 336

    # Pad/trim to model dimensions
    def prepare_for_model(real, imag, target_bins, target_frames):
        # Take first target_bins (drop Nyquist)
        r = real[:, :target_bins]
        i = imag[:, :target_bins]

        # Pad or trim frames
        if r.shape[0] < target_frames:
            pad_frames = target_frames - r.shape[0]
            r = np.pad(r, ((0, pad_frames), (0, 0)), mode='constant')
            i = np.pad(i, ((0, pad_frames), (0, 0)), mode='constant')
        else:
            r = r[:target_frames]
            i = i[:target_frames]

        return r, i

    left_real_model, left_imag_model = prepare_for_model(left_real, left_imag, model_bins, model_frames)
    right_real_model, right_imag_model = prepare_for_model(right_real, right_imag, model_bins, model_frames)

    # CaC format: [L_real, L_imag, R_real, R_imag] -> [4, bins, frames]
    cac = np.stack([
        left_real_model.T,
        left_imag_model.T,
        right_real_model.T,
        right_imag_model.T,
    ], axis=0)[np.newaxis, ...]

    raw_audio = np.stack([padded_left, padded_right], axis=0)[np.newaxis, ...]

    save_array(cac, output_dir, "model_input_spec")
    save_array(raw_audio, output_dir, "model_input_audio")

    # Run CoreML model
    print("\n5. Running CoreML inference...")
    model_path = Path("/Users/zakkeown/Code/HTDemucsCoreML/Resources/Models/htdemucs_6s.mlpackage")
    mlmodel = ct.models.MLModel(str(model_path))

    output = mlmodel.predict({
        "spectrogram": cac.astype(np.float32),
        "raw_audio": raw_audio.astype(np.float32)
    })

    freq_out = output["add_66"]  # [1, 6, 4, 2048, 336]
    time_out = output["add_67"]  # [1, 6, 2, 343980]

    save_array(freq_out, output_dir, "model_output_freq")
    save_array(time_out, output_dir, "model_output_time")

    print(f"   freq_out shape: {freq_out.shape}")
    print(f"   time_out shape: {time_out.shape}")

    # Process each stem
    print("\n6. Processing stems through iSTFT + time combination...")

    stem_names = ["drums", "bass", "other", "vocals", "guitar", "piano"]

    for stem_idx in range(6):
        stem_name = stem_names[stem_idx]
        print(f"\n   --- {stem_name} (stem {stem_idx}) ---")

        for channel_idx in range(2):
            channel_name = "left" if channel_idx == 0 else "right"

            # Extract freq output for this stem/channel
            # freq_out shape: [1, 6, 4, 2048, 336]
            # CaC: [L_real, L_imag, R_real, R_imag]
            if channel_idx == 0:
                stem_real = freq_out[0, stem_idx, 0]  # L_real [2048, 336]
                stem_imag = freq_out[0, stem_idx, 1]  # L_imag [2048, 336]
            else:
                stem_real = freq_out[0, stem_idx, 2]  # R_real [2048, 336]
                stem_imag = freq_out[0, stem_idx, 3]  # R_imag [2048, 336]

            # Transpose to [frames, bins]
            stem_real_t = stem_real.T  # [336, 2048]
            stem_imag_t = stem_imag.T

            # Trim to original frame count
            original_frames = left_real.shape[0]
            stem_real_t = stem_real_t[:original_frames]
            stem_imag_t = stem_imag_t[:original_frames]

            # Add Nyquist bin (zero)
            stem_real_full = np.pad(stem_real_t, ((0, 0), (0, 1)), mode='constant')
            stem_imag_full = np.pad(stem_imag_t, ((0, 0), (0, 1)), mode='constant')

            # iSTFT
            freq_audio = compute_istft_pytorch_compatible(stem_real_full, stem_imag_full, original_length)

            # Get time output for this stem/channel
            time_audio = time_out[0, stem_idx, channel_idx, :original_length]

            # Combine
            combined = freq_audio + time_audio

            if channel_idx == 0:  # Only save left channel for space
                save_array(freq_audio, output_dir, f"{stem_name}_left_freq")
                save_array(time_audio, output_dir, f"{stem_name}_left_time")
                save_array(combined, output_dir, f"{stem_name}_left_combined")

    # Save metadata
    print("\n7. Saving metadata...")
    metadata = {
        "original_length": original_length,
        "segment_samples": SEGMENT_SAMPLES,
        "stft_frames": left_real.shape[0],
        "model_frames": model_frames,
        "model_bins": model_bins,
        "stems": stem_names,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Full pipeline ground truth generation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Print sample values for quick verification
    print("\nSample values (bass, left channel):")
    bass_combined = np.fromfile(output_dir / "bass_left_combined.bin", dtype=np.float32)
    print(f"  Combined first 10: {bass_combined[:10]}")
    print(f"  Combined RMS: {np.sqrt(np.mean(bass_combined**2)):.6f}")


if __name__ == "__main__":
    main()
