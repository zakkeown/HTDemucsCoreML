#!/usr/bin/env python3
"""
Run direct CoreML inference matching Swift implementation.

This provides a true parity reference by using the exact same processing
that our Swift pipeline does:
1. Pad audio to segment size
2. Compute STFT
3. Run CoreML model
4. Compute iSTFT
5. Combine freq + time branches

No shifts, no special windowing, just direct model application.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import coremltools as ct

# Constants matching Swift
SAMPLE_RATE = 44100
SEGMENT_SAMPLES = 343980
FFT_SIZE = 4096
HOP_LENGTH = 1024
NUM_BINS = FFT_SIZE // 2 + 1


def compute_stft(audio: np.ndarray) -> tuple:
    """Compute STFT matching PyTorch conventions (normalized=True)."""
    import math
    pad_size = FFT_SIZE // 2
    padded = np.pad(audio, pad_size, mode='reflect')
    window = np.hanning(FFT_SIZE + 1)[:-1]
    num_frames = (len(padded) - FFT_SIZE) // HOP_LENGTH + 1

    # PyTorch normalized=True divides by sqrt(N)
    norm_factor = 1.0 / math.sqrt(FFT_SIZE)

    real_out = []
    imag_out = []

    for i in range(num_frames):
        start = i * HOP_LENGTH
        frame = padded[start:start + FFT_SIZE] * window
        spectrum = np.fft.rfft(frame) * norm_factor
        real_out.append(spectrum.real)
        imag_out.append(spectrum.imag)

    return np.array(real_out), np.array(imag_out)


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


def pad_audio_to_segment(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad audio using reflect mode."""
    if len(audio) >= target_length:
        return audio[:target_length]
    return np.pad(audio, (0, target_length - len(audio)), mode='reflect')


def process_audio_direct(model, left: np.ndarray, right: np.ndarray) -> dict:
    """Process audio through CoreML model directly, matching Swift pipeline."""
    original_length = len(left)
    model_bins = 2048
    model_frames = 336

    # Pad to segment size
    padded_left = pad_audio_to_segment(left, SEGMENT_SAMPLES)
    padded_right = pad_audio_to_segment(right, SEGMENT_SAMPLES)

    # Compute STFT
    left_real, left_imag = compute_stft(padded_left)
    right_real, right_imag = compute_stft(padded_right)

    # Prepare CaC format for model
    def prepare_for_model(real, imag, target_bins, target_frames):
        r = real[:, :target_bins]
        i = imag[:, :target_bins]
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

    cac = np.stack([
        left_real_model.T,
        left_imag_model.T,
        right_real_model.T,
        right_imag_model.T,
    ], axis=0)[np.newaxis, ...]

    raw_audio = np.stack([padded_left, padded_right], axis=0)[np.newaxis, ...]

    # Run model
    output = model.predict({
        "spectrogram": cac.astype(np.float32),
        "raw_audio": raw_audio.astype(np.float32)
    })

    freq_out = output["add_66"]
    time_out = output["add_67"]

    # Process each stem
    stem_names = ["drums", "bass", "other", "vocals", "guitar", "piano"]
    results = {}
    original_frames = left_real.shape[0]

    for stem_idx, stem_name in enumerate(stem_names):
        stereo_output = []

        for channel_idx in range(2):
            # Extract freq output
            if channel_idx == 0:
                stem_real = freq_out[0, stem_idx, 0].T[:original_frames]
                stem_imag = freq_out[0, stem_idx, 1].T[:original_frames]
            else:
                stem_real = freq_out[0, stem_idx, 2].T[:original_frames]
                stem_imag = freq_out[0, stem_idx, 3].T[:original_frames]

            # Add Nyquist bin
            stem_real_full = np.pad(stem_real, ((0, 0), (0, 1)), mode='constant')
            stem_imag_full = np.pad(stem_imag, ((0, 0), (0, 1)), mode='constant')

            # iSTFT
            freq_audio = compute_istft(stem_real_full, stem_imag_full, original_length)

            # Time output
            time_audio = time_out[0, stem_idx, channel_idx, :original_length]

            # Combine
            combined = freq_audio + time_audio
            stereo_output.append(combined)

        results[stem_name] = np.array(stereo_output).T  # (samples, channels)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run direct CoreML separation")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--model", default="/Users/zakkeown/Code/HTDemucsCoreML/Resources/Models/htdemucs_6s.mlpackage",
                       help="Model path")

    args = parser.parse_args()

    if args.output is None:
        script_dir = Path(__file__).parent
        args.output = script_dir / "outputs"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = ct.models.MLModel(args.model)

    print(f"Loading audio: {args.input}")
    audio, sr = sf.read(args.input)
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=1)

    left = audio[:, 0].astype(np.float32)
    right = audio[:, 1].astype(np.float32)

    print(f"Processing {len(left)} samples...")
    results = process_audio_direct(model, left, right)

    print("Saving stems...")
    for stem_name, stem_audio in results.items():
        output_path = output_dir / f"{stem_name}_coreml_direct.wav"
        sf.write(str(output_path), stem_audio, SAMPLE_RATE)
        rms = np.sqrt(np.mean(stem_audio**2))
        print(f"  ✓ {stem_name}: RMS={rms:.6f}")

    print("\n✓ Direct CoreML separation complete")


if __name__ == "__main__":
    main()
