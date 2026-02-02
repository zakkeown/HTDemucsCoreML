#!/usr/bin/env python3
"""Generate golden STFT outputs from PyTorch for Swift validation."""

import numpy as np
import torch
import argparse
from pathlib import Path


def generate_test_audio(test_name: str, sample_rate: int = 44100) -> np.ndarray:
    """Generate test audio signals."""
    if test_name == "silence":
        return np.zeros(88200, dtype=np.float32)

    elif test_name == "sine_440hz":
        duration = 2.0
        samples = int(duration * sample_rate)
        t = np.arange(samples) / sample_rate
        freq = 440.0
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    elif test_name == "white_noise":
        rng = np.random.RandomState(42)  # Fixed seed
        return rng.uniform(-1, 1, 88200).astype(np.float32)

    else:
        raise ValueError(f"Unknown test: {test_name}")


def compute_pytorch_stft(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute STFT using PyTorch (reference implementation)."""
    audio_torch = torch.from_numpy(audio)

    # Match Swift parameters
    stft_complex = torch.stft(
        audio_torch,
        n_fft=4096,
        hop_length=1024,
        window=torch.hann_window(4096),
        return_complex=True,
        center=False  # Match Swift (no padding)
    )

    # Extract real and imaginary parts
    real = stft_complex.real.numpy().astype(np.float32)
    imag = stft_complex.imag.numpy().astype(np.float32)

    # Transpose to [numFrames, numBins] (Swift format)
    real = real.T
    imag = imag.T

    return real, imag


def main():
    parser = argparse.ArgumentParser(description="Generate golden STFT fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Resources/GoldenOutputs"),
        help="Output directory for fixtures"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = ["silence", "sine_440hz", "white_noise"]

    for test_name in test_cases:
        print(f"Generating {test_name}...")

        # Generate audio
        audio = generate_test_audio(test_name)

        # Compute PyTorch STFT (golden reference)
        real, imag = compute_pytorch_stft(audio)

        # Save as NPZ
        output_path = args.output_dir / f"{test_name}.npz"
        np.savez(
            output_path,
            audio=audio,
            stft_real=real,
            stft_imag=imag
        )

        print(f"  Audio: {audio.shape}, STFT: {real.shape}")
        print(f"  Saved to {output_path}")

    print(f"\n✓ Generated {len(test_cases)} golden fixtures")


if __name__ == "__main__":
    main()
