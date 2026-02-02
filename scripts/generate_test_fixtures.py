#!/usr/bin/env python3
"""Generate test fixtures for HTDemucs CoreML validation."""

import torch
import numpy as np
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model

def generate_fixtures():
    """Generate golden output fixtures from PyTorch Demucs."""
    fixture_dir = Path("test_fixtures")
    fixture_dir.mkdir(exist_ok=True)

    print("Loading htdemucs_6s model...")
    try:
        model = get_model('htdemucs_6s')
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have an internet connection for the first run.")
        print("The model will be downloaded to ~/.cache/torch/hub/")
        raise

    # Generate test audio: 10-second stereo at 44.1kHz
    sample_rate = 44100
    duration = 10.0
    num_samples = int(sample_rate * duration)

    # Set random seed for reproducible fixtures
    torch.manual_seed(42)

    # Create diverse test signals
    test_cases = {
        'silence': torch.zeros(2, num_samples),
        'sine_440hz': generate_sine_wave(440, sample_rate, num_samples),
        'white_noise': torch.randn(2, num_samples) * 0.1,
    }

    for name, audio in test_cases.items():
        print(f"\nProcessing {name}...")

        # Save input audio
        np.save(fixture_dir / f"{name}_input.npy", audio.numpy())

        # Run full Demucs model
        with torch.no_grad():
            # apply_model expects (batch, channels, samples) and returns (batch, sources, channels, samples)
            audio_batch = audio.unsqueeze(0)
            sources = apply_model(model, audio_batch, shifts=0, split=True, overlap=0.25)

        # Save outputs
        np.save(fixture_dir / f"{name}_output.npy", sources.numpy())

        print(f"  Saved {name}: input shape {audio.shape}, output shape {sources.shape}")

    # Save model metadata
    metadata = {
        'sample_rate': sample_rate,
        'duration': duration,
        'num_samples': num_samples,
        'sources': ['drums', 'bass', 'vocals', 'other', 'piano', 'guitar'],
        'model': 'htdemucs_6s',
    }
    np.save(fixture_dir / "metadata.npy", metadata)

    print(f"\nTest fixtures saved to {fixture_dir}")

def generate_sine_wave(freq, sample_rate, num_samples):
    """Generate stereo sine wave."""
    t = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    # Stereo: slightly different phase for L/R
    left = torch.sin(2 * np.pi * freq * t)
    right = torch.sin(2 * np.pi * freq * t + 0.1)
    return torch.stack([left, right])

if __name__ == '__main__':
    generate_fixtures()
