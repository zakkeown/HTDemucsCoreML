#!/usr/bin/env python3
"""Generate test audio fixtures for parity testing."""

import numpy as np
import soundfile as sf
from pathlib import Path

def generate_simple_mix():
    """Generate a simple 5-second stereo mix with known components."""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))

    # Generate simple stems
    drums = np.sin(2 * np.pi * 80 * t) * 0.3  # Bass drum-like
    bass = np.sin(2 * np.pi * 110 * t) * 0.4   # Bass note
    vocals = np.sin(2 * np.pi * 440 * t) * 0.5 # A4 note
    other = np.sin(2 * np.pi * 220 * t) * 0.2  # Background

    # Create stereo versions
    drums_stereo = np.stack([drums, drums])
    bass_stereo = np.stack([bass, bass])
    vocals_stereo = np.stack([vocals, vocals])
    other_stereo = np.stack([other, other])

    # Mix all stems
    mix = drums_stereo + bass_stereo + vocals_stereo + other_stereo
    mix = mix / np.max(np.abs(mix)) * 0.9  # Normalize

    return mix.T, drums_stereo.T, bass_stereo.T, vocals_stereo.T, other_stereo.T, sr

def main():
    fixtures_dir = Path(__file__).parent
    fixtures_dir.mkdir(exist_ok=True)

    print("Generating test fixtures...")

    # Generate simple synthetic mix
    mix, drums, bass, vocals, other, sr = generate_simple_mix()

    sf.write(fixtures_dir / "simple_mix.wav", mix, sr)
    sf.write(fixtures_dir / "simple_mix_drums.wav", drums, sr)
    sf.write(fixtures_dir / "simple_mix_bass.wav", bass, sr)
    sf.write(fixtures_dir / "simple_mix_vocals.wav", vocals, sr)
    sf.write(fixtures_dir / "simple_mix_other.wav", other, sr)

    print("✓ Generated simple_mix.wav and ground truth stems")
    print("\nNote: For real validation, add:")
    print("  - MUSDB18 test tracks")
    print("  - Various genres and quality levels")
    print("  - Real-world recordings")

if __name__ == "__main__":
    main()
