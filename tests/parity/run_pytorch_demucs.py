#!/usr/bin/env python3
"""Run PyTorch HTDemucs on test audio for parity comparison."""

import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf

def load_audio(path: str, sr: int = 44100):
    """Load audio and ensure correct format."""
    audio, file_sr = sf.read(path)

    # Convert to torch tensor and transpose to (channels, samples)
    audio = torch.from_numpy(audio.T).float()

    # Resample if needed
    if file_sr != sr:
        import torchaudio
        audio = torchaudio.functional.resample(audio, file_sr, sr)

    # Mono to stereo if needed
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)

    return audio

def separate_with_pytorch(audio_path: str, output_dir: str, model_name: str = "htdemucs_6s"):
    """Separate audio using PyTorch HTDemucs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = get_model(model_name)
    model.eval()

    print(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path)

    print("Running separation...")
    with torch.no_grad():
        sources = apply_model(
            model,
            audio.unsqueeze(0),  # Add batch dimension
            device="cpu",
            shifts=1,
            split=True,
            overlap=0.25
        )

    sources = sources.squeeze(0)  # Remove batch dimension

    # htdemucs_6s outputs: drums, bass, other, vocals, guitar, piano
    stem_names = ["drums", "bass", "other", "vocals", "guitar", "piano"]

    print("Saving stems...")
    for i, stem_name in enumerate(stem_names):
        stem_audio = sources[i].numpy().T  # Transpose to (samples, channels)
        output_path = output_dir / f"{stem_name}_pytorch.wav"
        sf.write(output_path, stem_audio, 44100)
        print(f"  ✓ {output_path}")

    return sources

def main():
    parser = argparse.ArgumentParser(description="Run PyTorch HTDemucs separation")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--model", default="htdemucs_6s", help="Model name")

    args = parser.parse_args()

    # Default output to outputs/ in the parity directory
    if args.output is None:
        script_dir = Path(__file__).parent
        args.output = script_dir / "outputs"

    separate_with_pytorch(args.input, args.output, args.model)
    print("\n✓ PyTorch separation complete")

if __name__ == "__main__":
    main()
