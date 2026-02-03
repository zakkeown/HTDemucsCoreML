#!/usr/bin/env python3
"""Compute BSS metrics (SDR/SIR/SAR) for parity testing."""

import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import mir_eval
import pandas as pd

def load_stem(path: Path) -> np.ndarray:
    """Load audio stem."""
    audio, sr = sf.read(path)
    if sr != 44100:
        raise ValueError(f"Expected 44.1kHz, got {sr}Hz")
    return audio

def compute_bss_metrics(reference: np.ndarray, estimated: np.ndarray) -> dict:
    """Compute SDR, SIR, SAR metrics.

    Args:
        reference: Ground truth stem (samples, channels)
        estimated: Estimated stem (samples, channels)

    Returns:
        Dictionary with sdr, sir, sar values
    """
    # Ensure same length
    min_len = min(len(reference), len(estimated))
    reference = reference[:min_len]
    estimated = estimated[:min_len]

    # Compute metrics per channel then average
    sdrs, sirs, sars = [], [], []

    for ch in range(reference.shape[1]):
        ref_ch = reference[:, ch]
        est_ch = estimated[:, ch]

        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            ref_ch[np.newaxis, :],
            est_ch[np.newaxis, :]
        )

        sdrs.append(sdr[0])
        sirs.append(sir[0])
        sars.append(sar[0])

    return {
        'sdr': np.mean(sdrs),
        'sir': np.mean(sirs),
        'sar': np.mean(sars)
    }

def compare_outputs(output_dir: Path, reference_dir: Path = None) -> pd.DataFrame:
    """Compare PyTorch and CoreML outputs.

    Args:
        output_dir: Directory with *_pytorch.wav and *_coreml.wav files
        reference_dir: Optional directory with ground truth stems

    Returns:
        DataFrame with metrics per stem
    """
    stems = ["drums", "bass", "vocals", "other", "piano", "guitar"]
    results = []

    for stem in stems:
        pytorch_path = output_dir / f"{stem}_pytorch.wav"
        coreml_path = output_dir / f"{stem}_coreml.wav"

        if not pytorch_path.exists() or not coreml_path.exists():
            print(f"⚠ Skipping {stem} - files not found")
            continue

        pytorch_audio = load_stem(pytorch_path)
        coreml_audio = load_stem(coreml_path)

        # Compute metrics: CoreML vs PyTorch (PyTorch as reference)
        metrics = compute_bss_metrics(pytorch_audio, coreml_audio)

        results.append({
            'stem': stem,
            'sdr': metrics['sdr'],
            'sir': metrics['sir'],
            'sar': metrics['sar']
        })

        print(f"{stem:8s}: SDR={metrics['sdr']:6.2f} dB, "
              f"SIR={metrics['sir']:6.2f} dB, SAR={metrics['sar']:6.2f} dB")

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Compute parity metrics")
    parser.add_argument("--output-dir", default=None,
                        help="Directory with separated stems")
    parser.add_argument("--save-csv", help="Save results to CSV")
    parser.add_argument("--reference", help="Reference audio file (ground truth)")
    parser.add_argument("--estimated", help="Estimated audio file (model output)")
    parser.add_argument("--stem", help="Stem name (for single file comparison)")

    args = parser.parse_args()

    # Single file comparison mode (for MUSDB18 testing)
    if args.reference and args.estimated:
        ref_audio = load_stem(Path(args.reference))
        est_audio = load_stem(Path(args.estimated))
        metrics = compute_bss_metrics(ref_audio, est_audio)

        stem_name = args.stem or "unknown"
        print(f"{stem_name}: SDR={metrics['sdr']:.2f} dB, "
              f"SIR={metrics['sir']:.2f} dB, SAR={metrics['sar']:.2f} dB")
        return 0

    # Directory comparison mode (CoreML vs PyTorch)
    # Default to outputs/ in script directory
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "outputs"
    else:
        output_dir = Path(args.output_dir)

    print("Computing CoreML vs PyTorch metrics...")
    print("=" * 60)

    df = compare_outputs(output_dir)

    print("=" * 60)
    print(f"\nAverage SDR: {df['sdr'].mean():.2f} dB")
    print(f"Average SIR: {df['sir'].mean():.2f} dB")
    print(f"Average SAR: {df['sar'].mean():.2f} dB")

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\n✓ Saved results to {args.save_csv}")

    # Check if within acceptable threshold
    threshold = 2.0  # dB
    if df['sdr'].mean() >= -threshold:
        print(f"\n✓ PASS: SDR within {threshold} dB threshold")
        return 0
    else:
        print(f"\n✗ FAIL: SDR below {threshold} dB threshold")
        return 1

if __name__ == "__main__":
    exit(main())
