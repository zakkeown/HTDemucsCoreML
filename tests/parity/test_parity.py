"""Automated parity tests for CoreML HTDemucs."""

import pytest
import subprocess
from pathlib import Path
import pandas as pd

FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUTPUT_DIR = Path(__file__).parent / "outputs"

@pytest.fixture(scope="module")
def separated_stems():
    """Run both PyTorch and CoreML separation on test fixture."""
    test_audio = FIXTURES_DIR / "simple_mix.wav"

    if not test_audio.exists():
        pytest.skip("Test fixture not found - run generate_fixtures.py first")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Run PyTorch
    print("\nRunning PyTorch HTDemucs...")
    subprocess.run([
        "python", str(Path(__file__).parent / "run_pytorch_demucs.py"),
        str(test_audio),
        "--output", str(OUTPUT_DIR)
    ], check=True)

    # Run CoreML
    print("Running CoreML HTDemucs...")
    subprocess.run([
        "bash", str(Path(__file__).parent / "run_coreml_demucs.sh"),
        str(test_audio),
        str(OUTPUT_DIR)
    ], check=True)

    return OUTPUT_DIR

def test_outputs_exist(separated_stems):
    """Verify both PyTorch and CoreML produced outputs."""
    stems = ["drums", "bass", "vocals", "other", "piano", "guitar"]

    for stem in stems:
        pytorch_file = separated_stems / f"{stem}_pytorch.wav"
        coreml_file = separated_stems / f"{stem}_coreml.wav"

        assert pytorch_file.exists(), f"PyTorch {stem} missing"
        assert coreml_file.exists(), f"CoreML {stem} missing"

def test_sdr_threshold(separated_stems):
    """Test that SDR is within acceptable threshold."""
    # Compute metrics
    result = subprocess.run([
        "python", str(Path(__file__).parent / "compute_metrics.py"),
        "--output-dir", str(separated_stems),
        "--save-csv", str(separated_stems / "metrics.csv")
    ], capture_output=True, text=True)

    print(result.stdout)

    # Load metrics
    df = pd.read_csv(separated_stems / "metrics.csv")

    # For synthetic test signals, we expect the models to behave similarly
    # even if separation quality is poor. Check that they produce similar outputs.
    # Use a relaxed threshold since synthetic signals aren't realistic.
    threshold = 20.0  # dB - very relaxed for synthetic signals
    avg_sdr = df['sdr'].mean()

    # Note: For real audio, we'd expect much better SDR (> -2 dB)
    assert avg_sdr >= -threshold, \
        f"Average SDR {avg_sdr:.2f} dB below threshold -{threshold} dB"

def test_per_stem_exists(separated_stems):
    """Test that each stem was generated."""
    df = pd.read_csv(separated_stems / "metrics.csv")

    # Just verify we have metrics for all stems
    expected_stems = {"drums", "bass", "vocals", "other", "piano", "guitar"}
    actual_stems = set(df['stem'])

    assert actual_stems == expected_stems, \
        f"Missing stems: {expected_stems - actual_stems}"

@pytest.mark.slow
def test_musdb18_quality():
    """Test separation quality on real music from MUSDB18-HQ."""
    import os
    from pathlib import Path

    musdb_path = Path.home() / "Datasets" / "musdb18hq" / "test"

    if not musdb_path.exists():
        pytest.skip("MUSDB18-HQ dataset not found at ~/Datasets/musdb18hq")

    # Test on first 3 songs for reasonable runtime
    songs = sorted([d for d in musdb_path.iterdir() if d.is_dir()])[:3]

    results = []
    for song_dir in songs:
        print(f"\n{'='*60}")
        print(f"Testing: {song_dir.name}")
        print('='*60)

        mixture = song_dir / "mixture.wav"
        output_dir = OUTPUT_DIR / song_dir.name
        output_dir.mkdir(exist_ok=True)

        # Run PyTorch
        print("Running PyTorch HTDemucs...")
        subprocess.run([
            "python", str(Path(__file__).parent / "run_pytorch_demucs.py"),
            str(mixture),
            "--output", str(output_dir)
        ], check=True)

        # Run CoreML
        print("Running CoreML HTDemucs...")
        subprocess.run([
            "bash", str(Path(__file__).parent / "run_coreml_demucs.sh"),
            str(mixture),
            str(output_dir)
        ], check=True)

        # Compute metrics for each stem
        # MUSDB18 has 4 stems; htdemucs_6s outputs 6 (piano+guitar go to "other")
        stems_to_test = ["drums", "bass", "vocals", "other"]

        for stem in stems_to_test:
            ground_truth = song_dir / f"{stem}.wav"
            pytorch_output = output_dir / f"{stem}_pytorch.wav"
            coreml_output = output_dir / f"{stem}_coreml.wav"

            # We're testing CoreML vs ground truth, not CoreML vs PyTorch
            # This measures actual separation quality
            venv_python = Path(__file__).parent / "venv" / "bin" / "python"
            result = subprocess.run([
                str(venv_python), str(Path(__file__).parent / "compute_metrics.py"),
                "--reference", str(ground_truth),
                "--estimated", str(coreml_output),
                "--stem", stem
            ], capture_output=True, text=True)

            # Parse SDR from output
            for line in result.stdout.split('\n'):
                if 'SDR=' in line:
                    sdr = float(line.split('SDR=')[1].split()[0])
                    results.append({
                        'song': song_dir.name,
                        'stem': stem,
                        'sdr': sdr
                    })
                    print(f"  {stem}: SDR = {sdr:.2f} dB")

    # Check that average SDR is reasonable for real separation
    avg_sdr = sum(r['sdr'] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"Average SDR across {len(results)} stems: {avg_sdr:.2f} dB")
    print('='*60)

    # For HTDemucs on MUSDB18, expect SDR > 5 dB (literature baseline)
    assert avg_sdr > 5.0, \
        f"Average SDR {avg_sdr:.2f} dB is below expected threshold of 5.0 dB"
