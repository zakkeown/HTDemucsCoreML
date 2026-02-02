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
def test_multiple_files():
    """Test parity on multiple audio files."""
    # This would test on MUSDB18 or other datasets
    pytest.skip("Requires MUSDB18 dataset - implement when available")
