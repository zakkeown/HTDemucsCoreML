# Parity Testing Guide

## Overview

Parity testing validates that the CoreML implementation of HTDemucs produces separation quality equivalent to the original PyTorch implementation.

## Metrics

- **SDR** (Signal-to-Distortion Ratio): Overall quality of separation
- **SIR** (Signal-to-Interference Ratio): How well sources are separated
- **SAR** (Signal-to-Artifacts Ratio): Amount of artifacts introduced

Higher values (in dB) indicate better performance. Industry standards:
- SDR > 10 dB: Excellent
- SDR 5-10 dB: Good
- SDR < 5 dB: Poor

## Quick Start

```bash
# 1. Set up Python environment
cd tests/parity
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate test fixtures
python fixtures/generate_fixtures.py

# 3. Build Swift package
cd ../..
swift build -c release

# 4. Run parity tests
cd tests/parity
pytest test_parity.py -v

# 5. Generate report
python generate_report.py
open outputs/parity_report.html
```

## Testing with Real Audio

```bash
# 1. Add your audio file
cp /path/to/song.wav tests/parity/fixtures/real_song.wav

# 2. Run PyTorch separation
cd tests/parity
source venv/bin/activate
python run_pytorch_demucs.py fixtures/real_song.wav

# 3. Run CoreML separation
./run_coreml_demucs.sh fixtures/real_song.wav

# 4. Compute metrics
python compute_metrics.py

# 5. Generate report
python generate_report.py
open outputs/parity_report.html
```

## Testing with MUSDB18

```bash
# Download MUSDB18 dataset
# https://sigsep.github.io/datasets/musdb.html

# Run on all test tracks
for track in /path/to/musdb18/test/*.wav; do
    echo "Testing: $track"
    python run_pytorch_demucs.py "$track"
    ./run_coreml_demucs.sh "$track"
    python compute_metrics.py
done

# Aggregate results
python aggregate_results.py
```

## Expected Results

CoreML should match PyTorch within 1-2 dB:

| Metric | Expected Range |
|--------|---------------|
| SDR    | > 8 dB        |
| SIR    | > 10 dB       |
| SAR    | > 8 dB        |

If results are significantly lower:
- Check audio preprocessing (sample rate, channels)
- Verify model conversion accuracy
- Compare intermediate outputs (STFT, spectrograms)

## Troubleshooting

**SDR < 5 dB:**
- Verify model file integrity
- Check for silent or corrupted stems
- Ensure sample rate is 44.1kHz

**Memory errors:**
- Reduce chunk size in SeparationPipeline
- Test with shorter audio clips
- Check available RAM

**Import errors:**
- Verify Python environment activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

## Advanced Usage

### Benchmark Against Other Models

```bash
# Add comparison with Spleeter
pip install spleeter
python compare_spleeter.py

# Add comparison with Open-Unmix
pip install openunmix
python compare_openunmix.py
```

### Profile Performance

```bash
# Time PyTorch separation
time python run_pytorch_demucs.py fixtures/test.wav

# Time CoreML separation
time ./run_coreml_demucs.sh fixtures/test.wav

# Compare throughput
python benchmark_performance.py
```

### Generate Spectrograms

```bash
# Visualize differences
python visualize_spectrograms.py \
    --pytorch outputs/vocals_pytorch.wav \
    --coreml outputs/vocals_coreml.wav
```

## CI Integration

Parity tests run automatically on:
- Every commit to main
- All pull requests
- Manual workflow dispatch

View results: GitHub Actions > Parity Tests > Artifacts

## Understanding the Results

### Synthetic Test Signals

The default test fixture uses synthetic sine waves. HTDemucs was trained on real music, so:
- Absolute SDR values will be low (often negative)
- The key validation is that CoreML and PyTorch produce **similar** outputs
- This confirms correct model architecture and weight porting

### Real Music Tracks

When testing with real music:
- SDR should be > 5 dB for vocals and drums
- SDR may be lower for other, piano, and guitar (more challenging)
- CoreML and PyTorch should match within 1-2 dB

### Interpreting Differences

If CoreML differs from PyTorch by more than 2 dB:
1. Check STFT implementation matches exactly
2. Verify model input/output shapes
3. Compare intermediate activations
4. Check for numerical precision issues
5. Validate audio encoding/decoding pipeline

## Manual Testing

For detailed analysis, run each step manually:

```bash
# 1. Generate fixtures
python fixtures/generate_fixtures.py

# 2. Run separations
python run_pytorch_demucs.py fixtures/simple_mix.wav
./run_coreml_demucs.sh fixtures/simple_mix.wav

# 3. Compute metrics
python compute_metrics.py --save-csv outputs/metrics.csv

# 4. Generate report
python generate_report.py

# 5. View results
open outputs/parity_report.html
```

## Files and Structure

```
tests/parity/
├── fixtures/
│   ├── generate_fixtures.py    # Synthetic audio generator
│   └── simple_mix.wav           # Default test fixture
├── outputs/                     # Generated outputs (gitignored)
│   ├── *_pytorch.wav            # PyTorch stems
│   ├── *_coreml.wav             # CoreML stems
│   ├── metrics.csv              # BSS metrics
│   ├── metrics_plot.png         # Visualization
│   └── parity_report.html       # Full report
├── run_pytorch_demucs.py        # PyTorch runner
├── run_coreml_demucs.sh         # CoreML runner
├── compute_metrics.py           # Metrics computation
├── generate_report.py           # Report generator
├── test_parity.py               # Pytest suite
├── requirements.txt             # Python dependencies
└── README.md                    # Quick reference

```

## Contributing

To add new tests:

1. Add test fixtures to `fixtures/`
2. Update `test_parity.py` with new test cases
3. Run tests locally to verify
4. Submit PR with test results in description

## References

- [HTDemucs Paper](https://arxiv.org/abs/2211.08553)
- [BSS Eval Metrics](https://hal.inria.fr/inria-00544230/document)
- [MUSDB18 Dataset](https://sigsep.github.io/datasets/musdb.html)
- [mir_eval Documentation](https://craffel.github.io/mir_eval/)
