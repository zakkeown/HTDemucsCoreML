# Test Fixtures

Golden output files for validating HTDemucs CoreML conversion.

## Files

- `*_input.npy`: Input audio (shape: [2, 441000] - stereo, 10s at 44.1kHz)
- `*_output.npy`: PyTorch Demucs output (shape: [1, 6, 2, 441000] - 6 stereo stems)
- `metadata.npy`: Test configuration metadata

## Test Cases

1. **silence**: All zeros - tests numerical stability
2. **sine_440hz**: Pure tone at 440Hz - tests frequency bin alignment
3. **white_noise**: Random noise - tests statistical properties

## Regenerating Fixtures

```bash
python scripts/generate_test_fixtures.py
```

Note: Requires ~100MB disk space and 5-10 minutes to run.
