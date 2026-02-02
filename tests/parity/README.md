# Parity Testing Suite

## Setup

```bash
cd tests/parity
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
# Run parity tests
pytest test_parity.py -v

# Generate comparison report
python compare_models.py --input ../fixtures/test_audio.wav
```

## Metrics

- **SDR** (Signal-to-Distortion Ratio) - Overall separation quality (higher is better)
- **SIR** (Signal-to-Interference Ratio) - Interference from other sources (higher is better)
- **SAR** (Signal-to-Artifacts Ratio) - Artifacts introduced (higher is better)

## Expected Results

CoreML should match PyTorch within 1-2 dB across all metrics.
