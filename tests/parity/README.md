# Parity Testing Suite

## Setup

### 1. Build the CLI (from project root)

```bash
swift build -c release --product htdemucs-cli
```

### 2. Install Python dependencies

```bash
cd tests/parity
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
# Activate venv first (if not already active)
source venv/bin/activate

# Run parity tests
pytest test_parity.py -v

# Generate HTML report with visualizations
python generate_report.py outputs
```

## Metrics

- **SDR** (Signal-to-Distortion Ratio) - Overall separation quality (higher is better)
- **SIR** (Signal-to-Interference Ratio) - Interference from other sources (higher is better)
- **SAR** (Signal-to-Artifacts Ratio) - Artifacts introduced (higher is better)

## Expected Results

CoreML should match PyTorch within 1-2 dB across all metrics.
