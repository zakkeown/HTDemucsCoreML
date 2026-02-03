# HTDemucs CoreML

**The first high-precision CoreML port of Meta's HTDemucs music source separation model.**

Separate any song into 6 stems—drums, bass, vocals, other, piano, guitar—running natively on Apple Silicon via CoreML. No Python runtime, no cloud API, just fast on-device inference.

## What Makes This Different

HTDemucs is notoriously difficult to port. The model uses complex-valued STFT/iSTFT operations that CoreML doesn't support natively. Previous attempts either failed or required keeping PyTorch in the loop.

This project solves that by:

1. **Model surgery** — Extract the "inner model" that operates on spectrograms, bypassing the problematic STFT layers
2. **Native signal processing** — Implement STFT/iSTFT using Apple's vDSP Accelerate framework, matching HTDemucs exactly
3. **Mixed precision** — FP32 for normalization and attention (precision-sensitive), FP16 elsewhere (performance)

The result: CoreML inference that matches PyTorch output within perceptual tolerance.

## Quick Start

```bash
# Separate a song into stems
htdemucs-cli separate song.mp3 --output-dir stems/
```

Output:
```
stems/
├── drums.wav
├── bass.wav
├── vocals.wav
├── other.wav
├── piano.wav
└── guitar.wav
```

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/youruser/HTDemucsCoreML.git", from: "1.0.0")
]
```

### Building from Source

```bash
git clone https://github.com/youruser/HTDemucsCoreML.git
cd HTDemucsCoreML
swift build -c release
```

The CLI tool will be at `.build/release/htdemucs-cli`.

## Usage

### Command Line

```bash
# Basic separation
htdemucs-cli separate input.mp3 --output-dir output/

# Specify output format
htdemucs-cli separate input.wav --output-dir output/ --format flac

# Process multiple files
htdemucs-cli separate *.mp3 --output-dir stems/
```

### As a Library

```swift
import HTDemucsKit

let pipeline = try SeparationPipeline()
let stems = try await pipeline.separate(url: audioURL)

// Access individual stems
try stems.drums.write(to: drumsURL)
try stems.vocals.write(to: vocalsURL)
```

See [Swift API Guide](docs/swift-api-guide.md) for progress tracking, configuration, and advanced usage.

## The 6 Stems

| Stem | Description |
|------|-------------|
| **drums** | Kick, snare, hi-hats, cymbals, percussion |
| **bass** | Bass guitar, synth bass, sub-bass |
| **vocals** | Lead vocals, backing vocals, spoken word |
| **other** | Everything else—synths, pads, FX, strings |
| **piano** | Acoustic and electric piano, keys |
| **guitar** | Acoustic and electric guitar |

## Requirements

- **macOS 13+** or **iOS 18+**
- **Apple Silicon recommended** (Intel Macs work but slower)
- ~500MB RAM per separation

## Documentation

- [Architecture Overview](docs/architecture.md) — How the pipeline works
- [Swift API Guide](docs/swift-api-guide.md) — Using HTDemucsKit in your projects
- [Technical Decisions](docs/technical-decisions.md) — Why things are built this way

## Quality

CoreML output matches PyTorch reference within 1-2 dB across SDR/SIR/SAR metrics. For audio applications, this is perceptually identical.

## License

This project builds on Meta's Demucs model. See the [original repository](https://github.com/facebookresearch/demucs) for model licensing.
