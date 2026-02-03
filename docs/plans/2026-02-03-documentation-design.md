# Documentation Overhaul Design

**Date:** 2026-02-03
**Status:** Approved

## Overview

HTDemucs has been successfully ported to CoreML with high precision. The temporary development documentation served its purpose and should be replaced with permanent, user-facing and developer-facing documentation.

## Cleanup

Remove all temporary development documentation (13 files):

### Phase Reports
- `docs/phase1-completion-report.md`
- `docs/phase2b-completion-report.md`
- `docs/phase3-completion-report.md`
- `docs/parity-testing-guide.md`

### Implementation Plans
- `docs/plans/2026-02-01-htdemucs-coreml-design.md`
- `docs/plans/2026-02-01-phase1-python-foundation.md`
- `docs/plans/2026-02-01-phase2-implementation-plan.md`
- `docs/plans/2026-02-01-phase2-swift-integration-design.md`
- `docs/plans/2026-02-02-phase3-audio-io-design.md`
- `docs/plans/2026-02-02-phase3-audio-io-implementation.md`
- `docs/plans/2026-02-02-phase4-quality-validation.md`

### Task Summaries
- `TASK9_IMPLEMENTATION_SUMMARY.md`
- `TASK9_CODE_REFERENCE.md`

### Research
- `initial-research.md`

## New Documentation Structure

```
README.md                      # Project showcase → quick start → usage
docs/
├── architecture.md            # Pipeline diagram, component breakdown, data flow
├── swift-api-guide.md         # SPM install, basic usage, progress tracking, errors
└── technical-decisions.md     # Why vDSP, why mixed precision, chunking math, model surgery
```

## README.md

**Purpose:** Showcase the project, then provide quick start and usage.

**Outline:**
1. **What it is** - HTDemucs 6-stem separation running natively on Apple Silicon via CoreML
2. **Why it matters** - First high-precision CoreML port, vDSP STFT approach, performance characteristics
3. **Quick demo** - CLI one-liner to separate a track
4. **Installation** - Swift Package Manager / building from source
5. **Usage** - CLI examples, then library integration teaser (pointing to docs/)
6. **Stems** - What the 6 stems are (drums, bass, vocals, other, piano, guitar)
7. **Requirements** - macOS 13+, iOS 18+, Apple Silicon recommended

## docs/architecture.md

**Purpose:** Explain how the system works for both ML engineers and Swift developers.

**Outline:**

1. **High-level pipeline diagram** (ASCII or Mermaid)
   - Audio in → Decode → STFT → Chunk → CoreML → Reassemble → iSTFT → Encode → Stems out

2. **Why this architecture?**
   - PyTorch HTDemucs has STFT/iSTFT baked into the model, but CoreML can't handle those ops reliably
   - Solution: Extract the "inner model" (spectrogram → separated spectrograms), implement STFT/iSTFT natively with vDSP

3. **Component breakdown** (one paragraph each):
   - `AudioDecoder` / `AudioEncoder` - FFmpeg-based I/O
   - `AudioFFT` - vDSP STFT/iSTFT with Hann windowing
   - `ChunkProcessor` - Handles long audio via overlap-add
   - `InferenceEngine` - CoreML model wrapper
   - `SeparationPipeline` - Orchestrates the full flow
   - `SeparationCoordinator` - Async progress streaming

4. **Data flow details**
   - Sample rates, FFT size (4096), hop length (1024)
   - Chunk size and overlap strategy
   - How stems are reconstructed

**Audience layering:** Lead with the "what" for Swift devs, follow with the "why this way" for ML engineers.

## docs/swift-api-guide.md

**Purpose:** Get developers using HTDemucsKit in their own projects quickly.

**Outline:**

1. **Installation** - SPM dependency declaration

2. **Basic usage** - Minimal code to separate a file
   ```swift
   let pipeline = SeparationPipeline(modelPath: ...)
   let stems = try await pipeline.separate(audioURL: input)
   ```

3. **Progress tracking** - Using AsyncStream for UI updates

4. **Accessing individual stems** - drums, bass, vocals, other, piano, guitar

5. **Configuration options** - Chunk size, output format, etc.

6. **Error handling** - Common errors and how to handle them

7. **Memory considerations** - Tips for long audio files

## docs/technical-decisions.md

**Purpose:** Explain the "why" behind architectural choices. Future-self reference + educational for ML engineers and curious Swift devs.

**Outline:**

1. **Why strip STFT/iSTFT from the model?**
   - CoreML doesn't handle complex number operations reliably
   - ONNX converters faced the same problem - this approach mirrors successful ONNX ports
   - vDSP is hardware-optimized and gives us control over precision

2. **Why vDSP instead of Accelerate's newer APIs?**
   - Availability across macOS/iOS versions
   - FFT performance characteristics
   - Hann window implementation specifics

3. **Mixed precision strategy**
   - FP32 for normalization and attention (precision-sensitive)
   - FP16 for everything else (performance)
   - How this was determined through validation

4. **Chunking and overlap-add**
   - Why chunking is necessary (memory, CoreML input constraints)
   - Overlap ratio and why it matters for seamless reconstruction
   - Edge case handling (short files, exact chunk boundaries)

5. **Model surgery approach**
   - What "inner model" extraction means
   - The spectral normalization layer and why it stays in Python
   - Validation methodology (PyTorch vs CoreML comparison)

## Inline Code Documentation

**Purpose:** Make the tricky parts self-explanatory for future readers.

### AudioFFT.swift (STFT/iSTFT)
- Explain the Hann window formula and why it's used
- Document the FFT size (4096) and hop length (1024) relationship
- Clarify the overlap-add reconstruction math
- Note the normalization to match PyTorch's `normalized=True`

### ChunkProcessor.swift (Chunking logic)
- Document chunk size in samples vs. spectral frames
- Explain overlap regions and crossfade strategy
- Note edge cases: files shorter than one chunk, remainder handling

### InferenceEngine.swift (CoreML inference)
- Document input tensor shape expectations [batch, channels, freq, time]
- Explain output stem ordering (drums, bass, vocals, other, piano, guitar)
- Note memory management for MLMultiArray

### Documentation style
- Block comments before complex functions explaining the "why"
- Inline comments for non-obvious math (formulas, magic numbers)
- Reference the technical-decisions.md doc for deeper context where appropriate

## Key Claims

- **First dedicated CoreML port of HTDemucs** - Existing solutions use MPS (PyTorch on Metal), not CoreML
- **High precision** - Mixed FP32/FP16 strategy preserves audio quality
- **Native signal processing** - vDSP STFT/iSTFT, not in-model operations
