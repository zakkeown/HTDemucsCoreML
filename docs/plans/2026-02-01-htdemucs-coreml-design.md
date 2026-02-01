# HTDemucs CoreML Conversion Design

**Date:** 2026-02-01
**Target:** Native CoreML implementation of HTDemucs-6s for iOS/macOS
**Quality Bar:** Perceptually identical to PyTorch (SNR > 60dB, SI-SDR < 0.1dB)
**Platform:** iOS 18+, iPhone 15 Pro / iPad Pro M-series, batch processing

## Executive Summary

This design details the conversion of Facebook's HTDemucs-6s (Hybrid Transformer Demucs) model to CoreML for native iOS/macOS deployment. The core challenge is CoreML's lack of complex number support for STFT/iSTFT operations. Our solution uses clean architectural separation: implement STFT/iSTFT natively in Swift using vDSP, extract and convert only the pure neural network components to CoreML.

The system processes stereo audio in 10-second chunks with 1-second overlap, achieving perceptually identical quality to PyTorch through careful precision management (selective FP16/FP32) and extensive validation at each layer.

## Architecture

### System Components

The system consists of five independently testable layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    iOS/macOS Application                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: STFT (Swift/vDSP)                                 │
│    Input:  Stereo audio [2][Float] (441,000 samples/10s)   │
│    Output: Real/Imag spectrograms [2, 2049, 431]           │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: CoreML Model (Neural Network)                     │
│    Input:  Real/Imag spectrograms [1, 2, 2049, 431]        │
│    Output: 6 separation masks [1, 6, 2, 2049, 431]         │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: iSTFT (Swift/vDSP)                                │
│    Input:  6 masked spectrograms (real/imag)               │
│    Output: 6 stereo stems [2][Float] each                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: Pipeline Integration (Swift)                      │
│    Chunking, overlap-add, crossfading                       │
│    Output: 6 separated stems (drums/bass/vocals/etc)       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Python Preprocessing                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Model Surgery (PyTorch)                           │
│    Extract inner model (post-STFT, pre-iSTFT)              │
│    Output: InnerHTDemucs module                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow & Tensor Shapes

For a 10-second stereo audio chunk at 44.1kHz:

**Audio Input → STFT**
- Input: Stereo audio `[2][Float]` (left/right), each 441,000 samples
- Processing: Per-channel STFT (4096 FFT, 1024 hop, Hann window)
- Output: `real: [2, 2049, 431]`, `imag: [2, 2049, 431]`
  - 2 channels (stereo)
  - 2049 frequency bins (rfft: 4096/2 + 1)
  - 431 time frames

**STFT → CoreML Model**
- Input: Real and imag spectrograms, each `[1, 2, 2049, 431]` (add batch dimension)
- Processing: Hybrid Transformer Demucs with cross-domain attention
- Output: 6 separation masks `[1, 6, 2, 2049, 431]`
  - Multiplicative soft masks (~0.0-1.0 range)
  - Order: drums, bass, vocals, other, piano, guitar

**CoreML Masks → iSTFT**
- Input: Apply 6 masks element-wise to original complex spectrogram
  - `masked_real = real * mask`
  - `masked_imag = imag * mask`
- Processing: Per-stem, per-channel inverse FFT + overlap-add
- Output: 6 stereo stems, each `[2][Float]` of 441,000 samples

### Architectural Rationale

**Why Clean Separation (vs Integrated STFT)?**

Alternative approaches considered:
1. **Convmelspec hybrid:** Use Adobe's convmelspec for STFT inside CoreML
   - Pro: Less Swift code, potential GPU acceleration of STFT
   - Con: Harder to debug, less flexible, still need Swift iSTFT

2. **Mixxx fork:** Use Mixxx's convolution-based STFT/iSTFT rewrite
   - Pro: Proven to work (SI-SDR < 0.1dB)
   - Con: Less control, harder to test independently

**Selected: Clean separation**
- Full control over STFT precision (Float32 throughout)
- Independent validation of each component
- Clear boundaries for testing
- No dependency on external modified models

## Layer 1: PyTorch Model Surgery

### Approach: Subclass with Bypassed STFT

Instead of graph cutting, we subclass `HDemucs` to expose the inner model:

```python
class InnerHTDemucs(nn.Module):
    """Extracted HTDemucs core without STFT/iSTFT operations."""

    def __init__(self, sources=['drums', 'bass', 'vocals', 'other', 'piano', 'guitar']):
        super().__init__()
        # Load pretrained htdemucs_6s
        original_model = get_model('htdemucs_6s')
        original_model.eval()

        # Extract neural network components (excludes inline STFT/iSTFT)
        self.sources = sources
        self.audio_channels = 2
        self.encoder = original_model.encoder
        self.decoder = original_model.decoder
        self.tencoder = original_model.tencoder  # Time domain
        self.tdecoder = original_model.tdecoder

    def forward(self, mag_real, mag_imag):
        """
        Args:
            mag_real: [batch, channels, freq_bins, time_frames]
            mag_imag: [batch, channels, freq_bins, time_frames]

        Returns:
            masks: [batch, sources, channels, freq_bins, time_frames]
        """
        # Complex-as-Channels (CaC) representation
        x = torch.cat([mag_real, mag_imag], dim=1)

        # Frequency domain processing
        x = self.encoder(x)
        x = self.transformer(x)
        masks = self.decoder(x)

        return masks
```

**Key Insight:** HTDemucs uses "Complex-as-Channels" representation internally—real and imaginary components concatenated along channel dimension. This is already CoreML-compatible (Float32 tensors, no complex dtypes).

### Validation

1. Run full `htdemucs_6s` on test audio
2. Capture intermediate tensor right after STFT (real/imag)
3. Feed to `InnerHTDemucs`
4. Compare masks: `torch.allclose(original_masks, extracted_masks, rtol=1e-5, atol=1e-7)`

**Alternative if subclassing fails:** Use forward hooks to intercept at STFT boundary.

## Layer 2: CoreML Conversion

### Precision Strategy

Goal: Selective FP16/FP32 for quality without sacrificing performance.

**Precision-Sensitive Operations (must stay FP32):**

```python
PRECISION_SENSITIVE_OPS = {
    # Normalization (overflow risk in FP16)
    "pow", "sqrt", "real_div", "l2_norm",

    # Reduction (accumulation errors)
    "reduce_mean", "reduce_sum",

    # Attention (precision critical)
    "softmax", "matmul"
}
```

**Conversion Code:**

```python
def precision_selector(op):
    """Returns True if op should use FP16, False for FP32."""
    return op.op_type not in PRECISION_SENSITIVE_OPS

mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="spectrogram_real", shape=(1, 2, 2049, 431)),
        ct.TensorType(name="spectrogram_imag", shape=(1, 2, 2049, 431))
    ],
    outputs=[
        ct.TensorType(name="masks", shape=(1, 6, 2, 2049, 431))
    ],
    compute_precision=ct.transform.FP16ComputePrecision(
        op_selector=precision_selector
    ),
    compute_units=ct.ComputeUnit.CPU_AND_GPU,  # Phase 1: validation
    minimum_deployment_target=ct.target.iOS18   # For SDPA support
)
```

### Compute Unit Strategy (3 Phases)

**Phase 1: CPU_AND_GPU** (validation)
- Most predictable precision, easiest debugging
- FP32 available on CPU fallback
- **Goal:** Prove conversion works, hit quality targets

**Phase 2: ALL** (enable ANE)
- Add Neural Engine for performance
- Profile with Xcode Instruments (monitor actual compute unit usage)
- **Goal:** Validate ANE doesn't degrade quality below SNR > 60dB

**Phase 3: Palettization** (deployment)
- Apply 6-bit weight compression after quality validation
- Reduces ~80MB → ~16MB
- **Goal:** Maintain SNR > 60dB with compressed weights

### Validation

- Numerical diff: `np.allclose(pytorch_out, coreml_out, rtol=1e-3, atol=1e-4)`
- Log actual precision used per layer
- Track max absolute error and mean relative error per stem

## Layer 3: Native STFT (Swift)

### Implementation using vDSP

```swift
import Accelerate

class AudioFFT {
    private var fftSetup: FFTSetup
    private let log2n: vDSP_Length
    private let fftSize: Int
    private let hopLength: Int
    private var window: [Float]

    init(fftSize: Int = 4096, hopLength: Int = 1024) {
        self.fftSize = fftSize
        self.hopLength = hopLength
        self.log2n = vDSP_Length(log2(Double(fftSize)))
        self.fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!

        // Create Hann window (matching PyTorch)
        self.window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
    }

    func stft(_ audio: [Float]) throws -> (real: [[Float]], imag: [[Float]]) {
        // Validate input
        guard audio.count >= fftSize else {
            throw AudioProcessingError.audioTooShort(audio.count, fftSize)
        }
        guard audio.allSatisfy({ $0.isFinite }) else {
            throw AudioProcessingError.invalidAudioData("NaN or Inf values")
        }

        // Pad if needed
        let paddedLength = ((audio.count - fftSize) / hopLength + 1) * hopLength + fftSize
        var padded = audio
        if audio.count < paddedLength {
            padded.append(contentsOf: [Float](repeating: 0, count: paddedLength - audio.count))
        }

        let numFrames = (padded.count - fftSize) / hopLength + 1
        var real = [[Float]]()
        var imag = [[Float]]()

        // Process each frame
        for frame in 0..<numFrames {
            let start = frame * hopLength
            var windowed = [Float](repeating: 0, count: fftSize)

            // Apply window
            vDSP_vmul(Array(padded[start..<start+fftSize]), 1,
                     window, 1, &windowed, 1, vDSP_Length(fftSize))

            // Perform FFT
            // ... (use vDSP_fft_zrip for real FFT)

            real.append(frameReal)
            imag.append(frameImag)
        }

        return (real, imag)
    }
}
```

### Validation

**Property Tests:**
- Energy conservation (Parseval's theorem): `sum(|STFT|²) ≈ sum(|signal|²) / hop_length`
- COLA constraint: `sum(window[n + m*hop]) = constant`
- Symmetry: `STFT[f] = conj(STFT[N-f])` for real input

**Golden Output Tests:**
- Generate reference from `torch.stft(audio, n_fft=4096, hop_length=1024, window=torch.hann_window(4096))`
- Compare element-wise with tolerance `rtol=1e-5, atol=1e-6`

**Edge Cases:**
- Silent audio (all zeros)
- Single sine wave (verify bin alignment)
- Non-divisible lengths (padding behavior)

## Layer 4: Native iSTFT (Swift)

### Implementation

```swift
func istft(real: [[Float]], imag: [[Float]]) throws -> [Float] {
    guard real.count == imag.count else {
        throw AudioProcessingError.mismatchedDimensions
    }

    let numFrames = real.count
    let outputLength = (numFrames - 1) * hopLength + fftSize
    var output = [Float](repeating: 0, count: outputLength)
    var windowSum = [Float](repeating: 0, count: outputLength)

    // Reconstruct with overlap-add
    for (frameIdx, (frameReal, frameImag)) in zip(real, imag).enumerated() {
        // Inverse FFT
        var timeFrame = performInverseFFT(frameReal, frameImag)

        // Apply window
        vDSP_vmul(timeFrame, 1, window, 1, &timeFrame, 1, vDSP_Length(fftSize))

        // Overlap-add
        let start = frameIdx * hopLength
        for i in 0..<fftSize {
            output[start + i] += timeFrame[i]
            windowSum[start + i] += window[i] * window[i]  // Window normalization
        }
    }

    // Normalize by window sum (COLA compliance)
    for i in 0..<outputLength {
        if windowSum[i] > 1e-8 {
            output[i] /= windowSum[i]
        }
    }

    return output
}
```

### Validation

**Property Tests:**
- Round-trip: `audio → STFT → iSTFT ≈ audio` (tolerance: max_error < 1e-5)
- COLA compliance: verify unity gain reconstruction

**Golden Output Tests:**
- Use PyTorch STFT outputs, compare `torch.istft()` vs Swift implementation
- Tolerance: `rtol=1e-5, atol=1e-6`

## Layer 5: End-to-End Pipeline

### Chunking Strategy

```
Audio: |----chunk0----|----chunk1----|----chunk2----|
       |<---10s--->|
       |1s|  8s  |1s|
           |<---10s--->|
           |1s|  8s  |1s|
                 |<---10s--->|
                 |1s|  8s  |1s|

- Chunk size: 10 seconds (441,000 samples at 44.1kHz)
- Overlap: 1 second each side (44,100 samples)
- Stride: 8 seconds (352,800 samples)
- Crossfade: Linear blend in overlap regions
```

### Pipeline Implementation

```swift
class HTDemucsProcessor {
    private let fft: AudioFFT
    private let model: MLModel

    private let chunkDuration: Float = 10.0
    private let overlapDuration: Float = 1.0
    private let sampleRate: Int = 44100

    private var chunkSamples: Int { Int(chunkDuration * Float(sampleRate)) }
    private var overlapSamples: Int { Int(overlapDuration * Float(sampleRate)) }
    private var hopSamples: Int { chunkSamples - 2 * overlapSamples }

    func separate(
        stereoAudio: [[Float]],
        progress: @escaping (Float) -> Void
    ) async throws -> [StemType: [[Float]]] {
        let audioLength = stereoAudio[0].count

        // Initialize output buffers (6 stems, stereo)
        var outputs: [StemType: [[Float]]] = [:]
        for stem in StemType.allCases {
            outputs[stem] = [
                [Float](repeating: 0, count: audioLength),
                [Float](repeating: 0, count: audioLength)
            ]
        }

        var weights = [Float](repeating: 0, count: audioLength)
        let totalChunks = (audioLength + hopSamples - 1) / hopSamples
        var processedChunks = 0

        // Process chunks with overlap
        for chunkStart in stride(from: 0, to: audioLength, by: hopSamples) {
            try await autoreleasepool {
                let chunkEnd = min(chunkStart + chunkSamples, audioLength)
                let actualChunkSize = chunkEnd - chunkStart

                // Extract and process chunk
                let chunk = extractChunk(stereoAudio, start: chunkStart,
                                        length: actualChunkSize, padTo: chunkSamples)
                let stemChunks = try await processChunk(chunk)

                // Create blend window
                let window = createBlendWindow(
                    chunkSize: actualChunkSize,
                    overlapSize: overlapSamples,
                    isFirst: chunkStart == 0,
                    isLast: chunkEnd >= audioLength
                )

                // Accumulate into outputs
                for (stem, stemAudio) in stemChunks {
                    for channel in 0..<2 {
                        for i in 0..<actualChunkSize {
                            outputs[stem]![channel][chunkStart + i] +=
                                stemAudio[channel][i] * window[i]
                        }
                    }
                }

                // Accumulate weights
                for i in 0..<actualChunkSize {
                    weights[chunkStart + i] += window[i]
                }
            }

            processedChunks += 1
            progress(Float(processedChunks) / Float(totalChunks))
        }

        // Normalize by accumulated weights
        for stem in StemType.allCases {
            for channel in 0..<2 {
                for i in 0..<audioLength where weights[i] > 0 {
                    outputs[stem]![channel][i] /= weights[i]
                }
            }
        }

        return outputs
    }
}
```

### Blend Window Function

```swift
private func createBlendWindow(
    chunkSize: Int,
    overlapSize: Int,
    isFirst: Bool,
    isLast: Bool
) -> [Float] {
    var window = [Float](repeating: 1.0, count: chunkSize)

    // Fade in at start (unless first chunk)
    if !isFirst {
        for i in 0..<overlapSize {
            window[i] = Float(i) / Float(overlapSize)
        }
    }

    // Fade out at end (unless last chunk)
    if !isLast {
        for i in 0..<overlapSize {
            let idx = chunkSize - overlapSize + i
            window[idx] = 1.0 - Float(i) / Float(overlapSize)
        }
    }

    return window
}
```

### Single Chunk Processing

```swift
private func processChunk(_ chunk: [[Float]]) async throws -> [StemType: [[Float]]] {
    // 1. STFT per channel
    var stftReal: [[Float]] = []
    var stftImag: [[Float]] = []
    for channel in chunk {
        let (real, imag) = try fft.stft(channel)
        stftReal.append(contentsOf: real)
        stftImag.append(contentsOf: imag)
    }

    // 2. CoreML inference
    let realInput = MLMultiArray(stftReal)  // [1, 2, 2049, 431]
    let imagInput = MLMultiArray(stftImag)

    let prediction = try await model.prediction(from: [
        "spectrogram_real": realInput,
        "spectrogram_imag": imagInput
    ])

    let masks = extractMasks(from: prediction)  // [6, 2, 2049, 431]

    // 3. Apply masks and iSTFT per stem
    var stemOutputs: [StemType: [[Float]]] = [:]
    for (stemIdx, stem) in StemType.allCases.enumerated() {
        var stemAudio: [[Float]] = []
        for channel in 0..<2 {
            let maskedReal = applyMask(stftReal[channel], mask: masks[stemIdx][channel])
            let maskedImag = applyMask(stftImag[channel], mask: masks[stemIdx][channel])
            let audio = try fft.istft(real: maskedReal, imag: maskedImag)
            stemAudio.append(audio)
        }
        stemOutputs[stem] = stemAudio
    }

    return stemOutputs
}
```

## Testing Strategy

### Test Matrix by Layer

**Layer 1: PyTorch Model Surgery**

*Property Tests:*
- No STFT/iSTFT ops in extracted model graph
- All parameters frozen (no gradients)
- Output shape validation

*Golden Output Tests:*
- Capture intermediate spectrograms from full Demucs
- Compare extracted model output
- Tolerance: `torch.allclose(rtol=1e-5, atol=1e-7)`

*Fixtures:* 3-5 diverse 10s audio samples

**Layer 2: CoreML Conversion**

*Property Tests:*
- Runs on CPU_AND_GPU compute units
- Model size ~80MB uncompressed
- Correct input/output names and shapes

*Golden Output Tests:*
- Same spectrograms as Layer 1
- Compare CoreML vs PyTorch predictions
- Tolerance: `np.allclose(rtol=1e-3, atol=1e-4)`

*Precision Monitoring:*
- Log FP16 vs FP32 op distribution
- Flag FP16 overflow warnings

**Layer 3: Native STFT (Swift)**

*Property Tests:*
- Parseval's theorem (energy conservation)
- COLA constraint
- Real FFT symmetry

*Golden Output Tests:*
- Compare against `torch.stft()` output
- Tolerance: `rtol=1e-5, atol=1e-6`

*Edge Cases:*
- Silence, sine waves, non-divisible lengths

**Layer 4: Native iSTFT (Swift)**

*Property Tests:*
- Round-trip test: `audio → STFT → iSTFT ≈ audio`
- Max error < 1e-5

*Golden Output Tests:*
- Compare against `torch.istft()` output
- Tolerance: `rtol=1e-5, atol=1e-6`

**Layer 5: End-to-End Pipeline**

*Perceptual Metrics:*
- **SI-SDR:** Target < 0.1dB difference vs PyTorch
- **SNR:** Target > 60dB
- Compute per-stem and averaged

*Golden Output Tests:*
- 10+ diverse songs (3-5 min, various genres)
- Save outputs for manual listening comparison

*Edge Cases:*
- Silence, mono-to-stereo, clipping, quiet sections

### Automation

- **Python tests:** pytest, run in CI
- **Swift tests:** XCTest, run in CI
- **Golden outputs:** Stored as compressed numpy/binary (~100MB)
- **Failures block merge**

## Error Handling

### Python (Layers 1-2)

```python
class ModelExtractionError(Exception):
    """Raised when model surgery fails."""
    pass

def extract_inner_model():
    try:
        model = get_model('htdemucs_6s')
    except Exception as e:
        raise ModelExtractionError(f"Failed to load model: {e}")

    # Validate structure
    if not hasattr(model, 'encoder'):
        raise ModelExtractionError("Unexpected model structure")

    return InnerHTDemucs(model)
```

### Swift (Layers 3-5)

```swift
enum AudioProcessingError: Error {
    case audioTooShort(length: Int, required: Int)
    case invalidSampleRate(actual: Int, expected: Int)
    case invalidAudioData(reason: String)
    case modelInferenceFailed(underlyingError: Error)
}

func stft(_ audio: [Float]) throws -> (real: [[Float]], imag: [[Float]]) {
    guard audio.count >= fftSize else {
        throw AudioProcessingError.audioTooShort(audio.count, fftSize)
    }

    guard audio.allSatisfy({ $0.isFinite }) else {
        throw AudioProcessingError.invalidAudioData("NaN or Inf values")
    }

    // Process...
}
```

### Edge Cases Handled

- Audio length < FFT size (4096 samples)
- Non-44.1kHz sample rate (resample or error)
- Mono input (convert to stereo: duplicate channel)
- Very long audio (>30 min: memory warnings)
- CoreML inference failures (retry once, then smaller chunks)
- NaN/Inf values (reject with clear error)

### Logging & Telemetry

```swift
import os.log

let logger = Logger(subsystem: "com.app.demucs", category: "processing")

logger.info("Processing chunk \(index): \(duration)s")
logger.debug("STFT shape: \(real.count)×\(real[0].count)")
logger.warning("CoreML fell back to CPU")
logger.error("Inference failed: \(error)")
```

**Key Metrics:**
- Processing time per chunk
- Peak memory usage
- Compute unit distribution (ANE/GPU/CPU)
- Quality degradation warnings

## Development Plan

### Phase 1: Python Foundation (Week 1-2)

**Goal:** Prove model surgery and CoreML conversion work

1. Environment setup
   - Install: `torch`, `torchaudio`, `coremltools>=8.0`, `demucs`
   - Download `htdemucs_6s` weights
   - Create test fixtures (save PyTorch outputs as .npy)

2. Implement `InnerHTDemucs` extraction
   - Write subclass
   - Validate with forward hooks
   - **Checkpoint:** Layer 1 tests pass

3. CoreML conversion
   - Trace with example spectrograms
   - Convert with precision selector (CPU_AND_GPU)
   - Numerical diff tests
   - **Checkpoint:** Layer 2 tests pass (rtol=1e-3, atol=1e-4)

### Phase 2: Swift STFT/iSTFT (Week 2-3)

**Goal:** Implement and validate native signal processing

4. Implement STFT
   - `AudioFFT` class with vDSP
   - Hann windowing, frame generation, FFT
   - Property and golden output tests
   - **Checkpoint:** Layer 3 tests pass (rtol=1e-5)

5. Implement iSTFT
   - Inverse FFT per frame
   - Overlap-add with window compensation
   - Round-trip validation
   - **Checkpoint:** Layer 4 tests pass (rtol=1e-5)

### Phase 3: Integration (Week 3-4)

**Goal:** Connect all components, validate end-to-end

6. Swift CoreML integration
   - Load `.mlpackage`
   - Wire STFT → CoreML → iSTFT
   - Single 10s chunk processing
   - **Checkpoint:** Single-chunk inference works

7. Chunking and overlap-add
   - Segment long audio (10s chunks, 1s overlap)
   - Linear crossfade
   - Reconstruct full stems
   - **Checkpoint:** Process 3-minute song

8. End-to-end validation
   - Test suite (10+ songs)
   - SI-SDR and SNR metrics
   - Manual listening tests
   - **Checkpoint:** SI-SDR < 0.1dB, SNR > 60dB

### Phase 4: Optimization (Week 4-5, optional)

**Goal:** Performance tuning for production

9. Enable ANE (ALL compute units)
   - Profile with Xcode Instruments
   - Verify quality maintained

10. Apply palettization
    - 6-bit weight compression
    - Revalidate quality metrics

## Success Criteria

### Quality Metrics

- **SI-SDR difference:** < 0.1dB vs PyTorch Demucs
- **SNR:** > 60dB (perceptually identical)
- **Per-layer validation:** All tolerance thresholds met
- **Manual listening:** No audible artifacts in test suite

### Performance Targets

- **Processing time:** < 60s for 3-minute song (batch processing acceptable)
- **Memory usage:** < 2GB peak on iPhone 15 Pro
- **Model size:** < 100MB uncompressed, < 20MB with palettization

### Platform Requirements

- **iOS version:** 18+ (for SDPA support)
- **Devices:** iPhone 15 Pro, iPad Pro M-series minimum
- **Compute units:** CPU_AND_GPU initially, ALL (with ANE) after validation

## Open Questions & Risks

### Known Risks

1. **Transformer conversion instability**
   - Mitigation: Disable fast path (`torch.backends.mha.set_fastpath_enabled(False)`)
   - Fallback: Split model if ANECompilerService times out

2. **Precision degradation on ANE**
   - Mitigation: Extensive testing with ALL compute units
   - Fallback: Stay on CPU_AND_GPU if quality drops

3. **Memory pressure on lower-end devices**
   - Mitigation: Target iPhone 15 Pro+ only (6GB RAM)
   - Future: Adaptive chunk sizes for broader compatibility

### To Investigate

- Optimal crossfade curve (linear vs cosine vs Hann-based)
- Whether `matmul` in attention should be FP16 or FP32
- Metal acceleration for FFT (MPSGraph) vs vDSP performance comparison

## References

### Code Repositories

- **facebook/demucs** — Original PyTorch implementation
- **sevagh/demucs.cpp** — C++ reference for low-memory implementation
- **mixxxdj/demucs** — ONNX export with rewritten STFT
- **adobe-research/convmelspec** — Convertible STFT via convolutions
- **apple/ml-ane-transformers** — ANE-optimized transformer patterns

### Documentation

- coremltools 8.0 documentation (supported operations)
- Apple Technical Note TN3151 (ANE deployment debugging)
- WWDC 2024: "Accelerate machine learning with Metal"
- arXiv:2211.08553 (HTDemucs paper)

### Key Insights from Research

- Mixxx project achieved SI-SDR < 0.1dB with conv-based STFT rewrite
- Whisper.cpp demonstrates successful CoreML encoder integration
- vDSP maintains Float32 precision for STFT/iSTFT
- 6-bit palettization reduces model size 5x with minimal quality impact
- Complex number support won't come to CoreML—architectural solution required

---

**Next Steps:**
1. Initialize git repository (if not already done)
2. Set up Python environment and test fixtures
3. Begin Phase 1: Model surgery implementation
