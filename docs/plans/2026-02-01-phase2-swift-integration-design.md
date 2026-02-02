# Phase 2: Swift STFT/iSTFT + CoreML Integration Design

**Date:** 2026-02-01
**Target:** Native Swift implementation of STFT/iSTFT with full CoreML integration
**Quality Bar:** PyTorch parity (rtol=1e-5, atol=1e-6) + battle-tested edge cases
**Platform:** iOS target, macOS CLI for development

## Executive Summary

Phase 2 implements native STFT/iSTFT in Swift using Accelerate's vDSP framework, then integrates with the Phase 1 CoreML model to create a complete audio source separation pipeline. The approach is correctness-first: achieve perfect numerical parity with PyTorch STFT/iSTFT (including property tests and edge cases) before integration, ensuring debugging the full pipeline is straightforward.

**Key Design Decisions:**
- Swift wrapper layer over vDSP (clean APIs, maximum performance)
- Library + CLI structure (platform-agnostic core, easy iOS migration)
- Hybrid testing (Python for golden outputs, Swift for algorithmic validation)
- Battle-tested STFT/iSTFT before CoreML integration

## Architecture Overview

### Project Structure

```
HTDemucsCoreML/
├── .worktrees/
│   └── phase2-swift-integration/     # New worktree from main
│       ├── Package.swift              # Swift package manifest
│       ├── Sources/
│       │   ├── HTDemucsKit/           # Core library (iOS/macOS)
│       │   │   ├── Audio/
│       │   │   │   ├── AudioFFT.swift           # vDSP wrapper
│       │   │   │   ├── STFTProcessor.swift      # STFT/iSTFT logic
│       │   │   │   └── AudioTypes.swift         # Common types
│       │   │   ├── CoreML/
│       │   │   │   ├── ModelLoader.swift        # Load .mlpackage
│       │   │   │   └── InferenceEngine.swift    # Run CoreML
│       │   │   └── Pipeline/
│       │   │       ├── SeparationPipeline.swift # End-to-end pipeline
│       │   │       └── ChunkProcessor.swift     # Chunking/overlap-add
│       │   └── htdemucs-cli/          # macOS CLI tool
│       │       └── main.swift         # CLI entry point
│       ├── Tests/
│       │   └── HTDemucsKitTests/
│       │       ├── AudioFFTTests.swift          # vDSP wrapper tests
│       │       ├── STFTPropertyTests.swift      # Parseval, COLA, etc.
│       │       ├── RoundTripTests.swift         # STFT→iSTFT validation
│       │       ├── PyTorchParityTests.swift     # Golden output comparison
│       │       └── EdgeCaseTests.swift          # Mono, long audio, etc.
│       └── Resources/
│           └── GoldenOutputs/         # .npy files from PyTorch
├── scripts/
│   └── generate_swift_fixtures.py     # Export PyTorch → Swift fixtures
└── src/htdemucs_coreml/              # Phase 1 Python (in main)
```

### Component Layers

1. **Audio Layer** - STFT/iSTFT with vDSP (battle-tested)
2. **CoreML Layer** - Load and run Phase 1 model
3. **Pipeline Layer** - Chunking, overlap-add, full orchestration
4. **CLI Tool** - macOS testing harness

## Layer 1: Audio - vDSP STFT/iSTFT

### AudioFFT Class Design

**Purpose:** Clean Swift wrapper over vDSP FFT operations. Hides complexity, provides performance.

```swift
import Accelerate

public class AudioFFT {
    // MARK: - Configuration (matches PyTorch)
    public let fftSize: Int = 4096
    public let hopLength: Int = 1024
    private let log2n: vDSP_Length

    // MARK: - vDSP State (reused across calls)
    private var fftSetup: FFTSetup
    private var window: [Float]             // Hann window, pre-computed

    // MARK: - Working Buffers (avoid reallocation)
    private var splitComplexReal: [Float]
    private var splitComplexImag: [Float]
    private var windowedFrame: [Float]

    // MARK: - Initialization
    public init() throws {
        self.log2n = vDSP_Length(log2(Double(fftSize)))

        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            throw AudioFFTError.fftSetupFailed
        }
        self.fftSetup = setup

        // Pre-compute Hann window (expensive, do once)
        self.window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))

        // Allocate working buffers (half-size for real FFT)
        let halfSize = fftSize / 2
        self.splitComplexReal = [Float](repeating: 0, count: halfSize)
        self.splitComplexImag = [Float](repeating: 0, count: halfSize)
        self.windowedFrame = [Float](repeating: 0, count: fftSize)
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    // MARK: - Public API

    /// Compute Short-Time Fourier Transform
    /// - Parameter audio: Input audio samples
    /// - Returns: (real, imag) spectrograms, each [numFrames][numBins]
    /// - Throws: AudioFFTError if audio too short or invalid
    public func stft(_ audio: [Float]) throws -> (real: [[Float]], imag: [[Float]]) {
        // Validate input
        guard audio.count >= fftSize else {
            throw AudioFFTError.audioTooShort(audio.count, fftSize)
        }
        guard audio.allSatisfy({ $0.isFinite }) else {
            throw AudioFFTError.invalidAudioData("NaN or Inf values")
        }

        // Compute number of frames
        let numFrames = (audio.count - fftSize) / hopLength + 1
        let numBins = fftSize / 2 + 1  // Real FFT output size

        var realOutput: [[Float]] = []
        var imagOutput: [[Float]] = []

        // Process each frame
        for frameIdx in 0..<numFrames {
            let start = frameIdx * hopLength
            let end = start + fftSize

            // Extract and window frame
            let frame = Array(audio[start..<end])
            vDSP_vmul(frame, 1, window, 1, &windowedFrame, 1, vDSP_Length(fftSize))

            // Perform real FFT
            let (frameReal, frameImag) = performRealFFT(windowedFrame)

            realOutput.append(frameReal)
            imagOutput.append(frameImag)
        }

        return (real: realOutput, imag: imagOutput)
    }

    /// Compute inverse Short-Time Fourier Transform
    /// - Parameters:
    ///   - real: Real component spectrogram [numFrames][numBins]
    ///   - imag: Imaginary component spectrogram [numFrames][numBins]
    /// - Returns: Reconstructed audio samples
    /// - Throws: AudioFFTError if dimensions mismatch
    public func istft(real: [[Float]], imag: [[Float]]) throws -> [Float] {
        guard real.count == imag.count else {
            throw AudioFFTError.mismatchedDimensions
        }

        let numFrames = real.count
        let outputLength = (numFrames - 1) * hopLength + fftSize

        var output = [Float](repeating: 0, count: outputLength)
        var windowSum = [Float](repeating: 0, count: outputLength)

        // Reconstruct with overlap-add
        for (frameIdx, (frameReal, frameImag)) in zip(real, imag).enumerated() {
            // Inverse FFT
            var timeFrame = performInverseRealFFT(frameReal, frameImag)

            // Apply window
            vDSP_vmul(timeFrame, 1, window, 1, &timeFrame, 1, vDSP_Length(fftSize))

            // Overlap-add
            let start = frameIdx * hopLength
            for i in 0..<fftSize {
                output[start + i] += timeFrame[i]
                windowSum[start + i] += window[i] * window[i]
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

    // MARK: - Private Helpers

    private func performRealFFT(_ input: [Float]) -> ([Float], [Float]) {
        // vDSP real FFT implementation
        // Returns arrays of size (fftSize/2 + 1)
        // ...implementation details...
        fatalError("To be implemented with vDSP_fft_zrip")
    }

    private func performInverseRealFFT(_ real: [Float], _ imag: [Float]) -> [Float] {
        // vDSP inverse real FFT implementation
        // ...implementation details...
        fatalError("To be implemented with vDSP_fft_zrip inverse")
    }
}

// MARK: - Error Types

public enum AudioFFTError: Error, LocalizedError {
    case fftSetupFailed
    case audioTooShort(Int, Int)  // (actual, required)
    case invalidAudioData(String)
    case mismatchedDimensions

    public var errorDescription: String? {
        switch self {
        case .fftSetupFailed:
            return "Failed to create FFT setup"
        case .audioTooShort(let actual, let required):
            return "Audio too short: \(actual) samples, need at least \(required)"
        case .invalidAudioData(let reason):
            return "Invalid audio data: \(reason)"
        case .mismatchedDimensions:
            return "Real and imaginary spectrograms have mismatched dimensions"
        }
    }
}
```

**Key Design Decisions:**
- **Stateful setup:** FFTSetup and window created once, reused (performance)
- **Working buffers:** Pre-allocated to avoid malloc in hot path
- **Error handling:** Swift errors with descriptive messages
- **Memory safety:** `deinit` cleans up vDSP resources
- **Clean API:** Swift arrays in/out, vDSP internals hidden

### Validation Requirements

**Property Tests:**
1. **Parseval's Theorem:** `sum(|STFT|²) ≈ sum(|signal|²) / hop_length`
2. **COLA Constraint:** `sum(window[n + m*hop]) = constant`
3. **Symmetry:** `STFT[f] = conj(STFT[N-f])` for real input

**Round-Trip Test:**
- `audio → STFT → iSTFT ≈ audio` with max error < 1e-5

**PyTorch Parity:**
- Match `torch.stft()` output for all test cases: rtol=1e-5, atol=1e-6

## Layer 2: Testing Strategy

### Test Levels (4 tiers)

**Level 1: Property Tests (Swift XCTest, fast)**

```swift
// Tests/HTDemucsKitTests/STFTPropertyTests.swift

import XCTest
@testable import HTDemucsKit

final class STFTPropertyTests: XCTestCase {
    var fft: AudioFFT!

    override func setUp() {
        fft = try! AudioFFT()
    }

    func testParsevalTheorem() {
        // Generate test signal
        let audio = generateSineWave(freq: 440, sampleRate: 44100, duration: 1.0)

        // Compute STFT
        let (real, imag) = try! fft.stft(audio)

        // Energy in time domain
        let timeEnergy = audio.map { $0 * $0 }.reduce(0, +)

        // Energy in frequency domain
        var freqEnergy: Float = 0
        for (r, i) in zip(real, imag) {
            for (rVal, iVal) in zip(r, i) {
                freqEnergy += rVal * rVal + iVal * iVal
            }
        }

        // Parseval: freq energy * hop = time energy
        XCTAssertEqual(freqEnergy * Float(fft.hopLength), timeEnergy,
                       accuracy: timeEnergy * 0.001) // 0.1% tolerance
    }

    func testCOLAConstraint() {
        // Verify window overlap-add gives constant
        let windowSum = computeOverlapAddSum(window: fft.window,
                                             hop: fft.hopLength,
                                             length: fft.fftSize * 4)

        // Should be constant (within numerical precision)
        let mean = windowSum.reduce(0, +) / Float(windowSum.count)
        let maxDeviation = windowSum.map { abs($0 - mean) }.max()!

        XCTAssertLessThan(maxDeviation, 1e-6)
    }

    func testRealFFTSymmetry() {
        let audio = generateWhiteNoise(samples: 44100)
        let (real, imag) = try! fft.stft(audio)

        // For each frame, verify STFT[f] = conj(STFT[N-f])
        for frameIdx in 0..<real.count {
            let frame = real[frameIdx]
            let numBins = frame.count

            // DC and Nyquist should have zero imaginary part
            XCTAssertEqual(imag[frameIdx][0], 0, accuracy: 1e-6)
            XCTAssertEqual(imag[frameIdx][numBins-1], 0, accuracy: 1e-6)
        }
    }
}
```

**Level 2: Round-Trip Tests (Swift XCTest, fast)**

```swift
// Tests/HTDemucsKitTests/RoundTripTests.swift

final class RoundTripTests: XCTestCase {
    var fft: AudioFFT!

    func testPerfectReconstruction() {
        let testCases: [(name: String, generator: () -> [Float])] = [
            ("silence", { [Float](repeating: 0, count: 88200) }),
            ("sine_440", { generateSineWave(freq: 440, sampleRate: 44100, duration: 2.0) }),
            ("white_noise", { generateWhiteNoise(samples: 88200) })
        ]

        for testCase in testCases {
            let original = testCase.generator()

            // Forward transform
            let (real, imag) = try! fft.stft(original)

            // Inverse transform
            let reconstructed = try! fft.istft(real: real, imag: imag)

            // Compare (account for windowing truncation at edges)
            let compareLength = min(original.count, reconstructed.count)
            let maxError = zip(original.prefix(compareLength),
                              reconstructed.prefix(compareLength))
                .map { abs($0 - $1) }
                .max()!

            XCTAssertLessThan(maxError, 1e-5,
                            "Round-trip failed for \(testCase.name)")
        }
    }
}
```

**Level 3: PyTorch Parity (Python harness validates Swift CLI)**

```python
# tests/validate_swift_stft.py

import numpy as np
import torch
import subprocess
from pathlib import Path

def test_stft_matches_pytorch():
    """Validate Swift STFT output matches PyTorch."""
    test_cases = ['silence', 'sine_440hz', 'white_noise']

    for test_name in test_cases:
        # 1. Load test audio from Phase 1 fixtures
        audio = np.load(f"test_fixtures/{test_name}_input.npy")

        # 2. Save as WAV for Swift CLI
        import scipy.io.wavfile
        scipy.io.wavfile.write("temp_input.wav", 44100, audio.T)

        # 3. Run Swift CLI to compute STFT
        result = subprocess.run([
            ".build/debug/htdemucs-cli", "stft",
            "temp_input.wav",
            "--output", "swift_stft.npz"
        ], check=True)

        # 4. Load Swift output
        swift_output = np.load("swift_stft.npz")
        swift_real = swift_output['real']
        swift_imag = swift_output['imag']

        # 5. Compute PyTorch reference
        audio_torch = torch.from_numpy(audio)
        stft_torch = torch.stft(
            audio_torch,
            n_fft=4096,
            hop_length=1024,
            window=torch.hann_window(4096),
            return_complex=True
        )

        # 6. Compare (rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            swift_real,
            stft_torch.real.numpy(),
            rtol=1e-5, atol=1e-6,
            err_msg=f"Real component mismatch for {test_name}"
        )
        np.testing.assert_allclose(
            swift_imag,
            stft_torch.imag.numpy(),
            rtol=1e-5, atol=1e-6,
            err_msg=f"Imaginary component mismatch for {test_name}"
        )

        print(f"✓ PyTorch parity validated for {test_name}")
```

**Level 4: Edge Case Tests (Swift + Python validation)**

```swift
// Tests/HTDemucsKitTests/EdgeCaseTests.swift

final class EdgeCaseTests: XCTestCase {
    func testMonoAudio() {
        // Single channel input
        let mono = generateSineWave(freq: 440, sampleRate: 44100, duration: 1.0)
        let (real, imag) = try! fft.stft(mono)
        let reconstructed = try! fft.istft(real: real, imag: imag)

        assertArraysClose(mono, reconstructed, tolerance: 1e-5)
    }

    func testVeryShortAudio() {
        // Less than one frame (< 4096 samples)
        let short = [Float](repeating: 1.0, count: 2048)

        XCTAssertThrowsError(try fft.stft(short)) { error in
            XCTAssertTrue(error is AudioFFTError)
        }
    }

    func testVeryLongAudio() {
        // 10 minutes at 44.1kHz = 26,460,000 samples
        let long = generateWhiteNoise(samples: 26_460_000)

        // Should complete without memory issues
        let (real, imag) = try! fft.stft(long)
        XCTAssertGreaterThan(real.count, 25000) // ~25,844 frames
    }

    func testClippedAudio() {
        // Audio at ±1.0 (typical digital audio limits)
        var audio = generateSineWave(freq: 440, sampleRate: 44100, duration: 1.0)
        audio = audio.map { $0 * 2.0 }.map { min(max($0, -1.0), 1.0) }

        let (real, imag) = try! fft.stft(audio)
        let reconstructed = try! fft.istft(real: real, imag: imag)

        // Should reconstruct clipped version
        assertArraysClose(audio, reconstructed, tolerance: 1e-5)
    }

    func testNonDivisibleLength() {
        // Length not evenly divisible by hop
        let audio = generateWhiteNoise(samples: 45000) // Not divisible by 1024

        let (real, imag) = try! fft.stft(audio)
        let reconstructed = try! fft.istft(real: real, imag: imag)

        // Should handle gracefully
        XCTAssertNotNil(reconstructed)
    }
}
```

### Test Execution Order

1. **Property tests** - Run on every save (instant feedback)
2. **Round-trip tests** - Fast validation of algorithm
3. **PyTorch parity** - Slower, validates numerical accuracy
4. **Edge cases** - Comprehensive, run before commit

## Layer 3: CoreML Integration

### ModelLoader

```swift
// Sources/HTDemucsKit/CoreML/ModelLoader.swift

import CoreML

public class ModelLoader {
    private let modelURL: URL
    private var model: MLModel?

    public init(modelPath: String) throws {
        self.modelURL = URL(fileURLWithPath: modelPath)

        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw ModelError.modelNotFound(modelPath)
        }
    }

    public func load() throws -> MLModel {
        if let cached = model {
            return cached
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU  // Phase 1 target

        let loadedModel = try MLModel(contentsOf: modelURL, configuration: config)
        self.model = loadedModel
        return loadedModel
    }
}
```

### InferenceEngine

```swift
// Sources/HTDemucsKit/CoreML/InferenceEngine.swift

import CoreML

public class InferenceEngine {
    private let model: MLModel

    public init(model: MLModel) {
        self.model = model
    }

    /// Run CoreML inference on spectrogram
    /// - Parameters:
    ///   - real: Real component [channels, freqBins, timeFrames]
    ///   - imag: Imaginary component [channels, freqBins, timeFrames]
    /// - Returns: Separation masks [sources, channels, freqBins, timeFrames]
    public func predict(real: [[Float]], imag: [[Float]]) throws -> [[[Float]]] {
        // Convert to MLMultiArray
        let realInput = try convertToMLMultiArray(real, shape: [1, 2, 2049, real[0].count])
        let imagInput = try convertToMLMultiArray(imag, shape: [1, 2, 2049, imag[0].count])

        // Create input features
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "spectrogram_real": MLFeatureValue(multiArray: realInput),
            "spectrogram_imag": MLFeatureValue(multiArray: imagInput)
        ])

        // Run prediction
        let output = try model.prediction(from: input)

        // Extract masks
        guard let masks = output.featureValue(for: "masks")?.multiArrayValue else {
            throw ModelError.invalidOutput
        }

        // Convert back to Swift arrays [6, 2, 2049, timeFrames]
        return convertFromMLMultiArray(masks)
    }
}
```

## Layer 4: Pipeline - Chunking and Integration

### ChunkProcessor

```swift
// Sources/HTDemucsKit/Pipeline/ChunkProcessor.swift

public class ChunkProcessor {
    private let chunkDuration: Float = 10.0  // seconds
    private let overlapDuration: Float = 1.0 // seconds
    private let sampleRate: Int = 44100

    private var chunkSamples: Int { Int(chunkDuration * Float(sampleRate)) }
    private var overlapSamples: Int { Int(overlapDuration * Float(sampleRate)) }
    private var hopSamples: Int { chunkSamples - 2 * overlapSamples }

    public func processInChunks(
        audio: [Float],
        processor: (ArraySlice<Float>) throws -> [Float]
    ) rethrows -> [Float] {
        var output = [Float](repeating: 0, count: audio.count)
        var weights = [Float](repeating: 0, count: audio.count)

        let totalChunks = (audio.count + hopSamples - 1) / hopSamples

        for chunkIdx in 0..<totalChunks {
            let start = chunkIdx * hopSamples
            let end = min(start + chunkSamples, audio.count)

            // Extract chunk
            let chunk = audio[start..<end]

            // Process chunk
            let processed = try processor(chunk)

            // Create blend window (linear crossfade in overlap regions)
            let window = createBlendWindow(
                chunkSize: processed.count,
                overlapSize: overlapSamples,
                isFirst: chunkIdx == 0,
                isLast: end >= audio.count
            )

            // Accumulate with blending
            for (i, value) in processed.enumerated() {
                output[start + i] += value * window[i]
                weights[start + i] += window[i]
            }
        }

        // Normalize by accumulated weights
        for i in 0..<audio.count where weights[i] > 0 {
            output[i] /= weights[i]
        }

        return output
    }

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
}
```

### SeparationPipeline (Full Integration)

```swift
// Sources/HTDemucsKit/Pipeline/SeparationPipeline.swift

public class SeparationPipeline {
    private let fft: AudioFFT
    private let inference: InferenceEngine
    private let chunker: ChunkProcessor

    public init(modelPath: String) throws {
        self.fft = try AudioFFT()

        let loader = try ModelLoader(modelPath: modelPath)
        let model = try loader.load()
        self.inference = InferenceEngine(model: model)

        self.chunker = ChunkProcessor()
    }

    /// Separate stereo audio into 6 stems
    /// - Parameter stereoAudio: [leftChannel, rightChannel]
    /// - Returns: Dictionary of [StemType: stereo audio]
    public func separate(stereoAudio: [[Float]]) throws -> [StemType: [[Float]]] {
        guard stereoAudio.count == 2 else {
            throw PipelineError.invalidChannelCount(stereoAudio.count)
        }

        let audioLength = stereoAudio[0].count

        // Initialize output buffers (6 stems × 2 channels)
        var outputs: [StemType: [[Float]]] = [:]
        for stem in StemType.allCases {
            outputs[stem] = [
                [Float](repeating: 0, count: audioLength),
                [Float](repeating: 0, count: audioLength)
            ]
        }

        var weights = [Float](repeating: 0, count: audioLength)

        // Process in chunks
        let totalChunks = (audioLength + chunker.hopSamples - 1) / chunker.hopSamples

        for chunkIdx in 0..<totalChunks {
            let start = chunkIdx * chunker.hopSamples
            let end = min(start + chunker.chunkSamples, audioLength)

            // Extract chunk (both channels)
            let leftChunk = Array(stereoAudio[0][start..<end])
            let rightChunk = Array(stereoAudio[1][start..<end])

            // STFT per channel
            let (leftReal, leftImag) = try fft.stft(leftChunk)
            let (rightReal, rightImag) = try fft.stft(rightChunk)

            // Stack for CoreML input [2, freqBins, timeFrames]
            let real = [leftReal, rightReal]
            let imag = [leftImag, rightImag]

            // CoreML inference → masks [6, 2, freqBins, timeFrames]
            let masks = try inference.predict(real: real, imag: imag)

            // Apply masks and iSTFT for each stem
            for (stemIdx, stem) in StemType.allCases.enumerated() {
                // Mask the spectrogram
                let maskedLeftReal = applyMask(leftReal, mask: masks[stemIdx][0])
                let maskedLeftImag = applyMask(leftImag, mask: masks[stemIdx][0])
                let maskedRightReal = applyMask(rightReal, mask: masks[stemIdx][1])
                let maskedRightImag = applyMask(rightImag, mask: masks[stemIdx][1])

                // iSTFT per channel
                let stemLeft = try fft.istft(real: maskedLeftReal, imag: maskedLeftImag)
                let stemRight = try fft.istft(real: maskedRightReal, imag: maskedRightImag)

                // Create blend window
                let window = chunker.createBlendWindow(
                    chunkSize: stemLeft.count,
                    overlapSize: chunker.overlapSamples,
                    isFirst: chunkIdx == 0,
                    isLast: end >= audioLength
                )

                // Accumulate into outputs
                for i in 0..<stemLeft.count {
                    outputs[stem]![0][start + i] += stemLeft[i] * window[i]
                    outputs[stem]![1][start + i] += stemRight[i] * window[i]
                }
            }

            // Accumulate weights (same for all stems)
            let window = chunker.createBlendWindow(
                chunkSize: end - start,
                overlapSize: chunker.overlapSamples,
                isFirst: chunkIdx == 0,
                isLast: end >= audioLength
            )
            for i in 0..<window.count {
                weights[start + i] += window[i]
            }
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

    private func applyMask(_ spectrogram: [[Float]], mask: [[Float]]) -> [[Float]] {
        return zip(spectrogram, mask).map { (spec, m) in
            zip(spec, m).map { $0 * $1 }
        }
    }
}

public enum StemType: CaseIterable {
    case drums, bass, vocals, other, piano, guitar
}
```

## CLI Tool

```swift
// Sources/htdemucs-cli/main.swift

import Foundation
import HTDemucsKit

@main
struct HTDemucsCLI {
    static func main() async throws {
        let args = CommandLine.arguments

        guard args.count >= 2 else {
            printUsage()
            return
        }

        switch args[1] {
        case "stft":
            try runSTFT(args: Array(args.dropFirst(2)))
        case "separate":
            try await runSeparation(args: Array(args.dropFirst(2)))
        case "validate":
            try runValidation(args: Array(args.dropFirst(2)))
        default:
            print("Unknown command: \(args[1])")
            printUsage()
        }
    }

    static func runSTFT(args: [String]) throws {
        // For PyTorch parity testing
        // ./htdemucs-cli stft input.wav --output stft.npz
    }

    static func runSeparation(args: [String]) async throws {
        // Full pipeline
        // ./htdemucs-cli separate input.wav --model model.mlpackage --output stems/
    }

    static func runValidation(args: [String]) throws {
        // Run all Swift tests
        // ./htdemucs-cli validate
    }

    static func printUsage() {
        print("""
        HTDemucs CLI - Swift STFT/iSTFT + CoreML Integration

        Commands:
          stft <input.wav> --output <stft.npz>
              Compute STFT for PyTorch parity testing

          separate <input.wav> --model <model.mlpackage> --output <dir/>
              Separate audio into 6 stems

          validate
              Run all validation tests
        """)
    }
}
```

## Development Plan

### Phase 2A: STFT/iSTFT Foundation (Week 1-2)

**Milestone 1: Basic STFT/iSTFT**
- Set up Swift package with library + CLI targets
- Implement AudioFFT class with vDSP
- Basic round-trip test passing

**Milestone 2: Property Validation**
- Parseval's theorem test passing
- COLA constraint test passing
- Real FFT symmetry test passing

**Milestone 3: PyTorch Parity**
- Python fixture generator working
- CLI exports STFT in NumPy format
- All 3 test cases match PyTorch (rtol=1e-5)

**Milestone 4: Edge Cases**
- Mono, long audio, clipped audio all passing
- Error handling comprehensive
- Ready for integration

### Phase 2B: CoreML Integration (Week 2-3)

**Milestone 5: Model Loading**
- ModelLoader successfully loads Phase 1 .mlpackage
- InferenceEngine runs on sample spectrograms
- Output shape validation

**Milestone 6: Pipeline Integration**
- SeparationPipeline end-to-end working
- Single 10s chunk processed correctly
- Output validated against Phase 1 Python

**Milestone 7: Chunking**
- ChunkProcessor handles long audio
- Overlap-add reconstruction working
- Crossfade quality validated

**Milestone 8: Full Validation**
- Process 3-minute song successfully
- Quality metrics: SNR > 60dB vs Python pipeline
- Performance profiling complete

## Success Criteria

### Quality Metrics

**STFT/iSTFT:**
- Round-trip error < 1e-5 (max absolute difference)
- PyTorch parity: rtol=1e-5, atol=1e-6 for all test cases
- All property tests passing (Parseval, COLA, symmetry)
- All edge cases handled gracefully

**End-to-End Pipeline:**
- SNR > 60dB compared to Phase 1 Python pipeline
- No audible artifacts in separated stems
- Consistent quality across all 6 stems

### Performance Targets

- STFT/iSTFT: < 100ms for 10s stereo audio (44.1kHz)
- Full separation: < 60s for 3-minute song (macOS M-series)
- Memory usage: < 2GB peak for 10-minute audio

### Platform Requirements

- macOS 13+ (for development/testing)
- iOS 18+ (for deployment target)
- Swift 6.0+
- Xcode 16+

## Risk Mitigation

### Known Risks

1. **vDSP precision issues**
   - Mitigation: Extensive validation against PyTorch
   - Fallback: Use higher precision intermediate buffers

2. **CoreML model loading on iOS**
   - Mitigation: Test with actual iOS device early
   - Fallback: Investigate bundle vs downloaded model strategies

3. **Memory pressure with long audio**
   - Mitigation: Profile memory usage, implement chunking carefully
   - Fallback: Reduce chunk size dynamically based on available memory

### To Investigate

- Optimal crossfade curve (linear vs cosine vs Hann)
- Metal Performance Shaders for FFT acceleration
- Real-time processing feasibility (< 10ms latency)

## References

- Phase 1 Design: `docs/plans/2026-02-01-htdemucs-coreml-design.md`
- Phase 1 Completion: `docs/phase1-completion-report.md`
- Apple vDSP Documentation: [Accelerate Framework](https://developer.apple.com/documentation/accelerate/vdsp)
- PyTorch STFT Reference: [torch.stft](https://pytorch.org/docs/stable/generated/torch.stft.html)

---

**Next Steps:**
1. Create git worktree for Phase 2 development
2. Initialize Swift package with library + CLI targets
3. Begin Milestone 1: Basic STFT/iSTFT implementation
