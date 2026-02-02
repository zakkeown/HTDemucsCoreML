# Phase 2: Swift STFT/iSTFT + CoreML Integration - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build battle-tested Swift STFT/iSTFT achieving PyTorch parity, then integrate with Phase 1 CoreML model for complete audio separation pipeline.

**Architecture:** Swift Accelerate/vDSP wrapper → Property-based validation → PyTorch numerical parity → CoreML integration → Chunked pipeline with overlap-add

**Tech Stack:** Swift 6.0+, Accelerate/vDSP, CoreML, XCTest, Python (validation harness)

---

## Task 1: Initialize Swift Package Structure

**Files:**
- Create: `Package.swift`
- Create: `Sources/HTDemucsKit/Audio/AudioTypes.swift`
- Create: `Sources/htdemucs-cli/main.swift`
- Create: `Tests/HTDemucsKitTests/AudioFFTTests.swift`

**Step 1: Create Package.swift manifest**

```bash
cd /Users/zakkeown/Code/HTDemucsCoreML/.worktrees/phase2-swift-integration
```

Create `Package.swift`:

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HTDemucsKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v18)
    ],
    products: [
        .library(
            name: "HTDemucsKit",
            targets: ["HTDemucsKit"]
        ),
        .executable(
            name: "htdemucs-cli",
            targets: ["htdemucs-cli"]
        )
    ],
    targets: [
        .target(
            name: "HTDemucsKit",
            dependencies: []
        ),
        .executableTarget(
            name: "htdemucs-cli",
            dependencies: ["HTDemucsKit"]
        ),
        .testTarget(
            name: "HTDemucsKitTests",
            dependencies: ["HTDemucsKit"]
        )
    ]
)
```

**Step 2: Create directory structure**

```bash
mkdir -p Sources/HTDemucsKit/Audio
mkdir -p Sources/HTDemucsKit/CoreML
mkdir -p Sources/HTDemucsKit/Pipeline
mkdir -p Sources/htdemucs-cli
mkdir -p Tests/HTDemucsKitTests
mkdir -p Resources/GoldenOutputs
```

**Step 3: Create placeholder AudioTypes.swift**

Create `Sources/HTDemucsKit/Audio/AudioTypes.swift`:

```swift
import Foundation

/// Common audio types and test utilities
public enum AudioTypes {
    /// Standard sample rate for testing
    public static let sampleRate = 44100
}

/// Test signal generators
public enum TestSignals {
    /// Generate sine wave
    public static func sine(frequency: Float, duration: Float, sampleRate: Int = AudioTypes.sampleRate) -> [Float] {
        let samples = Int(duration * Float(sampleRate))
        let angularFreq = 2 * Float.pi * frequency / Float(sampleRate)
        return (0..<samples).map { Float(sin(angularFreq * Float($0))) }
    }

    /// Generate white noise
    public static func whiteNoise(samples: Int) -> [Float] {
        return (0..<samples).map { _ in Float.random(in: -1...1) }
    }

    /// Generate silence
    public static func silence(samples: Int) -> [Float] {
        return [Float](repeating: 0, count: samples)
    }
}
```

**Step 4: Create placeholder CLI main.swift**

Create `Sources/htdemucs-cli/main.swift`:

```swift
import Foundation

@main
struct HTDemucsCLI {
    static func main() {
        print("HTDemucs CLI - Phase 2")
        print("Swift STFT/iSTFT + CoreML Integration")
        print("Version 0.1.0")
    }
}
```

**Step 5: Create placeholder test file**

Create `Tests/HTDemucsKitTests/AudioFFTTests.swift`:

```swift
import XCTest
@testable import HTDemucsKit

final class AudioFFTTests: XCTestCase {
    func testPlaceholder() {
        // Initial placeholder
        XCTAssertTrue(true)
    }
}
```

**Step 6: Verify package builds**

```bash
swift build
```

Expected output: Build succeeds, creates `.build/debug/` directory

**Step 7: Run initial test**

```bash
swift test
```

Expected output: "Test Suite 'All tests' passed"

**Step 8: Commit**

```bash
git add Package.swift Sources/ Tests/
git commit -m "$(cat <<'EOF'
feat: initialize Swift package structure for Phase 2

- Add Package.swift with HTDemucsKit library + CLI targets
- Create directory structure for Audio/CoreML/Pipeline layers
- Add AudioTypes and TestSignals for test utilities
- Add placeholder CLI and test files
- Verify build and tests pass

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement AudioFFT Skeleton

**Files:**
- Create: `Sources/HTDemucsKit/Audio/AudioFFT.swift`
- Modify: `Tests/HTDemucsKitTests/AudioFFTTests.swift`

**Step 1: Write failing initialization test**

Modify `Tests/HTDemucsKitTests/AudioFFTTests.swift`:

```swift
import XCTest
@testable import HTDemucsKit

final class AudioFFTTests: XCTestCase {
    func testInitialization() throws {
        // Should create AudioFFT with correct configuration
        let fft = try AudioFFT()

        XCTAssertEqual(fft.fftSize, 4096)
        XCTAssertEqual(fft.hopLength, 1024)
    }

    func testSTFTThrowsOnShortAudio() {
        let fft = try! AudioFFT()
        let shortAudio = [Float](repeating: 0, count: 2048) // Less than fftSize

        XCTAssertThrowsError(try fft.stft(shortAudio)) { error in
            guard case AudioFFTError.audioTooShort = error else {
                XCTFail("Expected audioTooShort error")
                return
            }
        }
    }
}
```

**Step 2: Run test to verify it fails**

```bash
swift test
```

Expected: Compilation error "Cannot find type 'AudioFFT' in scope"

**Step 3: Implement AudioFFT skeleton**

Create `Sources/HTDemucsKit/Audio/AudioFFT.swift`:

```swift
import Accelerate
import Foundation

/// Swift wrapper for vDSP FFT operations
/// Provides STFT/iSTFT with PyTorch-compatible parameters
public class AudioFFT {
    // MARK: - Configuration (matches PyTorch)
    public let fftSize: Int = 4096
    public let hopLength: Int = 1024
    private let log2n: vDSP_Length

    // MARK: - vDSP State
    private var fftSetup: FFTSetup
    private var window: [Float]

    // MARK: - Working Buffers
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

        // Pre-compute Hann window
        self.window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))

        // Allocate working buffers
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
    /// - Throws: AudioFFTError if audio invalid
    public func stft(_ audio: [Float]) throws -> (real: [[Float]], imag: [[Float]]) {
        // Validate input
        guard audio.count >= fftSize else {
            throw AudioFFTError.audioTooShort(audio.count, fftSize)
        }
        guard audio.allSatisfy({ $0.isFinite }) else {
            throw AudioFFTError.invalidAudioData("NaN or Inf values")
        }

        // TODO: Implement STFT logic
        return (real: [], imag: [])
    }

    /// Compute inverse Short-Time Fourier Transform
    /// - Parameters:
    ///   - real: Real component [numFrames][numBins]
    ///   - imag: Imaginary component [numFrames][numBins]
    /// - Returns: Reconstructed audio samples
    public func istft(real: [[Float]], imag: [[Float]]) throws -> [Float] {
        guard real.count == imag.count else {
            throw AudioFFTError.mismatchedDimensions
        }

        // TODO: Implement iSTFT logic
        return []
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

**Step 4: Run tests to verify they pass**

```bash
swift test
```

Expected: "Test Suite 'All tests' passed. Executed 2 tests"

**Step 5: Commit**

```bash
git add Sources/HTDemucsKit/Audio/AudioFFT.swift Tests/HTDemucsKitTests/AudioFFTTests.swift
git commit -m "$(cat <<'EOF'
feat: add AudioFFT skeleton with error handling

- Implement AudioFFT class with vDSP setup/teardown
- Pre-compute Hann window in init
- Add validation for audio length and finite values
- Define AudioFFTError enum with descriptive messages
- Add initialization and validation tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement vDSP Real FFT

**Files:**
- Modify: `Sources/HTDemucsKit/Audio/AudioFFT.swift`
- Modify: `Tests/HTDemucsKitTests/AudioFFTTests.swift`

**Step 1: Write failing FFT test**

Add to `Tests/HTDemucsKitTests/AudioFFTTests.swift`:

```swift
func testRealFFTOutputShape() throws {
    let fft = try AudioFFT()

    // Single frame worth of audio
    let audio = TestSignals.sine(frequency: 440, duration: 0.1, sampleRate: 44100)

    let (real, imag) = try fft.stft(audio)

    // Should have frames
    XCTAssertGreaterThan(real.count, 0)
    XCTAssertEqual(real.count, imag.count)

    // Each frame should have fftSize/2 + 1 bins (real FFT)
    if let firstFrame = real.first {
        XCTAssertEqual(firstFrame.count, 2049) // 4096/2 + 1
    }
}
```

**Step 2: Run test to verify it fails**

```bash
swift test --filter testRealFFTOutputShape
```

Expected: Test fails with "expected to be greater than 0, got 0"

**Step 3: Implement performRealFFT helper**

Modify `Sources/HTDemucsKit/Audio/AudioFFT.swift`, replace the `performRealFFT` stub:

```swift
// MARK: - Private Helpers

private func performRealFFT(_ input: [Float]) -> ([Float], [Float]) {
    let halfSize = fftSize / 2
    let numBins = halfSize + 1

    // Convert to interleaved format for vDSP
    var interleaved = input

    // Create split complex structure
    var splitComplex = DSPSplitComplex(
        realp: &splitComplexReal,
        imagp: &splitComplexImag
    )

    // Convert real input to split complex
    interleaved.withUnsafeBytes { inputPtr in
        let inputFloat = inputPtr.bindMemory(to: Float.self)
        vDSP_ctoz(
            UnsafePointer<DSPComplex>(OpaquePointer(inputFloat.baseAddress!)),
            2,
            &splitComplex,
            1,
            vDSP_Length(halfSize)
        )
    }

    // Perform forward FFT
    vDSP_fft_zrip(
        fftSetup,
        &splitComplex,
        1,
        log2n,
        FFTDirection(FFT_FORWARD)
    )

    // Scale output (vDSP doesn't scale forward transform)
    var scale = Float(0.5)
    vDSP_vsmul(splitComplexReal, 1, &scale, &splitComplexReal, 1, vDSP_Length(halfSize))
    vDSP_vsmul(splitComplexImag, 1, &scale, &splitComplexImag, 1, vDSP_Length(halfSize))

    // Extract bins (DC and Nyquist packed in splitComplexReal[0], splitComplexImag[0])
    var realOutput = [Float](repeating: 0, count: numBins)
    var imagOutput = [Float](repeating: 0, count: numBins)

    // DC bin (purely real)
    realOutput[0] = splitComplexReal[0]
    imagOutput[0] = 0

    // Positive frequencies
    for i in 1..<halfSize {
        realOutput[i] = splitComplexReal[i]
        imagOutput[i] = splitComplexImag[i]
    }

    // Nyquist bin (purely real, packed in imaginary DC)
    realOutput[halfSize] = splitComplexImag[0]
    imagOutput[halfSize] = 0

    return (realOutput, imagOutput)
}
```

**Step 4: Implement STFT loop**

Modify the `stft` method implementation:

```swift
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

    var realOutput: [[Float]] = []
    var imagOutput: [[Float]] = []

    // Process each frame
    for frameIdx in 0..<numFrames {
        let start = frameIdx * hopLength
        let end = start + fftSize

        // Extract frame
        let frame = Array(audio[start..<end])

        // Apply window
        vDSP_vmul(frame, 1, window, 1, &windowedFrame, 1, vDSP_Length(fftSize))

        // Perform FFT
        let (frameReal, frameImag) = performRealFFT(windowedFrame)

        realOutput.append(frameReal)
        imagOutput.append(frameImag)
    }

    return (real: realOutput, imag: imagOutput)
}
```

**Step 5: Run test to verify it passes**

```bash
swift test --filter testRealFFTOutputShape
```

Expected: Test passes

**Step 6: Run all tests**

```bash
swift test
```

Expected: All 3 tests pass

**Step 7: Commit**

```bash
git add Sources/HTDemucsKit/Audio/AudioFFT.swift Tests/HTDemucsKitTests/AudioFFTTests.swift
git commit -m "$(cat <<'EOF'
feat: implement vDSP real FFT for STFT

- Add performRealFFT helper using vDSP_fft_zrip
- Handle DC and Nyquist bin packing/unpacking
- Apply Hann window before FFT
- Implement STFT frame loop with hop length
- Add test for FFT output shape validation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement vDSP Inverse Real FFT

**Files:**
- Modify: `Sources/HTDemucsKit/Audio/AudioFFT.swift`
- Create: `Tests/HTDemucsKitTests/RoundTripTests.swift`

**Step 1: Write failing round-trip test**

Create `Tests/HTDemucsKitTests/RoundTripTests.swift`:

```swift
import XCTest
@testable import HTDemucsKit

final class RoundTripTests: XCTestCase {
    var fft: AudioFFT!

    override func setUp() {
        fft = try! AudioFFT()
    }

    func testSilenceRoundTrip() throws {
        let original = TestSignals.silence(samples: 88200) // 2 seconds

        let (real, imag) = try fft.stft(original)
        let reconstructed = try fft.istft(real: real, imag: imag)

        // Check reconstruction length
        XCTAssertEqual(reconstructed.count, original.count)

        // Check reconstruction accuracy
        let maxError = zip(original, reconstructed)
            .map { abs($0 - $1) }
            .max()!

        XCTAssertLessThan(maxError, 1e-5, "Round-trip error too large for silence")
    }

    func testSineWaveRoundTrip() throws {
        let original = TestSignals.sine(frequency: 440, duration: 2.0)

        let (real, imag) = try fft.stft(original)
        let reconstructed = try fft.istft(real: real, imag: imag)

        // Compare (allowing for edge effects)
        let compareLength = min(original.count, reconstructed.count)
        let maxError = zip(
            original.prefix(compareLength),
            reconstructed.prefix(compareLength)
        )
        .map { abs($0 - $1) }
        .max()!

        XCTAssertLessThan(maxError, 1e-5, "Round-trip error too large for sine wave")
    }
}
```

**Step 2: Run test to verify it fails**

```bash
swift test --filter RoundTripTests
```

Expected: Test fails (iSTFT returns empty array)

**Step 3: Implement performInverseRealFFT helper**

Modify `Sources/HTDemucsKit/Audio/AudioFFT.swift`, replace `performInverseRealFFT` stub:

```swift
private func performInverseRealFFT(_ real: [Float], _ imag: [Float]) -> [Float] {
    let halfSize = fftSize / 2

    // Pack DC and Nyquist into split complex format
    // DC → splitComplexReal[0], Nyquist → splitComplexImag[0]
    splitComplexReal[0] = real[0]
    splitComplexImag[0] = real[halfSize]

    // Pack positive frequencies
    for i in 1..<halfSize {
        splitComplexReal[i] = real[i]
        splitComplexImag[i] = imag[i]
    }

    var splitComplex = DSPSplitComplex(
        realp: &splitComplexReal,
        imagp: &splitComplexImag
    )

    // Perform inverse FFT
    vDSP_fft_zrip(
        fftSetup,
        &splitComplex,
        1,
        log2n,
        FFTDirection(FFT_INVERSE)
    )

    // Convert split complex back to real
    var output = [Float](repeating: 0, count: fftSize)
    output.withUnsafeMutableBytes { outputPtr in
        let outputFloat = outputPtr.bindMemory(to: Float.self)
        vDSP_ztoc(
            &splitComplex,
            1,
            UnsafeMutablePointer<DSPComplex>(OpaquePointer(outputFloat.baseAddress!)),
            2,
            vDSP_Length(halfSize)
        )
    }

    // Scale output (vDSP requires manual scaling for inverse)
    var scale = Float(1.0) / Float(fftSize)
    vDSP_vsmul(output, 1, &scale, &output, 1, vDSP_Length(fftSize))

    return output
}
```

**Step 4: Implement iSTFT with overlap-add**

Modify the `istft` method:

```swift
public func istft(real: [[Float]], imag: [[Float]]) throws -> [Float] {
    guard real.count == imag.count else {
        throw AudioFFTError.mismatchedDimensions
    }

    let numFrames = real.count
    guard numFrames > 0 else {
        return []
    }

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
    for i in 0..<outputLength where windowSum[i] > 1e-8 {
        output[i] /= windowSum[i]
    }

    return output
}
```

**Step 5: Run tests to verify they pass**

```bash
swift test --filter RoundTripTests
```

Expected: Both round-trip tests pass

**Step 6: Run all tests**

```bash
swift test
```

Expected: All tests pass

**Step 7: Commit**

```bash
git add Sources/HTDemucsKit/Audio/AudioFFT.swift Tests/HTDemucsKitTests/RoundTripTests.swift
git commit -m "$(cat <<'EOF'
feat: implement vDSP inverse real FFT for iSTFT

- Add performInverseRealFFT with DC/Nyquist unpacking
- Implement iSTFT with overlap-add reconstruction
- Apply COLA-compliant window normalization
- Add round-trip tests for silence and sine wave
- Verify reconstruction error < 1e-5

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement Property Tests

**Files:**
- Create: `Tests/HTDemucsKitTests/STFTPropertyTests.swift`

**Step 1: Write Parseval's theorem test**

Create `Tests/HTDemucsKitTests/STFTPropertyTests.swift`:

```swift
import XCTest
@testable import HTDemucsKit

final class STFTPropertyTests: XCTestCase {
    var fft: AudioFFT!

    override func setUp() {
        fft = try! AudioFFT()
    }

    func testParsevalTheorem() throws {
        // Generate test signal
        let audio = TestSignals.sine(frequency: 440, duration: 1.0)

        // Compute STFT
        let (real, imag) = try fft.stft(audio)

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
        let scaledFreqEnergy = freqEnergy * Float(fft.hopLength)
        XCTAssertEqual(
            scaledFreqEnergy,
            timeEnergy,
            accuracy: timeEnergy * 0.01,
            "Parseval's theorem violation"
        )
    }

    func testCOLAConstraint() throws {
        // Verify window overlap-add gives constant
        let windowSum = computeOverlapAddSum(
            window: fft.window,
            hop: fft.hopLength,
            length: fft.fftSize * 4
        )

        // Should be constant (within numerical precision)
        let mean = windowSum.reduce(0, +) / Float(windowSum.count)
        let maxDeviation = windowSum.map { abs($0 - mean) }.max()!

        XCTAssertLessThan(maxDeviation, 1e-6, "COLA constraint violated")
    }

    func testRealFFTSymmetry() throws {
        let audio = TestSignals.whiteNoise(samples: 44100)
        let (real, imag) = try fft.stft(audio)

        // For each frame, DC and Nyquist should have zero imaginary part
        for frameIdx in 0..<real.count {
            let numBins = real[frameIdx].count

            XCTAssertEqual(imag[frameIdx][0], 0, accuracy: 1e-6,
                         "DC bin should be purely real")
            XCTAssertEqual(imag[frameIdx][numBins-1], 0, accuracy: 1e-6,
                         "Nyquist bin should be purely real")
        }
    }

    // MARK: - Helpers

    private func computeOverlapAddSum(window: [Float], hop: Int, length: Int) -> [Float] {
        var sum = [Float](repeating: 0, count: length)

        var offset = 0
        while offset + window.count <= length {
            for i in 0..<window.count {
                sum[offset + i] += window[i] * window[i]
            }
            offset += hop
        }

        return sum
    }
}
```

**Step 2: Run test to verify it passes**

```bash
swift test --filter STFTPropertyTests
```

Expected: All 3 property tests pass

**Step 3: Commit**

```bash
git add Tests/HTDemucsKitTests/STFTPropertyTests.swift
git commit -m "$(cat <<'EOF'
test: add STFT property tests

- Add Parseval's theorem validation (energy conservation)
- Add COLA constraint test (constant overlap-add)
- Add real FFT symmetry test (DC and Nyquist purely real)
- All property tests passing

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Create Python Fixture Generator

**Files:**
- Create: `scripts/generate_swift_fixtures.py`
- Create: `Resources/GoldenOutputs/.gitkeep`

**Step 1: Write fixture generator script**

Create `scripts/generate_swift_fixtures.py`:

```python
#!/usr/bin/env python3
"""Generate golden STFT outputs from PyTorch for Swift validation."""

import numpy as np
import torch
import argparse
from pathlib import Path


def generate_test_audio(test_name: str, sample_rate: int = 44100) -> np.ndarray:
    """Generate test audio signals."""
    if test_name == "silence":
        return np.zeros(88200, dtype=np.float32)

    elif test_name == "sine_440hz":
        duration = 2.0
        samples = int(duration * sample_rate)
        t = np.arange(samples) / sample_rate
        freq = 440.0
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    elif test_name == "white_noise":
        rng = np.random.RandomState(42)  # Fixed seed
        return rng.uniform(-1, 1, 88200).astype(np.float32)

    else:
        raise ValueError(f"Unknown test: {test_name}")


def compute_pytorch_stft(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute STFT using PyTorch (reference implementation)."""
    audio_torch = torch.from_numpy(audio)

    # Match Swift parameters
    stft_complex = torch.stft(
        audio_torch,
        n_fft=4096,
        hop_length=1024,
        window=torch.hann_window(4096),
        return_complex=True,
        center=False  # Match Swift (no padding)
    )

    # Extract real and imaginary parts
    real = stft_complex.real.numpy().astype(np.float32)
    imag = stft_complex.imag.numpy().astype(np.float32)

    # Transpose to [numFrames, numBins] (Swift format)
    real = real.T
    imag = imag.T

    return real, imag


def main():
    parser = argparse.ArgumentParser(description="Generate golden STFT fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Resources/GoldenOutputs"),
        help="Output directory for fixtures"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = ["silence", "sine_440hz", "white_noise"]

    for test_name in test_cases:
        print(f"Generating {test_name}...")

        # Generate audio
        audio = generate_test_audio(test_name)

        # Compute PyTorch STFT (golden reference)
        real, imag = compute_pytorch_stft(audio)

        # Save as NPZ
        output_path = args.output_dir / f"{test_name}.npz"
        np.savez(
            output_path,
            audio=audio,
            stft_real=real,
            stft_imag=imag
        )

        print(f"  Audio: {audio.shape}, STFT: {real.shape}")
        print(f"  Saved to {output_path}")

    print(f"\n✓ Generated {len(test_cases)} golden fixtures")


if __name__ == "__main__":
    main()
```

**Step 2: Make script executable**

```bash
chmod +x scripts/generate_swift_fixtures.py
```

**Step 3: Create output directory**

```bash
mkdir -p Resources/GoldenOutputs
touch Resources/GoldenOutputs/.gitkeep
```

**Step 4: Run fixture generator**

```bash
cd /Users/zakkeown/Code/HTDemucsCoreML/.worktrees/phase2-swift-integration
python3 scripts/generate_swift_fixtures.py
```

Expected output:
```
Generating silence...
  Audio: (88200,), STFT: (84, 2049)
  Saved to Resources/GoldenOutputs/silence.npz
Generating sine_440hz...
  Audio: (88200,), STFT: (84, 2049)
  Saved to Resources/GoldenOutputs/sine_440hz.npz
Generating white_noise...
  Audio: (88200,), STFT: (84, 2049)
  Saved to Resources/GoldenOutputs/white_noise.npz

✓ Generated 3 golden fixtures
```

**Step 5: Verify fixtures created**

```bash
ls -lh Resources/GoldenOutputs/
```

Expected: See `.npz` files for each test case

**Step 6: Commit**

```bash
git add scripts/generate_swift_fixtures.py Resources/GoldenOutputs/
git commit -m "$(cat <<'EOF'
feat: add Python fixture generator for PyTorch parity

- Create script to generate golden STFT outputs from PyTorch
- Support silence, sine wave, and white noise test cases
- Match Swift parameters (4096 FFT, 1024 hop, no center padding)
- Save audio + STFT (real/imag) as NPZ files
- Generate fixtures for validation tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Implement PyTorch Parity Tests

**Files:**
- Create: `Tests/HTDemucsKitTests/PyTorchParityTests.swift`

**Step 1: Write PyTorch parity test**

Create `Tests/HTDemucsKitTests/PyTorchParityTests.swift`:

```swift
import XCTest
@testable import HTDemucsKit

final class PyTorchParityTests: XCTestCase {
    var fft: AudioFFT!

    override func setUp() {
        fft = try! AudioFFT()
    }

    func testSilenceMatchesPyTorch() throws {
        try verifyPyTorchParity(testCase: "silence")
    }

    func testSineWaveMatchesPyTorch() throws {
        try verifyPyTorchParity(testCase: "sine_440hz")
    }

    func testWhiteNoiseMatchesPyTorch() throws {
        try verifyPyTorchParity(testCase: "white_noise")
    }

    // MARK: - Helpers

    private func verifyPyTorchParity(testCase: String) throws {
        // Load golden fixture
        let goldenPath = "Resources/GoldenOutputs/\(testCase).npz"
        let (audio, pytorchReal, pytorchImag) = try loadGoldenFixture(path: goldenPath)

        // Compute Swift STFT
        let (swiftReal, swiftImag) = try fft.stft(audio)

        // Verify shapes match
        XCTAssertEqual(swiftReal.count, pytorchReal.count,
                      "\(testCase): frame count mismatch")
        XCTAssertEqual(swiftReal[0].count, pytorchReal[0].count,
                      "\(testCase): bin count mismatch")

        // Compare values (rtol=1e-5, atol=1e-6)
        let rtol: Float = 1e-5
        let atol: Float = 1e-6

        for (frameIdx, (sr, pr)) in zip(swiftReal, pytorchReal).enumerated() {
            for (binIdx, (sv, pv)) in zip(sr, pr).enumerated() {
                let tolerance = atol + rtol * abs(pv)
                let error = abs(sv - pv)

                XCTAssertLessThanOrEqual(
                    error,
                    tolerance,
                    "\(testCase) real mismatch at frame \(frameIdx), bin \(binIdx): " +
                    "Swift=\(sv), PyTorch=\(pv), error=\(error)"
                )
            }
        }

        for (frameIdx, (si, pi)) in zip(swiftImag, pytorchImag).enumerated() {
            for (binIdx, (sv, pv)) in zip(si, pi).enumerated() {
                let tolerance = atol + rtol * abs(pv)
                let error = abs(sv - pv)

                XCTAssertLessThanOrEqual(
                    error,
                    tolerance,
                    "\(testCase) imag mismatch at frame \(frameIdx), bin \(binIdx): " +
                    "Swift=\(sv), PyTorch=\(pv), error=\(error)"
                )
            }
        }

        print("✓ \(testCase): PyTorch parity verified")
    }

    private func loadGoldenFixture(path: String) throws -> ([Float], [[Float]], [[Float]]) {
        // Note: This requires Python NumPy loading, which isn't available in pure Swift
        // For now, we'll throw an error and implement this when we add NumPy bridge
        throw XCTSkip("NumPy loading not yet implemented - requires Python bridge")
    }
}
```

**Step 2: Run test to verify skip behavior**

```bash
swift test --filter PyTorchParityTests
```

Expected: Tests skipped with "NumPy loading not yet implemented"

**Step 3: Commit (tests skipped for now, will implement NumPy bridge later)**

```bash
git add Tests/HTDemucsKitTests/PyTorchParityTests.swift
git commit -m "$(cat <<'EOF'
test: add PyTorch parity tests (skipped, pending NumPy bridge)

- Add test structure for PyTorch numerical parity
- Implement tolerance checking (rtol=1e-5, atol=1e-6)
- Tests for silence, sine wave, white noise
- Currently skipped - requires Python/NumPy bridge for fixture loading
- Will implement bridge in next task

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Implement Edge Case Tests

**Files:**
- Create: `Tests/HTDemucsKitTests/EdgeCaseTests.swift`

**Step 1: Write edge case tests**

Create `Tests/HTDemucsKitTests/EdgeCaseTests.swift`:

```swift
import XCTest
@testable import HTDemucsKit

final class EdgeCaseTests: XCTestCase {
    var fft: AudioFFT!

    override func setUp() {
        fft = try! AudioFFT()
    }

    func testVeryShortAudioThrows() {
        // Less than one frame (< 4096 samples)
        let short = [Float](repeating: 1.0, count: 2048)

        XCTAssertThrowsError(try fft.stft(short)) { error in
            guard case AudioFFTError.audioTooShort = error else {
                XCTFail("Expected audioTooShort error, got \(error)")
                return
            }
        }
    }

    func testExactlyOneFrame() throws {
        // Exactly fftSize samples
        let audio = TestSignals.sine(frequency: 440, duration: 4096.0 / 44100.0)

        let (real, imag) = try fft.stft(audio)

        XCTAssertEqual(real.count, 1, "Should have exactly 1 frame")
        XCTAssertEqual(real[0].count, 2049)
    }

    func testVeryLongAudio() throws {
        // 10 minutes at 44.1kHz = 26,460,000 samples
        let long = TestSignals.whiteNoise(samples: 26_460_000)

        // Should complete without memory issues
        let (real, imag) = try fft.stft(long)

        // ~25,844 frames expected: (26460000 - 4096) / 1024 + 1
        XCTAssertGreaterThan(real.count, 25000)
        XCTAssertLessThan(real.count, 26000)
    }

    func testClippedAudio() throws {
        // Audio at ±1.0 (typical digital audio limits)
        var audio = TestSignals.sine(frequency: 440, duration: 1.0)
        audio = audio.map { ($0 * 2.0).clamped(to: -1.0...1.0) }

        let (real, imag) = try fft.stft(audio)
        let reconstructed = try fft.istft(real: real, imag: imag)

        // Should reconstruct clipped version accurately
        let maxError = zip(audio.prefix(reconstructed.count), reconstructed)
            .map { abs($0 - $1) }
            .max()!

        XCTAssertLessThan(maxError, 1e-5)
    }

    func testNonDivisibleLength() throws {
        // Length not evenly divisible by hop
        let audio = TestSignals.whiteNoise(samples: 45000) // Not divisible by 1024

        let (real, imag) = try fft.stft(audio)
        let reconstructed = try fft.istft(real: real, imag: imag)

        // Should handle gracefully
        XCTAssertGreaterThan(real.count, 0)
        XCTAssertGreaterThan(reconstructed.count, 0)
    }

    func testInvalidAudioData() {
        // Audio with NaN values
        var audio = TestSignals.sine(frequency: 440, duration: 1.0)
        audio[audio.count / 2] = Float.nan

        XCTAssertThrowsError(try fft.stft(audio)) { error in
            guard case AudioFFTError.invalidAudioData = error else {
                XCTFail("Expected invalidAudioData error")
                return
            }
        }
    }
}

// MARK: - Helpers

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        return min(max(self, range.lowerBound), range.upperBound)
    }
}
```

**Step 2: Run tests to verify they pass**

```bash
swift test --filter EdgeCaseTests
```

Expected: All edge case tests pass

**Step 3: Commit**

```bash
git add Tests/HTDemucsKitTests/EdgeCaseTests.swift
git commit -m "$(cat <<'EOF'
test: add comprehensive edge case tests

- Test very short audio (should throw)
- Test exactly one frame
- Test very long audio (10 minutes, memory stress)
- Test clipped audio (±1.0 limits)
- Test non-divisible length
- Test invalid data (NaN detection)
- All edge cases handled gracefully

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary: Phase 2A Complete

At this point, Phase 2A (STFT/iSTFT Foundation) is complete with:

- ✅ Swift package structure initialized
- ✅ AudioFFT class with vDSP FFT/iFFT
- ✅ Round-trip validation (error < 1e-5)
- ✅ Property tests (Parseval, COLA, symmetry)
- ✅ Edge case tests (all passing)
- ✅ Python fixture generator created
- ⏸️ PyTorch parity tests (structure ready, needs NumPy bridge)

**Next Tasks for Phase 2B (CoreML Integration):**

- Task 9: Implement NumPy fixture loader (Swift/Python bridge)
- Task 10: Complete PyTorch parity validation
- Task 11: Implement ModelLoader
- Task 12: Implement InferenceEngine
- Task 13: Implement ChunkProcessor
- Task 14: Implement SeparationPipeline
- Task 15: Build CLI tool
- Task 16: End-to-end validation

---

## Execution Notes

- Use `@superpowers:test-driven-development` for each implementation task
- Run `swift test` after every commit to ensure no regressions
- Use `swift build -c release` for performance testing
- Create small, focused commits (< 200 lines per commit)
- Test on real audio files before considering Phase 2B complete
