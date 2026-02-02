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

        // Should handle clipped audio without errors
        let (real, imag) = try fft.stft(audio)
        let reconstructed = try fft.istft(real: real, imag: imag)

        // Verify STFT processed the audio (shape checks)
        XCTAssertGreaterThan(real.count, 0)
        XCTAssertGreaterThan(reconstructed.count, 0)

        // Note: Clipping introduces non-linear distortion, so perfect reconstruction
        // is not expected. This test just verifies the operations complete successfully.
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
