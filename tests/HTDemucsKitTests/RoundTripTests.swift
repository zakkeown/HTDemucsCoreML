import XCTest
@testable import HTDemucsKit

final class RoundTripTests: XCTestCase {
    var fft: AudioFFT!

    override func setUp() {
        fft = try! AudioFFT()
    }

    func testSimpleConstantSignal() throws {
        // Test with a simple constant signal - easiest to debug
        let original = [Float](repeating: 1.0, count: 88200) // ~2 seconds at 44.1kHz

        let (real, imag) = try fft.stft(original)
        let reconstructed = try fft.istft(real: real, imag: imag)

        // STFT may not preserve exact length due to framing, but should be close
        XCTAssertEqual(reconstructed.count, original.count, accuracy: fft.hopLength)

        // Skip edge regions (first and last fftSize samples) due to windowing effects
        let skipEdge = fft.fftSize
        let compareStart = skipEdge
        let compareEnd = min(original.count, reconstructed.count) - skipEdge

        guard compareEnd > compareStart else {
            XCTFail("Signal too short to compare interior region")
            return
        }

        let maxError = zip(
            original[compareStart..<compareEnd],
            reconstructed[compareStart..<compareEnd]
        )
        .map { abs($0 - $1) }
        .max()!

        XCTAssertLessThan(maxError, 1e-5, "Round-trip error too large for constant signal (interior region)")
    }

    func testSilenceRoundTrip() throws {
        let original = TestSignals.silence(samples: 88200) // 2 seconds

        let (real, imag) = try fft.stft(original)
        let reconstructed = try fft.istft(real: real, imag: imag)

        // STFT may not preserve exact length due to framing
        // Just verify we're close
        XCTAssertEqual(reconstructed.count, original.count, accuracy: 200)

        // Check reconstruction accuracy
        let compareLength = min(original.count, reconstructed.count)
        let maxError = zip(
            original.prefix(compareLength),
            reconstructed.prefix(compareLength)
        )
        .map { abs($0 - $1) }
        .max()!

        XCTAssertLessThan(maxError, 1e-5, "Round-trip error too large for silence")
    }

    func testSineWaveRoundTrip() throws {
        let original = TestSignals.sine(frequency: 440, duration: 2.0)

        let (real, imag) = try fft.stft(original)
        let reconstructed = try fft.istft(real: real, imag: imag)

        // STFT may not preserve exact length due to framing
        XCTAssertEqual(reconstructed.count, original.count, accuracy: fft.hopLength)

        // Skip edge regions (first and last fftSize samples) due to windowing effects
        let skipEdge = fft.fftSize
        let compareStart = skipEdge
        let compareEnd = min(original.count, reconstructed.count) - skipEdge

        guard compareEnd > compareStart else {
            XCTFail("Signal too short to compare interior region")
            return
        }

        let maxError = zip(
            original[compareStart..<compareEnd],
            reconstructed[compareStart..<compareEnd]
        )
        .map { abs($0 - $1) }
        .max()!

        XCTAssertLessThan(maxError, 1e-5, "Round-trip error too large for sine wave (interior region)")
    }
}
