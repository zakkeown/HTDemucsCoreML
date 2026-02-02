import XCTest
import Accelerate
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

        // For each frame, verify Parseval's theorem holds
        // We'll check a few representative frames to avoid computing all
        let framesToCheck = [0, 10, 20, 30]

        for frameIdx in framesToCheck where frameIdx < real.count {
            // Frequency domain energy for this frame
            var freqEnergyRaw: Float = 0
            for (rVal, iVal) in zip(real[frameIdx], imag[frameIdx]) {
                freqEnergyRaw += rVal * rVal + iVal * iVal
            }

            // vDSP real FFT has a scaling such that for a windowed signal:
            // time_energy = fft_energy / (2 * N)
            // This accounts for the FFT's inherent scaling and the real FFT packing
            let fftScaling = Float(2 * fft.fftSize)
            let freqEnergy = freqEnergyRaw / fftScaling

            // Time domain energy for this frame (windowed)
            let start = frameIdx * fft.hopLength
            let end = start + fft.fftSize
            guard end <= audio.count else { continue }

            let frame = Array(audio[start..<end])
            var windowedFrame = [Float](repeating: 0, count: fft.fftSize)
            vDSP_vmul(frame, 1, fft.window, 1, &windowedFrame, 1, vDSP_Length(fft.fftSize))

            let timeEnergy = windowedFrame.map { $0 * $0 }.reduce(0, +)

            XCTAssertEqual(
                freqEnergy,
                timeEnergy,
                accuracy: timeEnergy * 0.01,
                "Parseval's theorem violation for frame \(frameIdx)"
            )
        }
    }

    func testCOLAConstraint() throws {
        // Verify window overlap-add gives constant
        let testLength = fft.fftSize * 8
        let windowSum = computeOverlapAddSum(
            window: fft.window,
            hop: fft.hopLength,
            length: testLength
        )

        // Check steady-state region (skip edge effects)
        let steadyStart = fft.fftSize
        let steadyEnd = testLength - fft.fftSize
        let steadyRegion = Array(windowSum[steadyStart..<steadyEnd])

        // Should be constant (within numerical precision)
        let mean = steadyRegion.reduce(0, +) / Float(steadyRegion.count)
        let maxDeviation = steadyRegion.map { abs($0 - mean) }.max()!

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
