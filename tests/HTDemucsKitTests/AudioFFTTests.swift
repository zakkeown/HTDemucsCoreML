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
}
