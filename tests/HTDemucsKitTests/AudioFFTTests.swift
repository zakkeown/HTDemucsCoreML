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
