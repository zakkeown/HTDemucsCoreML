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
        // Extract fixture name from path (e.g., "Resources/GoldenOutputs/silence.npz" -> "silence")
        let components = path.split(separator: "/")
        guard let filename = components.last else {
            throw NSError(
                domain: "PyTorchParityTests",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid path: \(path)"]
            )
        }

        let fixtureName = filename.replacingOccurrences(of: ".npz", with: "")
        return try TestSignals.loadNPZFixture(name: fixtureName)
    }
}
