import Foundation
import Testing
@testable import HTDemucsKit

@Suite("AudioEncoder Tests")
struct AudioEncoderTests {
    @Test func encodeToWAV() throws {
        let encoder = AudioEncoder()
        let leftChannel: [Float] = Array(repeating: 0.1, count: 44100)
        let rightChannel: [Float] = Array(repeating: -0.1, count: 44100)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test-output.wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try encoder.encode(
            leftChannel: leftChannel,
            rightChannel: rightChannel,
            sampleRate: 44100,
            format: .wav,
            destination: tempURL
        )

        // Verify file was created
        #expect(FileManager.default.fileExists(atPath: tempURL.path))

        // Verify file has content
        let data = try Data(contentsOf: tempURL)
        #expect(data.count > 1000)
    }

    @Test func encodeRoundTrip() throws {
        let encoder = AudioEncoder()
        let decoder = AudioDecoder()

        // Create simple test signal (pad to at least 1024 samples for frame processing)
        let originalLeft: [Float] = Array(repeating: 0.1, count: 2048)
        let originalRight: [Float] = Array(repeating: 0.2, count: 2048)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("roundtrip.wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        // Encode
        try encoder.encode(
            leftChannel: originalLeft,
            rightChannel: originalRight,
            sampleRate: 44100,
            format: .wav,
            destination: tempURL
        )

        // Decode
        let decoded = try decoder.decode(fileURL: tempURL)

        // Verify sample rate
        #expect(abs(decoded.sampleRate - 44100) < 0.1)

        // Verify length (allow small differences due to encoding)
        #expect(abs(decoded.frameCount - originalLeft.count) < 10)
    }

    @Test func encodeToInvalidPath() {
        let encoder = AudioEncoder()
        let left: [Float] = [0.1]
        let right: [Float] = [0.1]

        let invalidURL = URL(fileURLWithPath: "/invalid/path/output.wav")

        #expect(throws: AudioError.self) {
            try encoder.encode(
                leftChannel: left,
                rightChannel: right,
                sampleRate: 44100,
                format: .wav,
                destination: invalidURL
            )
        }
    }
}
