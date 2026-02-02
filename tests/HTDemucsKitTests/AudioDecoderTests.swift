import Testing
import Foundation
@testable import HTDemucsKit

@Suite("AudioDecoder Tests")
struct AudioDecoderTests {
    @Test("Decode WAV file")
    func testDecodeWAVFile() throws {
        let fixturePath = try resolveFixturePath("sine-440hz-1s.wav")
        let decoder = AudioDecoder()

        let decoded = try decoder.decode(fileURL: URL(fileURLWithPath: fixturePath))

        #expect(abs(decoded.sampleRate - 44100) < 0.1)
        #expect(decoded.channelCount == 2)
        #expect(abs(decoded.duration - 1.0) < 0.1)
        #expect(abs(Double(decoded.frameCount) - 44100) < 100)
    }

    @Test("Decode non-existent file throws fileNotFound")
    func testDecodeNonExistentFile() {
        let decoder = AudioDecoder()
        let url = URL(fileURLWithPath: "/tmp/nonexistent.mp3")

        #expect(throws: AudioError.self) {
            try decoder.decode(fileURL: url)
        }
    }

    @Test("Decode invalid file throws decodeFailed")
    func testDecodeInvalidFile() throws {
        // Create empty file
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("invalid.mp3")
        try Data().write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let decoder = AudioDecoder()

        #expect(throws: AudioError.self) {
            try decoder.decode(fileURL: tempURL)
        }
    }

    // Helper to find fixture files
    private func resolveFixturePath(_ name: String) throws -> String {
        var projectRoot = URL(fileURLWithPath: #file)
        while projectRoot.path != "/" {
            projectRoot = projectRoot.deletingLastPathComponent()
            let packagePath = projectRoot.appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: packagePath.path) {
                break
            }
        }

        let fixturePath = projectRoot
            .appendingPathComponent("Resources/TestAudio")
            .appendingPathComponent(name)
            .path

        guard FileManager.default.fileExists(atPath: fixturePath) else {
            throw SkipTest("Fixture not found: \(fixturePath)")
        }

        return fixturePath
    }
}
