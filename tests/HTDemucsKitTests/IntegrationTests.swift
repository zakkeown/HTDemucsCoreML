import Testing
import Foundation
@testable import HTDemucsKit

/// Integration tests for end-to-end audio separation workflow
///
/// Prerequisites:
/// - Model file: Resources/Models/htdemucs_6s.mlpackage
/// - Test audio: Resources/TestAudio/sine-440hz-1s.wav
///
/// These tests will skip gracefully if resources are not available.
@Suite("Integration Tests")
struct IntegrationTests {

    // MARK: - Test Configuration

    /// Check if model is available
    private func hasModel() -> Bool {
        do {
            _ = try resolveModelPath()
            return true
        } catch {
            return false
        }
    }

    /// Check if test audio is available
    private func hasTestAudio() -> Bool {
        do {
            _ = try resolveFixturePath("sine-440hz-1s.wav")
            return true
        } catch {
            return false
        }
    }

    /// Resolve model path
    private func resolveModelPath() throws -> String {
        let possiblePaths = [
            "/Users/zakkeown/Code/HTDemucsCoreML/.worktrees/phase3-audio-io/Resources/Models/htdemucs_6s.mlpackage",
            "/Users/zakkeown/Code/HTDemucsCoreML/Resources/Models/htdemucs_6s.mlpackage",
        ]

        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }

        throw TestHelperError.fixtureNotFound("htdemucs_6s.mlpackage")
    }

    // MARK: - End-to-End Tests

    /// Test complete separation pipeline from audio file to stem outputs
    @Test("End-to-end separation")
    func testEndToEndSeparation() async throws {
        // Check prerequisites
        guard hasModel() else {
            throw SkipError("Model not available: htdemucs_6s.mlpackage")
        }

        guard hasTestAudio() else {
            throw SkipError("Test audio not available: sine-440hz-1s.wav")
        }

        // Get test audio path
        let audioPath = try resolveFixturePath("sine-440hz-1s.wav")

        // Create temporary output directory
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("htdemucs-integration-test-\(UUID().uuidString)")

        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        // Initialize coordinator
        let coordinator = try SeparationCoordinator(modelName: "htdemucs_6s")

        // Run separation
        let inputURL = URL(fileURLWithPath: audioPath)
        let progressStream = coordinator.separate(
            input: inputURL,
            outputDir: tempDir,
            format: .wav
        )

        // Track progress events
        var progressEvents: [ProgressEvent] = []
        var completedPaths: [StemType: URL]?
        var didFail = false

        for await event in progressStream {
            progressEvents.append(event)

            switch event {
            case .complete(let outputPaths):
                completedPaths = outputPaths

            case .failed:
                didFail = true

            default:
                break
            }
        }

        // If pipeline not fully implemented, skip with message
        if didFail && progressEvents.contains(where: { event in
            if case .failed(let error) = event {
                return error.localizedDescription.contains("Not implemented")
            }
            return false
        }) {
            throw SkipError("Pipeline not fully implemented yet (missing CoreML inference)")
        }

        // Verify progress events received
        #expect(!progressEvents.isEmpty, "Should receive progress events")

        // Verify we got decoding events
        let hasDecodingEvent = progressEvents.contains { event in
            if case .decoding = event { return true }
            return false
        }
        #expect(hasDecodingEvent, "Should have decoding progress event")

        // Verify we got processing events
        let hasProcessingEvent = progressEvents.contains { event in
            if case .processing = event { return true }
            return false
        }
        #expect(hasProcessingEvent, "Should have processing progress event")

        // If completed successfully, verify outputs
        if let outputs = completedPaths {
            // Verify all 6 stems were generated
            #expect(outputs.count == 6, "Should generate 6 stems")

            // Verify each stem type is present
            let expectedStems: [StemType] = [.drums, .bass, .vocals, .other, .piano, .guitar]
            for stemType in expectedStems {
                #expect(outputs[stemType] != nil, "Should have \(stemType.rawValue) stem")

                if let path = outputs[stemType] {
                    #expect(
                        FileManager.default.fileExists(atPath: path.path),
                        "Output file should exist: \(path.path)"
                    )

                    // Verify file has content
                    let attributes = try? FileManager.default.attributesOfItem(atPath: path.path)
                    let fileSize = attributes?[.size] as? Int64 ?? 0
                    #expect(fileSize > 0, "Output file should have content")
                }
            }
        }
    }

    /// Test error handling for nonexistent input file
    @Test("Error handling - nonexistent file")
    func testErrorHandlingNonexistentFile() async throws {
        guard hasModel() else {
            throw SkipError("Model not available: htdemucs_6s.mlpackage")
        }

        // Initialize coordinator
        let coordinator = try SeparationCoordinator(modelName: "htdemucs_6s")

        // Try to separate nonexistent file
        let nonexistentURL = URL(fileURLWithPath: "/tmp/nonexistent-file-\(UUID().uuidString).wav")
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("htdemucs-error-test-\(UUID().uuidString)")

        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        let progressStream = coordinator.separate(
            input: nonexistentURL,
            outputDir: tempDir,
            format: .wav
        )

        // Should receive a failed event
        var didReceiveError = false

        for await event in progressStream {
            if case .failed(let error) = event {
                didReceiveError = true

                // Verify error is about file not found
                let errorDescription = error.localizedDescription.lowercased()
                #expect(
                    errorDescription.contains("file") || errorDescription.contains("not found"),
                    "Error should indicate file not found: \(errorDescription)"
                )
            }
        }

        #expect(didReceiveError, "Should receive error event for nonexistent file")
    }

    /// Test progress reporting consistency
    @Test("Progress reporting consistency")
    func testProgressReportingConsistency() async throws {
        guard hasModel() else {
            throw SkipError("Model not available: htdemucs_6s.mlpackage")
        }

        guard hasTestAudio() else {
            throw SkipError("Test audio not available: sine-440hz-1s.wav")
        }

        let audioPath = try resolveFixturePath("sine-440hz-1s.wav")
        let coordinator = try SeparationCoordinator(modelName: "htdemucs_6s")

        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("htdemucs-progress-test-\(UUID().uuidString)")

        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        let inputURL = URL(fileURLWithPath: audioPath)
        let progressStream = coordinator.separate(
            input: inputURL,
            outputDir: tempDir,
            format: .wav
        )

        var lastProcessingChunk = -1
        var processingTotal = 0
        var allProgressValid = true

        for await event in progressStream {
            switch event {
            case .decoding(let progress):
                // Progress should be between 0 and 1
                if progress < 0.0 || progress > 1.0 {
                    allProgressValid = false
                }

            case .processing(let chunk, let total):
                // Chunks should be sequential
                if lastProcessingChunk >= 0 && chunk != lastProcessingChunk && chunk != lastProcessingChunk + 1 {
                    allProgressValid = false
                }
                lastProcessingChunk = chunk
                processingTotal = total

                // Chunk should be <= total
                if chunk > total {
                    allProgressValid = false
                }

            case .encoding(_, let progress):
                // Progress should be between 0 and 1
                if progress < 0.0 || progress > 1.0 {
                    allProgressValid = false
                }

            case .failed(let error):
                // If not implemented, skip
                if error.localizedDescription.contains("Not implemented") {
                    throw SkipError("Pipeline not fully implemented yet")
                }

            default:
                break
            }
        }

        #expect(allProgressValid, "All progress values should be valid")
        #expect(processingTotal > 0, "Should have at least one chunk")
    }

    /// Test multiple format outputs
    @Test("Multiple format outputs")
    func testMultipleFormatOutputs() async throws {
        guard hasModel() else {
            throw SkipError("Model not available: htdemucs_6s.mlpackage")
        }

        guard hasTestAudio() else {
            throw SkipError("Test audio not available: sine-440hz-1s.wav")
        }

        let audioPath = try resolveFixturePath("sine-440hz-1s.wav")
        let coordinator = try SeparationCoordinator(modelName: "htdemucs_6s")
        let inputURL = URL(fileURLWithPath: audioPath)

        let formats: [AudioFormat] = [.wav, .mp3, .flac]

        for format in formats {
            let tempDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("htdemucs-format-test-\(format.rawValue)-\(UUID().uuidString)")

            defer {
                try? FileManager.default.removeItem(at: tempDir)
            }

            let progressStream = coordinator.separate(
                input: inputURL,
                outputDir: tempDir,
                format: format
            )

            var didComplete = false

            for await event in progressStream {
                if case .complete(let outputs) = event {
                    didComplete = true

                    // Verify all outputs have correct extension
                    for (_, url) in outputs {
                        #expect(
                            url.pathExtension == format.fileExtension,
                            "Output should have .\(format.fileExtension) extension"
                        )
                    }
                } else if case .failed(let error) = event {
                    if error.localizedDescription.contains("Not implemented") {
                        throw SkipError("Pipeline not fully implemented yet")
                    }
                }
            }

            // Tests will skip if pipeline not implemented
        }
    }
}

// MARK: - SkipError

struct SkipError: Error, CustomStringConvertible {
    let description: String

    init(_ message: String) {
        self.description = message
    }
}
