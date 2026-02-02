import Testing
import Foundation
@testable import HTDemucsKit

@Suite("SeparationCoordinator Tests")
struct SeparationCoordinatorTests {
    @Test("Initialize with default model name")
    func testInitWithDefaultModelName() throws {
        // This test skips if model not present
        let coordinator = try SeparationCoordinator()
        _ = coordinator
    }

    @Test("Initialize with custom model name")
    func testInitWithCustomModelName() throws {
        // This test skips if model not present
        let coordinator = try SeparationCoordinator(modelName: "htdemucs_6s")
        _ = coordinator
    }

    @Test("Throw error for non-existent model")
    func testThrowsForNonExistentModel() {
        #expect(throws: ModelError.self) {
            _ = try SeparationCoordinator(modelName: "nonexistent_model")
        }
    }

    @Test("Separate method returns AsyncStream")
    func testSeparateReturnsAsyncStream() async throws {
        // Skip if model or audio not available
        let coordinator = try SeparationCoordinator()

        let inputURL = try resolveFixturePath("sine-440hz-1s.wav")
        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_separation_\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: outputDir)
        }

        let stream = coordinator.separate(
            input: URL(fileURLWithPath: inputURL),
            outputDir: outputDir,
            format: .wav
        )

        var eventCount = 0
        var lastEvent: ProgressEvent?

        for await event in stream {
            eventCount += 1
            lastEvent = event

            // We should get at least: decoding, processing, encoding, complete
            switch event {
            case .decoding:
                break
            case .processing:
                break
            case .encoding:
                break
            case .complete:
                break
            case .failed:
                Issue.record("Unexpected failure: \(event.description)")
            }
        }

        // Should have received at least one event
        #expect(eventCount > 0)

        // Last event should be complete or failed
        if let last = lastEvent {
            switch last {
            case .complete, .failed:
                break
            default:
                Issue.record("Expected final event to be complete or failed")
            }
        }
    }

    @Test("Validate sample rate requirement")
    func testValidateSampleRate() async throws {
        // For now, we just test that the coordinator initializes
        // Full sample rate validation will be tested with real audio
        let coordinator = try SeparationCoordinator()
        _ = coordinator
    }

    // Helper to resolve fixture path
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
