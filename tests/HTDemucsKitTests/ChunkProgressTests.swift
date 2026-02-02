import Testing
import Foundation
@testable import HTDemucsKit

@Suite("Chunk Progress Tests")
struct ChunkProgressTests {
    @Test("ChunkProcessor reports progress for multiple chunks")
    func testChunkProcessorProgress() throws {
        let processor = ChunkProcessor()

        // Create audio longer than one chunk (10s @ 44.1kHz = 441000 samples)
        // Make it 30 seconds = 1,323,000 samples
        let audioLength = 30 * 44100
        let audio = [Float](repeating: 0.5, count: audioLength)

        var progressReports: [(chunk: Int, total: Int)] = []

        let result = try processor.processInChunks(
            audio: audio,
            processor: { chunk in
                // Simple passthrough processor
                return chunk
            },
            progressCallback: { chunk, total in
                progressReports.append((chunk, total))
            }
        )

        // Should have received progress updates
        #expect(progressReports.count > 0)

        // Should process multiple chunks for 30s audio
        if let last = progressReports.last {
            #expect(last.total > 1)
        }

        // Result should match input length
        #expect(result.count == audioLength)
    }

    @Test("ChunkProcessor reports progress for single chunk")
    func testChunkProcessorProgressSingleChunk() throws {
        let processor = ChunkProcessor()

        // Create audio shorter than one chunk (2s @ 44.1kHz = 88200 samples)
        let audioLength = 2 * 44100
        let audio = [Float](repeating: 0.5, count: audioLength)

        var progressReports: [(chunk: Int, total: Int)] = []

        let result = try processor.processInChunks(
            audio: audio,
            processor: { chunk in
                return chunk
            },
            progressCallback: { chunk, total in
                progressReports.append((chunk, total))
            }
        )

        // Should have received progress updates
        #expect(progressReports.count > 0)

        // Should indicate 1 total chunk
        if let last = progressReports.last {
            #expect(last.total == 1)
        }

        // Result should match input length
        #expect(result.count == audioLength)
    }

    @Test("ChunkProcessor works without progress callback")
    func testChunkProcessorWithoutCallback() {
        let processor = ChunkProcessor()

        let audioLength = 10 * 44100
        let audio = [Float](repeating: 0.5, count: audioLength)

        // Should work without callback
        let result = processor.processInChunks(
            audio: audio,
            processor: { chunk in
                return chunk
            }
        )

        #expect(result.count == audioLength)
    }

    @Test("SeparationCoordinator reports chunk progress")
    func testSeparationCoordinatorChunkProgress() async throws {
        // This test requires model and audio fixtures
        let coordinator = try SeparationCoordinator()

        let inputURL = try resolveFixturePath("sine-440hz-1s.wav")
        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_chunk_progress_\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: outputDir)
        }

        let stream = coordinator.separate(
            input: URL(fileURLWithPath: inputURL),
            outputDir: outputDir,
            format: .wav
        )

        var processingEvents: [(chunk: Int, total: Int)] = []

        for await event in stream {
            if case .processing(let chunk, let total) = event {
                processingEvents.append((chunk, total))
            }
        }

        // Should have received at least some processing events
        #expect(processingEvents.count > 0)

        // All events should have consistent total
        if let first = processingEvents.first {
            let expectedTotal = first.total
            for event in processingEvents {
                #expect(event.total == expectedTotal)
            }
        }
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
