import XCTest
@testable import HTDemucsKit

final class ChunkProcessorTests: XCTestCase {
    var processor: ChunkProcessor!

    override func setUp() {
        super.setUp()
        processor = ChunkProcessor()
    }

    override func tearDown() {
        processor = nil
        super.tearDown()
    }

    // MARK: - Configuration Tests

    func testConfiguration() {
        XCTAssertEqual(processor.chunkDuration, 10.0)
        XCTAssertEqual(processor.overlapDuration, 1.0)
        XCTAssertEqual(processor.sampleRate, 44100)
    }

    // MARK: - Short Audio Tests

    func testShortAudio_ProcessedDirectly() throws {
        // Audio shorter than chunk size should be processed directly without chunking
        let shortAudio = [Float](repeating: 1.0, count: 1000)

        let result = try processor.processInChunks(audio: shortAudio) { chunk in
            // Identity processor - multiply by 2 to verify this processor ran
            return chunk.map { $0 * 2.0 }
        }

        XCTAssertEqual(result.count, shortAudio.count)
        // All values should be 2.0 (1.0 * 2.0)
        for value in result {
            XCTAssertEqual(value, 2.0, accuracy: 0.001)
        }
    }

    func testEmptyAudio() throws {
        let result = try processor.processInChunks(audio: []) { chunk in
            return chunk
        }

        XCTAssertEqual(result.count, 0)
    }

    // MARK: - Exact Chunk Size Tests

    func testExactChunkSize() throws {
        // Audio exactly the chunk size
        let chunkSize = Int(processor.chunkDuration * Float(processor.sampleRate))
        let audio = [Float](repeating: 1.0, count: chunkSize)

        let result = try processor.processInChunks(audio: audio) { chunk in
            return chunk.map { $0 * 2.0 }
        }

        XCTAssertEqual(result.count, audio.count)
        for value in result {
            XCTAssertEqual(value, 2.0, accuracy: 0.001)
        }
    }

    // MARK: - Long Audio Tests

    func testLongAudio_MultipleChunks() throws {
        // Create audio longer than chunk size to trigger chunking
        let duration: Float = 25.0 // 25 seconds
        let totalSamples = Int(duration * Float(processor.sampleRate))
        let audio = [Float](repeating: 1.0, count: totalSamples)

        let result = try processor.processInChunks(audio: audio) { chunk in
            // Scale by 2
            return chunk.map { $0 * 2.0 }
        }

        XCTAssertEqual(result.count, audio.count)

        // All samples should be approximately 2.0 after proper blending
        for (idx, value) in result.enumerated() {
            XCTAssertEqual(value, 2.0, accuracy: 0.01,
                          "Sample \(idx) should be ~2.0 but was \(value)")
        }
    }

    // MARK: - Overlap-Add Correctness Tests

    func testOverlapAddWeights() throws {
        // Test that weights sum to ~1.0 everywhere
        let duration: Float = 25.0
        let totalSamples = Int(duration * Float(processor.sampleRate))
        let audio = [Float](repeating: 1.0, count: totalSamples)

        // Use identity processor
        let result = try processor.processInChunks(audio: audio) { chunk in
            return chunk
        }

        XCTAssertEqual(result.count, audio.count)

        // For constant input with identity processor, output should equal input
        for (idx, value) in result.enumerated() {
            XCTAssertEqual(value, 1.0, accuracy: 0.001,
                          "Sample \(idx) should be ~1.0 (weights normalized)")
        }
    }

    func testIdentityProcessor_ReconstructsInput() throws {
        // Identity processor with overlap-add should reconstruct input
        let duration: Float = 30.0
        let totalSamples = Int(duration * Float(processor.sampleRate))

        // Create varied input signal (sine wave)
        var audio = [Float](repeating: 0, count: totalSamples)
        for i in 0..<totalSamples {
            let t = Float(i) / Float(processor.sampleRate)
            audio[i] = sin(2.0 * .pi * 440.0 * t) // 440 Hz sine wave
        }

        let result = try processor.processInChunks(audio: audio) { chunk in
            return chunk // Identity
        }

        XCTAssertEqual(result.count, audio.count)

        // Should reconstruct original signal
        for (idx, value) in result.enumerated() {
            XCTAssertEqual(value, audio[idx], accuracy: 0.001,
                          "Sample \(idx) reconstruction failed")
        }
    }

    // MARK: - Edge Cases Tests

    func testFirstChunk_NoFadeIn() throws {
        // First chunk should have no fade-in (full weight at start)
        let duration: Float = 25.0
        let totalSamples = Int(duration * Float(processor.sampleRate))
        let audio = [Float](repeating: 1.0, count: totalSamples)

        var firstChunkProcessed = false
        let result = try processor.processInChunks(audio: audio) { chunk in
            if !firstChunkProcessed {
                firstChunkProcessed = true
                // First chunk should start at full amplitude
                XCTAssertEqual(chunk[0], 1.0, accuracy: 0.001)
            }
            return chunk.map { $0 * 2.0 }
        }

        // First sample of output should be 2.0 (no fade-in)
        XCTAssertEqual(result[0], 2.0, accuracy: 0.001)
    }

    func testLastChunk_NoFadeOut() throws {
        // Last chunk should have no fade-out (full weight at end)
        let duration: Float = 25.0
        let totalSamples = Int(duration * Float(processor.sampleRate))
        let audio = [Float](repeating: 1.0, count: totalSamples)

        let result = try processor.processInChunks(audio: audio) { chunk in
            return chunk.map { $0 * 2.0 }
        }

        // Last sample of output should be 2.0 (no fade-out)
        XCTAssertEqual(result[totalSamples - 1], 2.0, accuracy: 0.001)
    }

    func testTwoChunks_OverlapBlending() throws {
        // Test blending between exactly two chunks
        let hopSamples = Int(processor.chunkDuration * Float(processor.sampleRate)) - 2 * Int(processor.overlapDuration * Float(processor.sampleRate))
        let chunkSamples = Int(processor.chunkDuration * Float(processor.sampleRate))
        let totalSamples = chunkSamples + hopSamples + 100 // Ensures two chunks with overlap

        let audio = [Float](repeating: 1.0, count: totalSamples)

        let result = try processor.processInChunks(audio: audio) { chunk in
            return chunk
        }

        XCTAssertEqual(result.count, audio.count)

        // All samples should be ~1.0 after blending
        for (idx, value) in result.enumerated() {
            XCTAssertEqual(value, 1.0, accuracy: 0.001,
                          "Sample \(idx) blending failed")
        }
    }

    // MARK: - Error Handling Tests

    func testProcessorThrows() {
        let audio = [Float](repeating: 1.0, count: 1000)

        enum TestError: Error {
            case processingFailed
        }

        XCTAssertThrowsError(try processor.processInChunks(audio: audio) { _ in
            throw TestError.processingFailed
        }) { error in
            XCTAssertTrue(error is TestError)
        }
    }

    // MARK: - Numerical Precision Tests

    func testNumericalStability() throws {
        // Test with very small values
        let duration: Float = 15.0
        let totalSamples = Int(duration * Float(processor.sampleRate))
        let audio = [Float](repeating: 0.0001, count: totalSamples)

        let result = try processor.processInChunks(audio: audio) { chunk in
            return chunk.map { $0 * 10.0 }
        }

        for (idx, value) in result.enumerated() {
            XCTAssertEqual(value, 0.001, accuracy: 0.0001,
                          "Sample \(idx) precision issue")
        }
    }

    func testLargeValues() throws {
        // Test with large values
        let duration: Float = 15.0
        let totalSamples = Int(duration * Float(processor.sampleRate))
        let audio = [Float](repeating: 1000.0, count: totalSamples)

        let result = try processor.processInChunks(audio: audio) { chunk in
            return chunk.map { $0 * 2.0 }
        }

        for (idx, value) in result.enumerated() {
            XCTAssertEqual(value, 2000.0, accuracy: 0.1,
                          "Sample \(idx) large value handling failed")
        }
    }
}
