import Testing
import Foundation
@testable import HTDemucsKit

@Suite("ProgressEvent Tests")
struct ProgressEventTests {
    @Test("Decoding progress event")
    func testDecodingProgress() {
        let event = ProgressEvent.decoding(progress: 0.5)

        switch event {
        case .decoding(let progress):
            #expect(abs(progress - 0.5) < 0.001)
        default:
            Issue.record("Expected decoding event")
        }

        #expect(event.description.contains("Decoding"))
        #expect(event.description.contains("50"))
    }

    @Test("Processing progress event")
    func testProcessingProgress() {
        let event = ProgressEvent.processing(chunk: 3, total: 10)

        switch event {
        case .processing(let chunk, let total):
            #expect(chunk == 3)
            #expect(total == 10)
        default:
            Issue.record("Expected processing event")
        }

        #expect(event.description.contains("Processing"))
        // Description shows 1-indexed chunk number (chunk + 1)
        #expect(event.description.contains("4"))
        #expect(event.description.contains("10"))
    }

    @Test("Encoding progress event")
    func testEncodingProgress() {
        let event = ProgressEvent.encoding(stem: .drums, progress: 0.75)

        switch event {
        case .encoding(let stem, let progress):
            #expect(stem == .drums)
            #expect(abs(progress - 0.75) < 0.001)
        default:
            Issue.record("Expected encoding event")
        }

        #expect(event.description.contains("Encoding"))
        #expect(event.description.contains("drums"))
        #expect(event.description.contains("75"))
    }

    @Test("Complete event")
    func testCompleteEvent() {
        let outputPaths: [StemType: URL] = [
            .drums: URL(fileURLWithPath: "/tmp/drums.wav"),
            .bass: URL(fileURLWithPath: "/tmp/bass.wav")
        ]
        let event = ProgressEvent.complete(outputPaths: outputPaths)

        switch event {
        case .complete(let paths):
            #expect(paths.count == 2)
            #expect(paths[.drums]?.path == "/tmp/drums.wav")
            #expect(paths[.bass]?.path == "/tmp/bass.wav")
        default:
            Issue.record("Expected complete event")
        }

        #expect(event.description.contains("Complete"))
        #expect(event.description.contains("2"))
    }

    @Test("Failed event")
    func testFailedEvent() {
        enum TestError: Error {
            case testError
        }

        let event = ProgressEvent.failed(error: TestError.testError)

        switch event {
        case .failed(let error):
            #expect(error is TestError)
        default:
            Issue.record("Expected failed event")
        }

        #expect(event.description.contains("Failed"))
    }

    @Test("Event descriptions are unique")
    func testEventDescriptionsAreUnique() {
        let events: [ProgressEvent] = [
            .decoding(progress: 0.5),
            .processing(chunk: 1, total: 5),
            .encoding(stem: .vocals, progress: 0.3),
            .complete(outputPaths: [:]),
            .failed(error: NSError(domain: "test", code: 1))
        ]

        let descriptions = events.map { $0.description }
        let uniqueDescriptions = Set(descriptions)

        // Each event should have a distinct description pattern
        #expect(descriptions.count == uniqueDescriptions.count)
    }
}
