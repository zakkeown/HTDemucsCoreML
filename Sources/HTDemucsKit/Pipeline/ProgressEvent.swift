import Foundation

/// Progress events for tracking separation pipeline execution
public enum ProgressEvent: Sendable {
    /// Audio decoding in progress
    /// - Parameter progress: Progress from 0.0 to 1.0
    case decoding(progress: Float)

    /// Processing audio chunks
    /// - Parameters:
    ///   - chunk: Current chunk number (0-indexed)
    ///   - total: Total number of chunks
    case processing(chunk: Int, total: Int)

    /// Encoding a stem to output file
    /// - Parameters:
    ///   - stem: The stem type being encoded
    ///   - progress: Encoding progress from 0.0 to 1.0
    case encoding(stem: StemType, progress: Float)

    /// Separation completed successfully
    /// - Parameter outputPaths: Dictionary mapping stem types to output file URLs
    case complete(outputPaths: [StemType: URL])

    /// Separation failed with error
    /// - Parameter error: The error that occurred
    case failed(error: Error)

    /// Human-readable description of the progress event
    public var description: String {
        switch self {
        case .decoding(let progress):
            let percent = Int(progress * 100)
            return "Decoding audio... \(percent)%"

        case .processing(let chunk, let total):
            return "Processing chunk \(chunk + 1) of \(total)..."

        case .encoding(let stem, let progress):
            let percent = Int(progress * 100)
            return "Encoding \(stem.rawValue)... \(percent)%"

        case .complete(let outputPaths):
            return "Complete! Generated \(outputPaths.count) stems"

        case .failed(let error):
            return "Failed: \(error.localizedDescription)"
        }
    }
}
