import Foundation

/// Errors that occur during audio file I/O operations
public enum AudioError: Error, LocalizedError {
    case fileNotFound(path: String)
    case unsupportedFormat(format: String, reason: String)
    case decodeFailed(underlyingError: Error)
    case encodeFailed(stem: StemType, reason: String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Audio file not found: \(path)"
        case .unsupportedFormat(let format, let reason):
            return "Unsupported audio format '\(format)': \(reason)"
        case .decodeFailed(let error):
            return "Failed to decode audio: \(error.localizedDescription)"
        case .encodeFailed(let stem, let reason):
            return "Failed to encode stem '\(stem.rawValue)': \(reason)"
        }
    }
}

/// Errors that occur during audio processing
public enum ProcessingError: Error, LocalizedError {
    case invalidSampleRate(actual: Double, required: Double)
    case invalidChannelCount(actual: Int, required: Int)
    case inferenceFailed(chunk: Int, reason: String)
    case outOfMemory

    public var errorDescription: String? {
        switch self {
        case .invalidSampleRate(let actual, let required):
            return "Invalid sample rate \(actual) Hz (required: \(required) Hz)"
        case .invalidChannelCount(let actual, let required):
            return "Invalid channel count \(actual) (required: \(required))"
        case .inferenceFailed(let chunk, let reason):
            return "Inference failed on chunk \(chunk): \(reason)"
        case .outOfMemory:
            return "Out of memory - audio file too large"
        }
    }
}
