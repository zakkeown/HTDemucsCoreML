import CoreML
import Foundation

/// Stem types for audio separation
public enum StemType: String, CaseIterable, Sendable {
    case drums
    case bass
    case vocals
    case other
    case piano
    case guitar
}

/// Complete audio source separation pipeline
public class SeparationPipeline: @unchecked Sendable {
    private let fft: AudioFFT
    private let inference: InferenceEngine
    private let chunker: ChunkProcessor

    /// Initialize pipeline with CoreML model
    /// - Parameter model: Loaded MLModel instance
    /// - Throws: Error if initialization fails
    public init(model: MLModel) throws {
        self.fft = try AudioFFT()
        self.inference = InferenceEngine(model: model)
        self.chunker = ChunkProcessor()
    }

    /// Initialize pipeline with model path
    /// - Parameter modelPath: Path to .mlpackage file
    /// - Throws: Error if model loading fails
    public init(modelPath: String) throws {
        self.fft = try AudioFFT()

        let loader = try ModelLoader(modelPath: modelPath)
        let model = try loader.load()
        self.inference = InferenceEngine(model: model)

        self.chunker = ChunkProcessor()
    }

    /// Separate stereo audio into 6 stems
    /// - Parameter stereoAudio: [leftChannel, rightChannel]
    /// - Returns: Dictionary of [StemType: [left, right]]
    /// - Throws: PipelineError if separation fails
    public func separate(stereoAudio: [[Float]]) throws -> [StemType: [[Float]]] {
        guard stereoAudio.count == 2 else {
            throw PipelineError.invalidChannelCount(stereoAudio.count)
        }

        guard stereoAudio[0].count == stereoAudio[1].count else {
            throw PipelineError.channelLengthMismatch
        }

        let audioLength = stereoAudio[0].count
        guard audioLength > 0 else {
            throw PipelineError.emptyAudio
        }

        // Note: This is a simplified skeleton. Full implementation would:
        // 1. Process both channels together in stereo chunks
        // 2. Apply chunking with ChunkProcessor (10s chunks, 1s overlap)
        // 3. For each chunk: STFT → inference → mask application → iSTFT
        // 4. Blend chunks with proper overlap-add
        // 5. Return [StemType: [[Float]]] with 6 stems, each stereo

        // For now, throw not implemented to signal this needs Phase 1 model
        throw PipelineError.notImplemented("Requires actual CoreML model for full pipeline")
    }
}

// MARK: - Error Types

public enum PipelineError: Error, LocalizedError {
    case invalidChannelCount(Int)
    case channelLengthMismatch
    case emptyAudio
    case invalidSampleRate(got: Double, expected: Double)
    case notImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .invalidChannelCount(let count):
            return "Expected stereo (2 channels), got \(count)"
        case .channelLengthMismatch:
            return "Left and right channels have different lengths"
        case .emptyAudio:
            return "Audio cannot be empty"
        case .invalidSampleRate(let got, let expected):
            return "Invalid sample rate: got \(got) Hz, expected \(expected) Hz"
        case .notImplemented(let msg):
            return "Not implemented: \(msg)"
        }
    }
}
