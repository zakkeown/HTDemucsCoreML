import Foundation
import CoreML

/// Orchestrates the complete audio separation workflow
/// Coordinates decoding, separation, and encoding with progress reporting
public class SeparationCoordinator: @unchecked Sendable {
    private let pipeline: SeparationPipeline
    private let decoder: AudioDecoder
    private let encoder: AudioEncoder

    /// Initialize with model name from Resources/Models
    /// - Parameter modelName: Name of the CoreML model (default: "htdemucs_6s")
    /// - Throws: ModelError if model not found or loading fails
    public init(modelName: String = "htdemucs_6s") throws {
        let loader = try ModelLoader(modelName: modelName)
        let model = try loader.load()
        self.pipeline = try SeparationPipeline(model: model)
        self.decoder = AudioDecoder()
        self.encoder = AudioEncoder()
    }

    /// Separate audio into stems with progress reporting
    /// - Parameters:
    ///   - input: URL to input audio file
    ///   - outputDir: Directory for output stem files
    ///   - format: Output audio format (default: .wav)
    /// - Returns: AsyncStream of ProgressEvent updates
    public func separate(
        input: URL,
        outputDir: URL,
        format: AudioFormat = .wav
    ) -> AsyncStream<ProgressEvent> {
        let pipeline = self.pipeline
        let decoder = self.decoder
        let encoder = self.encoder

        return AsyncStream { continuation in
            Task {
                do {
                    // Step 1: Decode audio
                    continuation.yield(.decoding(progress: 0.0))

                    let decoded = try decoder.decode(fileURL: input)

                    continuation.yield(.decoding(progress: 1.0))

                    // Step 2: Validate sample rate
                    // htdemucs expects 44.1kHz
                    let expectedSampleRate = 44100.0
                    guard abs(decoded.sampleRate - expectedSampleRate) < 1.0 else {
                        throw PipelineError.invalidSampleRate(
                            got: decoded.sampleRate,
                            expected: expectedSampleRate
                        )
                    }

                    // Step 3: Run separation pipeline
                    // Calculate expected number of chunks (based on ChunkProcessor config)
                    let chunkDuration: Float = 10.0  // seconds
                    let overlapDuration: Float = 1.0  // seconds per side
                    let sampleRate = Int(decoded.sampleRate)
                    let chunkSamples = Int(chunkDuration * Float(sampleRate))
                    let overlapSamples = Int(overlapDuration * Float(sampleRate))
                    let hopSamples = chunkSamples - 2 * overlapSamples
                    let audioLength = decoded.leftChannel.count

                    let numChunks: Int
                    if audioLength <= chunkSamples {
                        numChunks = 1
                    } else {
                        numChunks = (audioLength - overlapSamples * 2 + hopSamples - 1) / hopSamples
                    }

                    continuation.yield(.processing(chunk: 0, total: numChunks))

                    let stems = try pipeline.separate(
                        stereoAudio: decoded.stereoArray,
                        progressCallback: { chunkIdx, total in
                            continuation.yield(.processing(chunk: chunkIdx, total: total))
                        }
                    )

                    continuation.yield(.processing(chunk: numChunks, total: numChunks))

                    // Step 4: Encode stems
                    var outputPaths: [StemType: URL] = [:]

                    // Create output directory if needed
                    try FileManager.default.createDirectory(
                        at: outputDir,
                        withIntermediateDirectories: true,
                        attributes: nil
                    )

                    for (stemType, stemAudio) in stems {
                        continuation.yield(.encoding(stem: stemType, progress: 0.0))

                        let outputURL = outputDir
                            .appendingPathComponent(stemType.rawValue)
                            .appendingPathExtension(format.fileExtension)

                        // Extract left and right channels
                        guard stemAudio.count == 2 else {
                            throw PipelineError.invalidChannelCount(stemAudio.count)
                        }

                        try encoder.encode(
                            leftChannel: stemAudio[0],
                            rightChannel: stemAudio[1],
                            sampleRate: Int(decoded.sampleRate),
                            format: format,
                            destination: outputURL
                        )

                        outputPaths[stemType] = outputURL
                        continuation.yield(.encoding(stem: stemType, progress: 1.0))
                    }

                    // Step 5: Complete
                    continuation.yield(.complete(outputPaths: outputPaths))
                    continuation.finish()

                } catch {
                    continuation.yield(.failed(error: error))
                    continuation.finish()
                }
            }
        }
    }
}
