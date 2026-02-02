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
    /// - Parameters:
    ///   - stereoAudio: [leftChannel, rightChannel]
    ///   - progressCallback: Optional callback for chunk progress (chunkIndex, totalChunks)
    /// - Returns: Dictionary of [StemType: [left, right]]
    /// - Throws: PipelineError if separation fails
    public func separate(
        stereoAudio: [[Float]],
        progressCallback: ((Int, Int) -> Void)? = nil
    ) throws -> [StemType: [[Float]]] {
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

        // Initialize output storage for 6 stems, each stereo
        var separatedStems: [Int: [[Float]]] = [:]
        for stemIdx in 0..<6 {
            separatedStems[stemIdx] = [
                [Float](repeating: 0, count: audioLength),  // left channel
                [Float](repeating: 0, count: audioLength)   // right channel
            ]
        }

        // Process stereo audio through ChunkProcessor
        // We'll use the left channel for chunking and synchronize both channels
        let leftChannel = stereoAudio[0]
        let rightChannel = stereoAudio[1]

        // Process with chunking - we use left channel for chunk boundaries
        let processedChunks = try processStereoInChunks(
            leftChannel: leftChannel,
            rightChannel: rightChannel,
            progressCallback: progressCallback
        )

        // processedChunks contains [6 stems][2 channels][samples]
        // Assign to output structure
        for stemIdx in 0..<6 {
            separatedStems[stemIdx] = processedChunks[stemIdx]
        }

        // Convert to dictionary with StemType keys
        let stemTypes = StemType.allCases
        var result: [StemType: [[Float]]] = [:]
        for (stemIdx, stemType) in stemTypes.enumerated() {
            result[stemType] = separatedStems[stemIdx]
        }

        return result
    }

    // MARK: - Private Helpers

    /// Process stereo audio in chunks with overlap-add
    /// - Parameters:
    ///   - leftChannel: Left channel audio samples
    ///   - rightChannel: Right channel audio samples
    ///   - progressCallback: Optional progress callback
    /// - Returns: Array of [6 stems][2 channels][samples]
    private func processStereoInChunks(
        leftChannel: [Float],
        rightChannel: [Float],
        progressCallback: ((Int, Int) -> Void)? = nil
    ) throws -> [[[Float]]] {
        let audioLength = leftChannel.count

        // ChunkProcessor configuration
        let chunkDuration: Float = 10.0
        let overlapDuration: Float = 1.0
        let sampleRate = 44100
        let chunkSamples = Int(chunkDuration * Float(sampleRate))
        let overlapSamples = Int(overlapDuration * Float(sampleRate))
        let hopSamples = chunkSamples - 2 * overlapSamples

        // Calculate number of chunks
        let numChunks: Int
        if audioLength <= chunkSamples {
            numChunks = 1
        } else {
            numChunks = (audioLength - overlapSamples * 2 + hopSamples - 1) / hopSamples
        }

        // Initialize output accumulation arrays for 6 stems, each stereo
        var stemOutputs: [[[Float]]] = []
        var stemWeights: [[[Float]]] = []

        for _ in 0..<6 {
            stemOutputs.append([
                [Float](repeating: 0, count: audioLength),
                [Float](repeating: 0, count: audioLength)
            ])
            stemWeights.append([
                [Float](repeating: 0, count: audioLength),
                [Float](repeating: 0, count: audioLength)
            ])
        }

        // Process each chunk
        for chunkIdx in 0..<numChunks {
            progressCallback?(chunkIdx, numChunks)

            let start = chunkIdx * hopSamples
            let end = min(start + chunkSamples, audioLength)

            // Extract stereo chunk
            let leftChunk = Array(leftChannel[start..<end])
            let rightChunk = Array(rightChannel[start..<end])

            // Process stereo chunk
            let processedStems = try processStereoChunk(
                leftChannel: leftChunk,
                rightChannel: rightChunk
            )

            // Create blend window
            let window = createBlendWindow(
                chunkSize: leftChunk.count,
                overlapSize: overlapSamples,
                isFirst: chunkIdx == 0,
                isLast: end >= audioLength
            )

            // Accumulate with blending for each stem and channel
            for stemIdx in 0..<6 {
                for channelIdx in 0..<2 {
                    let stemChannel = processedStems[stemIdx][channelIdx]
                    for (i, value) in stemChannel.enumerated() {
                        let outputIdx = start + i
                        if outputIdx < audioLength {
                            stemOutputs[stemIdx][channelIdx][outputIdx] += value * window[i]
                            stemWeights[stemIdx][channelIdx][outputIdx] += window[i]
                        }
                    }
                }
            }
        }

        progressCallback?(numChunks, numChunks)

        // Normalize by accumulated weights
        for stemIdx in 0..<6 {
            for channelIdx in 0..<2 {
                for i in 0..<audioLength where stemWeights[stemIdx][channelIdx][i] > 0 {
                    stemOutputs[stemIdx][channelIdx][i] /= stemWeights[stemIdx][channelIdx][i]
                }
            }
        }

        return stemOutputs
    }

    /// Process a single stereo chunk through STFT → inference → mask → iSTFT
    /// - Parameters:
    ///   - leftChannel: Left channel chunk samples
    ///   - rightChannel: Right channel chunk samples
    /// - Returns: Array of [6 stems][2 channels][samples]
    private func processStereoChunk(
        leftChannel: [Float],
        rightChannel: [Float]
    ) throws -> [[[Float]]] {
        // 1. Run STFT on both channels
        let (leftReal, leftImag) = try fft.stft(leftChannel)
        let (rightReal, rightImag) = try fft.stft(rightChannel)

        // 2. Prepare stereo spectrogram for inference
        // STFT returns [numFrames][numBins=2049]
        // InferenceEngine expects [2 channels][2049 freq bins][timeFrames]
        // Need to transpose from [frames][bins] to [bins][frames]
        let leftRealTransposed = transposeSpectrogram(leftReal)
        let leftImagTransposed = transposeSpectrogram(leftImag)
        let rightRealTransposed = transposeSpectrogram(rightReal)
        let rightImagTransposed = transposeSpectrogram(rightImag)

        let stereoReal = [leftRealTransposed, rightRealTransposed]
        let stereoImag = [leftImagTransposed, rightImagTransposed]

        // 3. Run inference to get masks [6][2][2049][timeFrames]
        let masks = try inference.predict(real: stereoReal, imag: stereoImag)

        // 4. Apply masks to spectrograms and run iSTFT for each stem and channel
        var stemOutputs: [[[Float]]] = []

        for stemIdx in 0..<6 {
            let stemMask = masks[stemIdx]  // [2][2049][timeFrames]

            var channelOutputs: [[Float]] = []

            // Process left channel (0) and right channel (1)
            for channelIdx in 0..<2 {
                let channelMask = stemMask[channelIdx]  // [2049 bins][timeFrames]
                let realSpec = channelIdx == 0 ? leftReal : rightReal  // [numFrames][2049 bins]
                let imagSpec = channelIdx == 0 ? leftImag : rightImag  // [numFrames][2049 bins]

                // Apply mask to spectrogram (element-wise multiply)
                // realSpec and imagSpec are [numFrames][numBins=2049]
                // channelMask is [numBins=2049][numFrames]
                // Need to align indices: realSpec[frameIdx][binIdx] * channelMask[binIdx][frameIdx]
                var maskedReal: [[Float]] = []
                var maskedImag: [[Float]] = []

                for frameIdx in 0..<realSpec.count {
                    let realFrame = realSpec[frameIdx]
                    let imagFrame = imagSpec[frameIdx]

                    var newRealFrame = [Float](repeating: 0, count: realFrame.count)
                    var newImagFrame = [Float](repeating: 0, count: imagFrame.count)

                    for binIdx in 0..<realFrame.count {
                        // channelMask is [bins][frames], so mask[binIdx][frameIdx]
                        let maskValue = channelMask[binIdx][frameIdx]
                        newRealFrame[binIdx] = realFrame[binIdx] * maskValue
                        newImagFrame[binIdx] = imagFrame[binIdx] * maskValue
                    }

                    maskedReal.append(newRealFrame)
                    maskedImag.append(newImagFrame)
                }

                // 5. Run iSTFT to get time-domain audio for this stem channel
                let stemAudio = try fft.istft(real: maskedReal, imag: maskedImag)
                channelOutputs.append(stemAudio)
            }

            stemOutputs.append(channelOutputs)
        }

        return stemOutputs
    }

    /// Create blend window with linear crossfade
    private func createBlendWindow(
        chunkSize: Int,
        overlapSize: Int,
        isFirst: Bool,
        isLast: Bool
    ) -> [Float] {
        var window = [Float](repeating: 1.0, count: chunkSize)

        // Fade in at start (unless first chunk)
        if !isFirst && chunkSize >= overlapSize {
            for i in 0..<min(overlapSize, chunkSize) {
                window[i] = Float(i) / Float(overlapSize)
            }
        }

        // Fade out at end (unless last chunk)
        if !isLast && chunkSize >= overlapSize {
            for i in 0..<min(overlapSize, chunkSize) {
                let idx = chunkSize - overlapSize + i
                if idx >= 0 && idx < chunkSize {
                    window[idx] = 1.0 - Float(i) / Float(overlapSize)
                }
            }
        }

        return window
    }

    /// Transpose spectrogram from [frames][bins] to [bins][frames]
    private func transposeSpectrogram(_ spec: [[Float]]) -> [[Float]] {
        guard let firstFrame = spec.first else {
            return []
        }

        let numFrames = spec.count
        let numBins = firstFrame.count

        var transposed = [[Float]](repeating: [Float](repeating: 0, count: numFrames), count: numBins)

        for (frameIdx, frame) in spec.enumerated() {
            for (binIdx, value) in frame.enumerated() {
                transposed[binIdx][frameIdx] = value
            }
        }

        return transposed
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
