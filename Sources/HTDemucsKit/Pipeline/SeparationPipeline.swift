import CoreML
import Foundation

/// Stem types for audio separation
/// IMPORTANT: Order MUST match PyTorch htdemucs_6s model output order!
public enum StemType: String, CaseIterable, Sendable {
    case drums  // Index 0
    case bass   // Index 1
    case other  // Index 2 (NOT vocals!)
    case vocals // Index 3 (NOT other!)
    case guitar // Index 4
    case piano  // Index 5
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
        // CRITICAL: Must match model's native segment size (343980 samples = ~7.8s)
        // Using larger chunks would cause dimension mismatches with model output
        let sampleRate = 44100
        let chunkSamples = 343980  // Model's native segment size
        let overlapSamples = Int(1.0 * Float(sampleRate))  // 1 second overlap
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

    /// Process a single stereo chunk through STFT → hybrid inference → iSTFT + time
    /// Uses the full hybrid model with both frequency and time branches.
    /// - Parameters:
    ///   - leftChannel: Left channel chunk samples
    ///   - rightChannel: Right channel chunk samples
    /// - Returns: Array of [6 stems][2 channels][samples]
    private func processStereoChunk(
        leftChannel: [Float],
        rightChannel: [Float]
    ) throws -> [[[Float]]] {
        // Store original chunk length for output trimming
        let originalLength = leftChannel.count

        // CRITICAL: Pad audio to segment size BEFORE STFT
        // The model expects consistent spectrogram and audio inputs.
        // Padding audio before STFT ensures the spectrogram represents the padded audio.
        let segmentSize = 343980
        let (paddedLeft, paddedRight) = padAudioToSegment(
            left: leftChannel,
            right: rightChannel,
            targetLength: segmentSize
        )

        // 1. Run STFT on padded audio
        let (leftReal, leftImag) = try fft.stft(paddedLeft)
        let (rightReal, rightImag) = try fft.stft(paddedRight)

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

        // DEBUG: Print input statistics
        #if DEBUG
        print("DEBUG: stereoReal[0][0] first 5 = \(stereoReal[0][0].prefix(5))")
        print("DEBUG: stereoReal[0][1000] first 5 = \(stereoReal[0][1000].prefix(5))")
        print("DEBUG: paddedLeft first 5 = \(paddedLeft.prefix(5))")
        #endif

        // 3. Run hybrid inference with both spectrogram and padded audio
        // Returns separated spectrograms AND time-domain audio from both branches
        let (freqReal, freqImag, timeOutput) = try inference.predictHybrid(
            real: stereoReal,
            imag: stereoImag,
            rawAudio: [paddedLeft, paddedRight]
        )

        #if DEBUG
        print("DEBUG: freqReal[1][0][0] first 5 = \(freqReal[1][0][0].prefix(5))")  // bass, left, bin 0
        print("DEBUG: timeOutput[1][0] first 5 = \(timeOutput[1][0].prefix(5))")  // bass, left
        #endif

        // 4. For each stem: iSTFT(freq) + time
        // The hybrid model output is: final = time_output + istft(freq_output)
        var stemOutputs: [[[Float]]] = []

        for stemIdx in 0..<6 {
            let stemFreqReal = freqReal[stemIdx]  // [2 channels][2049 bins][timeFrames]
            let stemFreqImag = freqImag[stemIdx]  // [2 channels][2049 bins][timeFrames]
            let stemTimeAudio = timeOutput[stemIdx]  // [2 channels][samples]

            var channelOutputs: [[Float]] = []

            // Process left channel (0) and right channel (1)
            for channelIdx in 0..<2 {
                let channelReal = stemFreqReal[channelIdx]  // [2049 bins][timeFrames]
                let channelImag = stemFreqImag[channelIdx]  // [2049 bins][timeFrames]

                // Transpose from [bins][frames] to [frames][bins] for iSTFT
                let numFrames = channelReal[0].count
                let numBins = channelReal.count

                var realTransposed: [[Float]] = []
                var imagTransposed: [[Float]] = []

                for frameIdx in 0..<numFrames {
                    var realFrame = [Float](repeating: 0, count: numBins)
                    var imagFrame = [Float](repeating: 0, count: numBins)

                    for binIdx in 0..<numBins {
                        realFrame[binIdx] = channelReal[binIdx][frameIdx]
                        imagFrame[binIdx] = channelImag[binIdx][frameIdx]
                    }

                    realTransposed.append(realFrame)
                    imagTransposed.append(imagFrame)
                }

                // 5. Run iSTFT on frequency branch output
                let freqAudio = try fft.istft(real: realTransposed, imag: imagTransposed, length: originalLength)

                // 6. CRITICAL: Sum frequency and time branches
                // This is the core of the hybrid model: output = time + istft(freq)
                let timeAudio = stemTimeAudio[channelIdx]
                var combinedAudio = [Float](repeating: 0, count: originalLength)

                let minLength = min(freqAudio.count, timeAudio.count, originalLength)
                for i in 0..<minLength {
                    combinedAudio[i] = freqAudio[i] + timeAudio[i]
                }

                channelOutputs.append(combinedAudio)
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

    /// Pad stereo audio to target segment length using reflect mode
    /// - Parameters:
    ///   - left: Left channel samples
    ///   - right: Right channel samples
    ///   - targetLength: Target segment length in samples
    /// - Returns: Padded (left, right) channels
    private func padAudioToSegment(
        left: [Float],
        right: [Float],
        targetLength: Int
    ) -> (left: [Float], right: [Float]) {
        guard left.count < targetLength else {
            return (Array(left.prefix(targetLength)), Array(right.prefix(targetLength)))
        }

        let padNeeded = targetLength - left.count

        // Reflect padding (NumPy-compatible: doesn't include edge)
        func reflectPad(_ audio: [Float], _ padAmount: Int) -> [Float] {
            var result = audio
            let n = audio.count
            for i in 0..<padAmount {
                // NumPy reflect mode: starts at n-2, wraps around interior
                let period = max(1, n - 1)
                let reflectIdx = n - 2 - (i % period)
                let safeIdx = max(0, min(reflectIdx, n - 1))
                result.append(audio[safeIdx])
            }
            return result
        }

        return (reflectPad(left, padNeeded), reflectPad(right, padNeeded))
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
