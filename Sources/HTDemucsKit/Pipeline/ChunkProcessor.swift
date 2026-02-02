import Foundation

/// Processes long audio in chunks with overlap-add for seamless blending
public class ChunkProcessor {
    // MARK: - Configuration
    public let chunkDuration: Float = 10.0  // seconds
    public let overlapDuration: Float = 1.0 // seconds per side
    public let sampleRate: Int = 44100

    // MARK: - Computed Properties
    private var chunkSamples: Int { Int(chunkDuration * Float(sampleRate)) }
    private var overlapSamples: Int { Int(overlapDuration * Float(sampleRate)) }
    private var hopSamples: Int { chunkSamples - 2 * overlapSamples }

    public init() {}

    /// Process audio in chunks with overlap-add
    /// - Parameters:
    ///   - audio: Input audio samples
    ///   - processor: Function to process each chunk
    ///   - progressCallback: Optional callback for chunk progress (chunkIndex, totalChunks)
    /// - Returns: Processed audio with seamless blending
    public func processInChunks(
        audio: [Float],
        processor: ([Float]) throws -> [Float],
        progressCallback: ((Int, Int) -> Void)? = nil
    ) rethrows -> [Float] {
        guard audio.count > 0 else { return [] }

        // If audio shorter than chunk size, process directly
        if audio.count <= chunkSamples {
            progressCallback?(0, 1)
            let result = try processor(audio)
            progressCallback?(1, 1)
            return result
        }

        var output = [Float](repeating: 0, count: audio.count)
        var weights = [Float](repeating: 0, count: audio.count)

        let numChunks = (audio.count - overlapSamples * 2 + hopSamples - 1) / hopSamples

        for chunkIdx in 0..<numChunks {
            // Report progress before processing
            progressCallback?(chunkIdx, numChunks)

            let start = chunkIdx * hopSamples
            let end = min(start + chunkSamples, audio.count)

            // Extract chunk
            let chunk = Array(audio[start..<end])

            // Process chunk
            let processed = try processor(chunk)

            // Create blend window
            let window = createBlendWindow(
                chunkSize: processed.count,
                overlapSize: overlapSamples,
                isFirst: chunkIdx == 0,
                isLast: end >= audio.count
            )

            // Accumulate with blending
            for (i, value) in processed.enumerated() {
                let outputIdx = start + i
                if outputIdx < output.count {
                    output[outputIdx] += value * window[i]
                    weights[outputIdx] += window[i]
                }
            }
        }

        // Report completion
        progressCallback?(numChunks, numChunks)

        // Normalize by accumulated weights
        for i in 0..<audio.count where weights[i] > 0 {
            output[i] /= weights[i]
        }

        return output
    }

    // MARK: - Private Helpers

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
}
