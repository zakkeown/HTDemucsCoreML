import Foundation

/// Processes long audio in chunks with overlap-add for seamless blending.
///
/// ## Why Chunking?
/// The CoreML model has fixed input dimensions (~7.8 seconds of audio).
/// For longer audio, we split into overlapping chunks, process each independently,
/// then blend them together using crossfade in the overlap regions.
///
/// ## Chunk Layout
/// ```
/// |-------- Chunk 0 --------|
///                  |-------- Chunk 1 --------|
///                                   |-------- Chunk 2 --------|
/// [======overlap=====][===hop===][======overlap=====]
/// ```
///
/// Each chunk overlaps its neighbors by `overlapDuration` on each side.
/// The non-overlapping center portion is `hopSamples = chunkSamples - 2 * overlapSamples`.
///
/// ## Crossfade Blending
/// In overlap regions, we blend using linear crossfade:
/// - Previous chunk fades out: weight decreases 1.0 -> 0.0
/// - Current chunk fades in: weight increases 0.0 -> 1.0
/// This eliminates discontinuities at chunk boundaries.
///
/// ## Edge Cases
/// - **Audio shorter than chunk size**: Process as single chunk, no blending needed
/// - **Last chunk shorter than full size**: Still processed, crossfade handles boundary
public class ChunkProcessor {
    // MARK: - Configuration

    /// Duration of each chunk in seconds.
    /// Chosen to provide sufficient context for the model while keeping memory reasonable.
    public let chunkDuration: Float = 10.0

    /// Overlap duration on each side of chunk boundaries, in seconds.
    /// 1 second provides enough samples for smooth crossfade blending.
    public let overlapDuration: Float = 1.0

    /// Expected sample rate. HTDemucs requires 44.1 kHz.
    public let sampleRate: Int = 44100

    // MARK: - Computed Properties

    /// Chunk size in samples: 10s * 44100 Hz = 441,000 samples
    private var chunkSamples: Int { Int(chunkDuration * Float(sampleRate)) }

    /// Overlap size in samples: 1s * 44100 Hz = 44,100 samples per side
    private var overlapSamples: Int { Int(overlapDuration * Float(sampleRate)) }

    /// Hop between chunk start positions: chunkSamples - 2 * overlapSamples
    /// With 10s chunks and 1s overlap: 441000 - 88200 = 352,800 samples
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

    /// Create blend window with linear crossfade for overlap-add reconstruction.
    ///
    /// ## Window Shape
    /// ```
    /// First chunk:     [=====1.0=====][fade out]
    /// Middle chunk:    [fade in][=====1.0=====][fade out]
    /// Last chunk:      [fade in][=====1.0=====]
    /// ```
    ///
    /// ## Crossfade Math
    /// When chunks overlap, their windows sum to 1.0:
    /// - Chunk N fade-out at position i: weight = 1.0 - i/overlap
    /// - Chunk N+1 fade-in at position i: weight = i/overlap
    /// - Sum: (1.0 - i/overlap) + (i/overlap) = 1.0
    ///
    /// This ensures constant gain through transitions.
    ///
    /// - Parameters:
    ///   - chunkSize: Actual size of this chunk (may be shorter than standard for last chunk)
    ///   - overlapSize: Number of samples in overlap region
    ///   - isFirst: True if this is the first chunk (no fade-in needed)
    ///   - isLast: True if this is the last chunk (no fade-out needed)
    /// - Returns: Window array with values in [0.0, 1.0]
    private func createBlendWindow(
        chunkSize: Int,
        overlapSize: Int,
        isFirst: Bool,
        isLast: Bool
    ) -> [Float] {
        var window = [Float](repeating: 1.0, count: chunkSize)

        // Fade in at start (unless first chunk)
        // Linear ramp: 0.0 -> 1.0 over overlapSize samples
        if !isFirst && chunkSize >= overlapSize {
            for i in 0..<min(overlapSize, chunkSize) {
                window[i] = Float(i) / Float(overlapSize)
            }
        }

        // Fade out at end (unless last chunk)
        // Linear ramp: 1.0 -> 0.0 over overlapSize samples
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
