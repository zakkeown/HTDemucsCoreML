import Accelerate
import Foundation

/// Swift implementation of STFT/iSTFT using Apple's vDSP Accelerate framework.
///
/// This class provides Short-Time Fourier Transform (STFT) and its inverse (iSTFT)
/// that exactly match HTDemucs' PyTorch implementation. Matching is critical because
/// the neural network was trained on spectrograms with specific characteristics.
///
/// ## Why vDSP?
/// CoreML cannot handle complex-valued tensors, so STFT/iSTFT must be performed
/// externally. vDSP is Apple's hardware-optimized signal processing library,
/// providing excellent performance across all Apple Silicon devices.
///
/// ## Key Parameters
/// - **FFT Size (4096)**: Window length in samples. At 44.1 kHz, this is ~93ms.
///   Larger windows give better frequency resolution but worse time resolution.
/// - **Hop Length (1024)**: Step size between windows. With FFT size 4096, this
///   gives 75% overlap between consecutive frames (4096/1024 = 4x overlap).
/// - **Frequency Bins (2048)**: Output has 2048 bins (we drop Nyquist to match HTDemucs).
///   Full FFT would give 2049 bins (fftSize/2 + 1).
///
/// ## Normalization
/// PyTorch's `torch.stft(normalized=True)` divides by `sqrt(N)`. vDSP's FFT
/// returns values scaled by 2.0. We apply `0.5 / sqrt(N)` to match exactly.
/// See `performRealFFT()` for the scaling implementation.
///
/// ## HTDemucs-Specific Padding
/// HTDemucs uses non-standard padding (not center padding):
/// - Left pad: `hop_length // 2 * 3` = 1536 samples
/// - Right pad: Computed to align with frame count
/// - Frame trim: Skip first 2 frames, keep `ceil(length / hop_length)` frames
/// This padding scheme must be replicated exactly for parity.
public class AudioFFT {
    // MARK: - Configuration (matches HTDemucs)

    /// FFT window size in samples. 4096 at 44.1 kHz = ~93ms window.
    /// This determines frequency resolution: bin_width = sample_rate / fft_size = 10.77 Hz
    public let fftSize: Int = 4096

    /// Step size between consecutive FFT frames. 1024 samples = ~23ms hop.
    /// With fftSize=4096, this gives 75% overlap (4x redundancy).
    /// Higher overlap improves time resolution and reconstruction quality.
    public let hopLength: Int = 1024

    /// log2(fftSize) for vDSP FFT setup. 4096 = 2^12, so log2n = 12.
    private let log2n: vDSP_Length

    // MARK: - vDSP State
    private var fftSetup: FFTSetup
    private var _window: [Float]

    /// Access to the Hann window for testing
    public var window: [Float] { _window }

    // MARK: - Working Buffers
    private var splitComplexReal: [Float]
    private var splitComplexImag: [Float]
    private var windowedFrame: [Float]

    // MARK: - Initialization
    public init() throws {
        self.log2n = vDSP_Length(log2(Double(fftSize)))

        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            throw AudioFFTError.fftSetupFailed
        }
        self.fftSetup = setup

        // Pre-compute Hann window.
        // Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / N))
        // This tapers the edges to zero, reducing spectral leakage.
        // The window is symmetric and satisfies COLA (Constant Overlap-Add)
        // property with 75% overlap, enabling perfect reconstruction.
        self._window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&_window, vDSP_Length(fftSize), 0)

        // Allocate working buffers
        let halfSize = fftSize / 2
        self.splitComplexReal = [Float](repeating: 0, count: halfSize)
        self.splitComplexImag = [Float](repeating: 0, count: halfSize)
        self.windowedFrame = [Float](repeating: 0, count: fftSize)
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    // MARK: - Public API

    /// Compute Short-Time Fourier Transform matching HTDemucs._spec exactly
    /// - Parameter audio: Input audio samples
    /// - Returns: (real, imag) spectrograms, each [numFrames][numBins] where numBins=2048 (no Nyquist)
    /// - Throws: AudioFFTError if audio invalid
    ///
    /// Uses HTDemucs-specific padding:
    /// - pad_left = hop_length // 2 * 3 = 1536
    /// - le = ceil(length / hop_length)
    /// - pad_right = pad_left + le * hop_length - length
    /// Then trims output to frames [2:2+le] and drops Nyquist bin.
    public func stft(_ audio: [Float]) throws -> (real: [[Float]], imag: [[Float]]) {
        // Validate input
        guard audio.count > 0 else {
            throw AudioFFTError.audioTooShort(audio.count, 1)
        }
        guard audio.allSatisfy({ $0.isFinite }) else {
            throw AudioFFTError.invalidAudioData("NaN or Inf values")
        }

        // HTDemucs padding scheme
        let length = audio.count
        let le = Int(ceil(Double(length) / Double(hopLength)))
        let padLeft = hopLength / 2 * 3  // 1536
        let padRight = padLeft + le * hopLength - length

        let paddedAudio = applyHTDemucsReflectPadding(audio, padLeft: padLeft, padRight: padRight)

        // Compute all frames
        let numFramesRaw = (paddedAudio.count - fftSize) / hopLength + 1

        var realOutput: [[Float]] = []
        var imagOutput: [[Float]] = []

        // Process each frame
        for frameIdx in 0..<numFramesRaw {
            let start = frameIdx * hopLength
            let end = start + fftSize

            // Extract frame
            let frame = Array(paddedAudio[start..<end])

            // Apply window
            vDSP_vmul(frame, 1, _window, 1, &windowedFrame, 1, vDSP_Length(fftSize))

            // Perform FFT (returns 2049 bins including Nyquist)
            let (frameReal, frameImag) = performRealFFT(windowedFrame)

            // Drop Nyquist bin to match HTDemucs [:-1] - keep only first 2048 bins
            realOutput.append(Array(frameReal.dropLast()))
            imagOutput.append(Array(frameImag.dropLast()))
        }

        // HTDemucs trims: z[..., 2: 2 + le] - skip first 2 frames, keep le frames
        let startFrame = 2
        let endFrame = min(startFrame + le, realOutput.count)

        let trimmedReal = Array(realOutput[startFrame..<endFrame])
        let trimmedImag = Array(imagOutput[startFrame..<endFrame])

        return (real: trimmedReal, imag: trimmedImag)
    }

    /// Apply HTDemucs-style reflect padding
    private func applyHTDemucsReflectPadding(_ audio: [Float], padLeft: Int, padRight: Int) -> [Float] {
        var padded = [Float](repeating: 0, count: padLeft + audio.count + padRight)

        // Left padding: reflect from start (PyTorch reflect mode)
        for i in 0..<padLeft {
            // PyTorch reflect: for pad of size p, uses indices [p, p-1, ..., 1] then repeats
            let reflectIdx = (padLeft - i) % max(1, audio.count)
            let idx = reflectIdx == 0 ? 0 : reflectIdx
            padded[i] = audio[min(idx, audio.count - 1)]
        }

        // Copy original audio
        for i in 0..<audio.count {
            padded[padLeft + i] = audio[i]
        }

        // Right padding: reflect from end (PyTorch reflect mode)
        for i in 0..<padRight {
            // PyTorch reflect: uses indices [n-2, n-3, ..., 0, 1, 2, ...]
            let cycle = 2 * (audio.count - 1)
            let pos = (audio.count - 2 - i) % cycle
            let reflectIdx = pos < 0 ? -pos : (pos >= audio.count ? cycle - pos : pos)
            padded[padLeft + audio.count + i] = audio[max(0, min(reflectIdx, audio.count - 1))]
        }

        return padded
    }

    /// Compute inverse Short-Time Fourier Transform using overlap-add reconstruction.
    ///
    /// ## Overlap-Add Algorithm
    /// iSTFT reconstructs audio from spectrograms using weighted overlap-add:
    /// 1. For each spectrogram frame, apply inverse FFT to get a time-domain frame
    /// 2. Apply the same Hann window used in STFT (for COLA compliance)
    /// 3. Add the windowed frame to the output at the appropriate position
    /// 4. Track the sum of squared window values at each position
    /// 5. Normalize by the window sum to get the final output
    ///
    /// ## COLA (Constant Overlap-Add) Property
    /// With 75% overlap and Hann window, the squared window values sum to a constant:
    /// sum(window^2) = constant for all output positions (except edges)
    /// This ensures perfect reconstruction: iSTFT(STFT(x)) = x
    ///
    /// ## HTDemucs Padding Inversion
    /// STFT trimmed 2 frames from the start and removed the Nyquist bin.
    /// iSTFT must reverse this:
    /// - Add 2 zero frames at the start
    /// - Add zero Nyquist bin to each frame
    /// - Remove the padding added during STFT
    ///
    /// - Parameters:
    ///   - real: Real component [numFrames][numBins] where numBins=2048 (no Nyquist)
    ///   - imag: Imaginary component [numFrames][numBins]
    ///   - length: Original audio length to reconstruct (for padding removal)
    /// - Returns: Reconstructed audio samples
    public func istft(real: [[Float]], imag: [[Float]], length: Int? = nil) throws -> [Float] {
        guard real.count == imag.count else {
            throw AudioFFTError.mismatchedDimensions
        }

        let numFrames = real.count
        guard numFrames > 0 else {
            return []
        }

        // Determine target length
        let targetLength = length ?? (numFrames * hopLength)
        let le = Int(ceil(Double(targetLength) / Double(hopLength)))

        // HTDemucs padding parameters
        let padLeft = hopLength / 2 * 3  // 1536
        let padRight = padLeft + le * hopLength - targetLength

        // Add 2 zero frames at start to undo the trim [2:2+le]
        let zeroFrame = [Float](repeating: 0, count: fftSize / 2)  // 2048 bins
        var fullReal = [zeroFrame, zeroFrame] + real
        var fullImag = [zeroFrame, zeroFrame] + imag

        // Ensure we have enough frames
        let totalFrames = fullReal.count

        // Full padded output length
        let paddedOutputLength = (totalFrames - 1) * hopLength + fftSize

        var output = [Float](repeating: 0, count: paddedOutputLength)
        var windowSum = [Float](repeating: 0, count: paddedOutputLength)

        // Overlap-add reconstruction loop
        // Each frame is inverse-FFT'd, windowed, then added to the output at its position.
        // The windowSum tracks the accumulated window weight at each sample position.
        for frameIdx in 0..<totalFrames {
            var frameReal = fullReal[frameIdx]
            var frameImag = fullImag[frameIdx]

            // Add Nyquist bin (zero) to restore 2049 bins for iFFT
            // HTDemucs drops the Nyquist in STFT output, so we add it back as zero
            frameReal.append(0)
            frameImag.append(0)

            // Inverse FFT: spectrum -> time-domain frame
            var timeFrame = performInverseRealFFT(frameReal, frameImag)

            // Apply synthesis window (same Hann window used in analysis)
            // This is required for COLA compliance with overlap-add
            vDSP_vmul(timeFrame, 1, _window, 1, &timeFrame, 1, vDSP_Length(fftSize))

            // Overlap-add: accumulate windowed frame and squared window weights
            // Position = frameIdx * hopLength (each frame is offset by hop samples)
            let start = frameIdx * hopLength
            for i in 0..<fftSize {
                output[start + i] += timeFrame[i]
                windowSum[start + i] += _window[i] * _window[i]  // Track window^2 for normalization
            }
        }

        // Normalize by accumulated squared window sum
        // With Hann window at 75% overlap, windowSum should be nearly constant (~1.5)
        // except at the edges. Division recovers the original amplitude.
        // The 1e-8 threshold prevents division by zero at silent edges.
        for i in 0..<paddedOutputLength where windowSum[i] > 1e-8 {
            output[i] /= windowSum[i]
        }

        // Remove HTDemucs padding
        let startIdx = padLeft
        let endIdx = min(startIdx + targetLength, output.count)

        // Ensure valid indices
        let safeStart = max(0, min(startIdx, output.count))
        let safeEnd = max(safeStart, min(endIdx, output.count))

        return Array(output[safeStart..<safeEnd])
    }

    // MARK: - Legacy API (for backwards compatibility)

    /// Legacy center-padding STFT (use stft() for HTDemucs compatibility)
    public func stftCenterPadding(_ audio: [Float]) throws -> (real: [[Float]], imag: [[Float]]) {
        guard audio.count > 0 else {
            throw AudioFFTError.audioTooShort(audio.count, 1)
        }
        guard audio.allSatisfy({ $0.isFinite }) else {
            throw AudioFFTError.invalidAudioData("NaN or Inf values")
        }

        let padSize = fftSize / 2
        let paddedAudio = applyCenterPadding(audio, padSize: padSize)
        let numFrames = (paddedAudio.count - fftSize) / hopLength + 1

        var realOutput: [[Float]] = []
        var imagOutput: [[Float]] = []

        for frameIdx in 0..<numFrames {
            let start = frameIdx * hopLength
            let frame = Array(paddedAudio[start..<(start + fftSize)])
            vDSP_vmul(frame, 1, _window, 1, &windowedFrame, 1, vDSP_Length(fftSize))
            let (frameReal, frameImag) = performRealFFT(windowedFrame)
            realOutput.append(frameReal)
            imagOutput.append(frameImag)
        }

        return (real: realOutput, imag: imagOutput)
    }

    private func applyCenterPadding(_ audio: [Float], padSize: Int) -> [Float] {
        var padded = [Float](repeating: 0, count: audio.count + 2 * padSize)

        for i in 0..<padSize {
            let reflectIdx = padSize - i
            if reflectIdx < audio.count {
                padded[i] = audio[reflectIdx]
            } else {
                padded[i] = audio[reflectIdx % max(1, audio.count)]
            }
        }

        for i in 0..<audio.count {
            padded[padSize + i] = audio[i]
        }

        for i in 0..<padSize {
            let reflectIdx = audio.count - 2 - i
            if reflectIdx >= 0 {
                padded[padSize + audio.count + i] = audio[reflectIdx]
            } else {
                let wrappedIdx = ((reflectIdx % audio.count) + audio.count) % max(1, audio.count)
                padded[padSize + audio.count + i] = audio[wrappedIdx]
            }
        }

        return padded
    }

    // MARK: - Private Helpers

    /// Perform forward real-to-complex FFT on a windowed frame.
    ///
    /// ## vDSP FFT Peculiarities
    /// vDSP's `vDSP_fft_zrip` is an "in-place" FFT that uses a packed format:
    /// - Input: Real samples packed as interleaved pairs
    /// - Output: DC component in real[0], Nyquist in imag[0], rest in split complex
    ///
    /// ## Normalization to Match PyTorch
    /// PyTorch's `torch.stft(normalized=True)` divides output by `sqrt(N)`.
    /// vDSP's FFT returns values scaled by 2.0 (due to the packed format).
    /// To match PyTorch: multiply by `0.5 / sqrt(N)`.
    ///
    /// Math: vDSP_output * 0.5 / sqrt(N) = PyTorch_output
    ///
    /// - Parameter input: Windowed audio frame of length fftSize
    /// - Returns: (real, imag) arrays of length fftSize/2 + 1 (2049 bins)
    private func performRealFFT(_ input: [Float]) -> ([Float], [Float]) {
        let halfSize = fftSize / 2
        let numBins = halfSize + 1  // 2049 bins including DC and Nyquist

        let interleaved = input

        splitComplexReal.withUnsafeMutableBufferPointer { realPtr in
            splitComplexImag.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(
                    realp: realPtr.baseAddress!,
                    imagp: imagPtr.baseAddress!
                )

                // Convert interleaved real input to split complex format for vDSP
                interleaved.withUnsafeBytes { inputPtr in
                    let inputFloat = inputPtr.bindMemory(to: Float.self)
                    vDSP_ctoz(
                        UnsafePointer<DSPComplex>(OpaquePointer(inputFloat.baseAddress!)),
                        2,
                        &splitComplex,
                        1,
                        vDSP_Length(halfSize)
                    )
                }

                // Perform forward FFT (real input -> complex output)
                vDSP_fft_zrip(
                    fftSetup,
                    &splitComplex,
                    1,
                    log2n,
                    FFTDirection(FFT_FORWARD)
                )

                // CRITICAL: Scale to match PyTorch's normalized=True
                // vDSP returns values 2x larger due to packed format, and PyTorch
                // divides by sqrt(N). Combined factor: 0.5 / sqrt(N)
                var scale: Float = 0.5 / Float(fftSize).squareRoot()
                vDSP_vsmul(realPtr.baseAddress!, 1, &scale, realPtr.baseAddress!, 1, vDSP_Length(halfSize))
                vDSP_vsmul(imagPtr.baseAddress!, 1, &scale, imagPtr.baseAddress!, 1, vDSP_Length(halfSize))
            }
        }

        var realOutput = [Float](repeating: 0, count: numBins)
        var imagOutput = [Float](repeating: 0, count: numBins)

        // DC bin
        realOutput[0] = splitComplexReal[0]
        imagOutput[0] = 0

        // Positive frequencies
        for i in 1..<halfSize {
            realOutput[i] = splitComplexReal[i]
            imagOutput[i] = splitComplexImag[i]
        }

        // Nyquist bin
        realOutput[halfSize] = splitComplexImag[0]
        imagOutput[halfSize] = 0

        return (realOutput, imagOutput)
    }

    /// Perform inverse complex-to-real FFT to reconstruct a time-domain frame.
    ///
    /// ## Inverse Normalization
    /// The forward FFT applied `0.5 / sqrt(N)`. To reconstruct the original signal,
    /// the inverse must apply `sqrt(N) / N` (which simplifies to `1 / sqrt(N)`).
    ///
    /// However, vDSP's inverse FFT is unnormalized, so we need to divide by N.
    /// Combined with undoing our forward scaling: `sqrt(N) / N`.
    ///
    /// ## vDSP Packed Format
    /// vDSP expects the inverse input in its packed format:
    /// - DC component in real[0]
    /// - Nyquist in imag[0]
    /// - Positive frequencies in the rest of the split complex array
    ///
    /// - Parameters:
    ///   - real: Real component of spectrum (2049 bins)
    ///   - imag: Imaginary component of spectrum (2049 bins)
    /// - Returns: Reconstructed time-domain frame of length fftSize
    private func performInverseRealFFT(_ real: [Float], _ imag: [Float]) -> [Float] {
        let halfSize = fftSize / 2

        // Pack into vDSP format: DC in real[0], Nyquist in imag[0]
        let nyquistReal = real.count > halfSize ? real[halfSize] : 0

        splitComplexReal[0] = real[0]      // DC component
        splitComplexImag[0] = nyquistReal  // Nyquist component

        // Copy positive frequency bins
        for i in 1..<halfSize {
            splitComplexReal[i] = i < real.count ? real[i] : 0
            splitComplexImag[i] = i < imag.count ? imag[i] : 0
        }

        var output = [Float](repeating: 0, count: fftSize)

        splitComplexReal.withUnsafeMutableBufferPointer { realPtr in
            splitComplexImag.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(
                    realp: realPtr.baseAddress!,
                    imagp: imagPtr.baseAddress!
                )

                // Perform inverse FFT (complex input -> real output)
                vDSP_fft_zrip(
                    fftSetup,
                    &splitComplex,
                    1,
                    log2n,
                    FFTDirection(FFT_INVERSE)
                )

                // Convert from split complex back to interleaved real
                output.withUnsafeMutableBytes { outputPtr in
                    let outputFloat = outputPtr.bindMemory(to: Float.self)
                    vDSP_ztoc(
                        &splitComplex,
                        1,
                        UnsafeMutablePointer<DSPComplex>(OpaquePointer(outputFloat.baseAddress!)),
                        2,
                        vDSP_Length(halfSize)
                    )
                }
            }
        }

        // CRITICAL: Inverse scaling to undo forward normalization
        // Forward: 0.5 / sqrt(N), so inverse needs: sqrt(N) / N = 1 / sqrt(N)
        // But vDSP's inverse is unnormalized, so we also divide by N.
        // Final scale: sqrt(N) / N
        var scale = Float(fftSize).squareRoot() / Float(fftSize)
        vDSP_vsmul(output, 1, &scale, &output, 1, vDSP_Length(fftSize))

        return output
    }
}

// MARK: - Error Types

public enum AudioFFTError: Error, LocalizedError {
    case fftSetupFailed
    case audioTooShort(Int, Int)
    case invalidAudioData(String)
    case mismatchedDimensions

    public var errorDescription: String? {
        switch self {
        case .fftSetupFailed:
            return "Failed to create FFT setup"
        case .audioTooShort(let actual, let required):
            return "Audio too short: \(actual) samples, need at least \(required)"
        case .invalidAudioData(let reason):
            return "Invalid audio data: \(reason)"
        case .mismatchedDimensions:
            return "Real and imaginary spectrograms have mismatched dimensions"
        }
    }
}
