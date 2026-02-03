import Accelerate
import Foundation

/// Swift wrapper for vDSP FFT operations
/// Provides STFT/iSTFT matching HTDemucs exactly
///
/// IMPORTANT: Uses HTDemucs-specific padding scheme (NOT standard center padding).
/// HTDemucs pads with hop_length//2*3 on left, and asymmetric right padding,
/// then trims the output frames to skip the first 2 and keep exactly `le` frames.
public class AudioFFT {
    // MARK: - Configuration (matches HTDemucs)
    public let fftSize: Int = 4096
    public let hopLength: Int = 1024
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

        // Pre-compute Hann window (denormalized for proper COLA)
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

    /// Compute inverse Short-Time Fourier Transform matching HTDemucs
    /// - Parameters:
    ///   - real: Real component [numFrames][numBins] where numBins=2048 (no Nyquist)
    ///   - imag: Imaginary component [numFrames][numBins]
    ///   - length: Original audio length to reconstruct
    /// - Returns: Reconstructed audio samples
    ///
    /// Handles the inverse of HTDemucs padding:
    /// - Adds 2 frames of zeros at start (to undo frame trimming)
    /// - Adds Nyquist bin (zeros) to each frame
    /// - Reconstructs with proper padding removal
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

        // Reconstruct with overlap-add
        for frameIdx in 0..<totalFrames {
            var frameReal = fullReal[frameIdx]
            var frameImag = fullImag[frameIdx]

            // Add Nyquist bin (zero) to restore 2049 bins for iFFT
            frameReal.append(0)
            frameImag.append(0)

            // Inverse FFT
            var timeFrame = performInverseRealFFT(frameReal, frameImag)

            // Apply window
            vDSP_vmul(timeFrame, 1, _window, 1, &timeFrame, 1, vDSP_Length(fftSize))

            // Overlap-add
            let start = frameIdx * hopLength
            for i in 0..<fftSize {
                output[start + i] += timeFrame[i]
                windowSum[start + i] += _window[i] * _window[i]
            }
        }

        // Normalize by window sum (COLA compliance)
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

    private func performRealFFT(_ input: [Float]) -> ([Float], [Float]) {
        let halfSize = fftSize / 2
        let numBins = halfSize + 1

        let interleaved = input

        splitComplexReal.withUnsafeMutableBufferPointer { realPtr in
            splitComplexImag.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(
                    realp: realPtr.baseAddress!,
                    imagp: imagPtr.baseAddress!
                )

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

                vDSP_fft_zrip(
                    fftSetup,
                    &splitComplex,
                    1,
                    log2n,
                    FFTDirection(FFT_FORWARD)
                )

                // Scale to match PyTorch normalized=True: 0.5/sqrt(N)
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

    private func performInverseRealFFT(_ real: [Float], _ imag: [Float]) -> [Float] {
        let halfSize = fftSize / 2

        // Handle case where Nyquist might not be present
        let nyquistReal = real.count > halfSize ? real[halfSize] : 0

        splitComplexReal[0] = real[0]
        splitComplexImag[0] = nyquistReal

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

                vDSP_fft_zrip(
                    fftSetup,
                    &splitComplex,
                    1,
                    log2n,
                    FFTDirection(FFT_INVERSE)
                )

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

        // Scale: sqrt(N)/N to undo forward normalization
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
