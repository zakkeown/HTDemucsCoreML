import Accelerate
import Foundation

/// Swift wrapper for vDSP FFT operations
/// Provides STFT/iSTFT with PyTorch-compatible parameters
public class AudioFFT {
    // MARK: - Configuration (matches PyTorch)
    public let fftSize: Int = 4096
    public let hopLength: Int = 1024
    private let log2n: vDSP_Length

    // MARK: - vDSP State
    private var fftSetup: FFTSetup
    private var window: [Float]

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

        // Pre-compute Hann window
        self.window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))

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

    /// Compute Short-Time Fourier Transform
    /// - Parameter audio: Input audio samples
    /// - Returns: (real, imag) spectrograms, each [numFrames][numBins]
    /// - Throws: AudioFFTError if audio invalid
    public func stft(_ audio: [Float]) throws -> (real: [[Float]], imag: [[Float]]) {
        // Validate input
        guard audio.count >= fftSize else {
            throw AudioFFTError.audioTooShort(audio.count, fftSize)
        }
        guard audio.allSatisfy({ $0.isFinite }) else {
            throw AudioFFTError.invalidAudioData("NaN or Inf values")
        }

        // Compute number of frames
        let numFrames = (audio.count - fftSize) / hopLength + 1

        var realOutput: [[Float]] = []
        var imagOutput: [[Float]] = []

        // Process each frame
        for frameIdx in 0..<numFrames {
            let start = frameIdx * hopLength
            let end = start + fftSize

            // Extract frame
            let frame = Array(audio[start..<end])

            // Apply window
            vDSP_vmul(frame, 1, window, 1, &windowedFrame, 1, vDSP_Length(fftSize))

            // Perform FFT
            let (frameReal, frameImag) = performRealFFT(windowedFrame)

            realOutput.append(frameReal)
            imagOutput.append(frameImag)
        }

        return (real: realOutput, imag: imagOutput)
    }

    /// Compute inverse Short-Time Fourier Transform
    /// - Parameters:
    ///   - real: Real component [numFrames][numBins]
    ///   - imag: Imaginary component [numFrames][numBins]
    /// - Returns: Reconstructed audio samples
    public func istft(real: [[Float]], imag: [[Float]]) throws -> [Float] {
        guard real.count == imag.count else {
            throw AudioFFTError.mismatchedDimensions
        }

        // TODO: Implement iSTFT logic
        return []
    }

    // MARK: - Private Helpers

    private func performRealFFT(_ input: [Float]) -> ([Float], [Float]) {
        let halfSize = fftSize / 2
        let numBins = halfSize + 1

        // Convert to interleaved format for vDSP
        let interleaved = input

        // Use withUnsafeMutableBufferPointer for proper pointer lifetime
        splitComplexReal.withUnsafeMutableBufferPointer { realPtr in
            splitComplexImag.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(
                    realp: realPtr.baseAddress!,
                    imagp: imagPtr.baseAddress!
                )

                // Convert real input to split complex
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

                // Perform forward FFT
                vDSP_fft_zrip(
                    fftSetup,
                    &splitComplex,
                    1,
                    log2n,
                    FFTDirection(FFT_FORWARD)
                )

                // Scale output (vDSP doesn't scale forward transform)
                var scale = Float(0.5)
                vDSP_vsmul(realPtr.baseAddress!, 1, &scale, realPtr.baseAddress!, 1, vDSP_Length(halfSize))
                vDSP_vsmul(imagPtr.baseAddress!, 1, &scale, imagPtr.baseAddress!, 1, vDSP_Length(halfSize))
            }
        }

        // Extract bins (DC and Nyquist packed in splitComplexReal[0], splitComplexImag[0])
        var realOutput = [Float](repeating: 0, count: numBins)
        var imagOutput = [Float](repeating: 0, count: numBins)

        // DC bin (purely real)
        realOutput[0] = splitComplexReal[0]
        imagOutput[0] = 0

        // Positive frequencies
        for i in 1..<halfSize {
            realOutput[i] = splitComplexReal[i]
            imagOutput[i] = splitComplexImag[i]
        }

        // Nyquist bin (purely real, packed in imaginary DC)
        realOutput[halfSize] = splitComplexImag[0]
        imagOutput[halfSize] = 0

        return (realOutput, imagOutput)
    }
}

// MARK: - Error Types

public enum AudioFFTError: Error, LocalizedError {
    case fftSetupFailed
    case audioTooShort(Int, Int)  // (actual, required)
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
