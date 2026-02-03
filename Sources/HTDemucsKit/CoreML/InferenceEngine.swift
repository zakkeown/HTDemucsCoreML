import CoreML
import Foundation

/// Runs CoreML inference on audio spectrograms and raw audio using the hybrid HTDemucs model.
///
/// ## Hybrid Model Architecture
/// HTDemucs is a hybrid model with two branches:
/// - **Frequency branch**: Processes spectrograms through encoder/transformer/decoder
/// - **Time branch**: Processes raw audio through a parallel network
///
/// The final output combines both: `stem = time_output + istft(freq_output)`
///
/// ## Input Tensor Shapes
/// The model accepts two inputs:
/// - `spectrogram`: `[1, 4, 2048, 336]` - batch, channels, freq bins, time frames
///   - Channels are Complex-as-Channels: `[L_real, L_imag, R_real, R_imag]`
/// - `raw_audio`: `[1, 2, 343980]` - batch, channels, samples
///   - Stereo audio at 44.1 kHz for ~7.8 seconds
///
/// ## Output Tensor Shapes
/// The model returns two outputs:
/// - `add_66` (freq output): `[1, 6, 4, 2048, 336]` - batch, stems, channels, freq, time
/// - `add_67` (time output): `[1, 6, 2, 343980]` - batch, stems, channels, samples
///
/// ## Stem Ordering (CRITICAL)
/// The model outputs stems in this order (must match PyTorch exactly):
/// ```
/// Index 0: drums
/// Index 1: bass
/// Index 2: other   <-- NOT vocals!
/// Index 3: vocals  <-- NOT other!
/// Index 4: guitar
/// Index 5: piano
/// ```
/// Getting this wrong will swap vocals and "other" instruments.
public class InferenceEngine {
    private let model: MLModel

    // MARK: - Model Configuration Constants
    // These must match the CoreML model's fixed input dimensions.
    // The model was converted with these specific sizes for the training segment.

    /// Fixed time frames in spectrogram input (336 frames for 7.8s segment)
    private let modelTimeFrames = 336

    /// Fixed audio samples in raw audio input (343,980 samples = 7.8s at 44.1kHz)
    private let modelAudioSamples = 343980

    /// Frequency bins in spectrogram (2048, excluding Nyquist)
    private let freqBins = 2048

    /// Number of separation stems (drums, bass, other, vocals, guitar, piano)
    private let numStems = 6

    /// Initialize with loaded CoreML model
    /// - Parameter model: The MLModel instance from ModelLoader
    public init(model: MLModel) {
        self.model = model
    }

    /// Run inference on stereo spectrogram and raw audio (hybrid model)
    /// - Parameters:
    ///   - real: Real component [2][freqBins][timeFrames] (stereo, freq, time)
    ///   - imag: Imaginary component [2][freqBins][timeFrames]
    ///   - rawAudio: Raw stereo audio [2][samples]
    /// - Returns: Tuple of (freqReal, freqImag, timeOutput) for combining with iSTFT
    /// - Throws: InferenceError if prediction fails
    public func predictHybrid(
        real: [[[Float]]],
        imag: [[[Float]]],
        rawAudio: [[Float]]
    ) throws -> (freqReal: [[[[Float]]]], freqImag: [[[[Float]]]], timeOutput: [[[Float]]]) {
        // Validate input shapes
        guard real.count == 2, imag.count == 2 else {
            throw InferenceError.invalidInputShape("Expected stereo input (2 channels)")
        }

        guard real[0].count == 2049 else {
            throw InferenceError.invalidInputShape("Expected 2049 frequency bins")
        }

        guard rawAudio.count == 2 else {
            throw InferenceError.invalidInputShape("Expected stereo raw audio (2 channels)")
        }

        let inputTimeFrames = real[0][0].count
        let inputAudioSamples = rawAudio[0].count

        #if DEBUG
        print("DEBUG InferenceEngine: inputTimeFrames = \(inputTimeFrames), modelTimeFrames = \(modelTimeFrames)")
        print("DEBUG InferenceEngine: inputAudioSamples = \(inputAudioSamples), modelAudioSamples = \(modelAudioSamples)")
        print("DEBUG InferenceEngine: real.count = \(real.count), real[0].count = \(real[0].count)")
        #endif

        // Pad or trim inputs to match model's expected sizes
        let (paddedReal, paddedImag) = padOrTrimSpectrogram(
            real: real,
            imag: imag,
            targetFrames: modelTimeFrames
        )

        let paddedAudio = padOrTrimAudio(
            audio: rawAudio,
            targetSamples: modelAudioSamples
        )

        // Create spectrogram MLMultiArray [1, 4, 2048, 336]
        // Note: Model expects 2048 bins (Nyquist bin excluded)
        let spectrogramInput = try createCombinedMLMultiArray(
            real: paddedReal,
            imag: paddedImag,
            shape: [1, 4, freqBins, modelTimeFrames]
        )

        #if DEBUG
        // Verify the MLMultiArray content
        print("DEBUG InferenceEngine: spectrogramInput[0:5] = \((0..<5).map { spectrogramInput[$0].floatValue })")
        print("DEBUG InferenceEngine: spectrogramInput.shape = \(spectrogramInput.shape)")
        print("DEBUG InferenceEngine: paddedAudio[0][0:5] = \(paddedAudio[0].prefix(5))")
        print("DEBUG InferenceEngine: paddedAudio[0].count = \(paddedAudio[0].count)")
        #endif

        // Create raw audio MLMultiArray [1, 2, 343980]
        let audioInput = try createAudioMLMultiArray(
            audio: paddedAudio,
            shape: [1, 2, modelAudioSamples]
        )

        #if DEBUG
        // Verify the audio MLMultiArray content
        print("DEBUG InferenceEngine: audioInput[0:5] = \((0..<5).map { audioInput[$0].floatValue })")
        print("DEBUG InferenceEngine: audioInput.shape = \(audioInput.shape)")
        #endif

        // Create input features with both inputs
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "spectrogram": MLFeatureValue(multiArray: spectrogramInput),
            "raw_audio": MLFeatureValue(multiArray: audioInput)
        ])

        // Run prediction
        let output = try model.prediction(from: inputFeatures)

        // Extract frequency output (add_66) - shape [1, 6, 4, 2048, 336]
        guard let freqOutputArray = output.featureValue(for: "add_66")?.multiArrayValue else {
            throw InferenceError.invalidOutput("add_66 output not found")
        }

        // Extract time output (add_67) - shape [1, 6, 2, 343980]
        guard let timeOutputArray = output.featureValue(for: "add_67")?.multiArrayValue else {
            throw InferenceError.invalidOutput("add_67 output not found")
        }

        // Convert outputs to arrays
        let (freqReal, freqImag) = try convertFreqOutputToSpectrograms(freqOutputArray, timeFrames: modelTimeFrames)
        let timeOutput = try convertTimeOutputToAudio(timeOutputArray, samples: modelAudioSamples)

        // Trim outputs back to original sizes if we padded
        let (trimmedFreqReal, trimmedFreqImag) = inputTimeFrames < modelTimeFrames
            ? trimSpectrograms(real: freqReal, imag: freqImag, targetFrames: inputTimeFrames)
            : (freqReal, freqImag)

        let trimmedTimeOutput = inputAudioSamples < modelAudioSamples
            ? trimAudio(audio: timeOutput, targetSamples: inputAudioSamples)
            : timeOutput

        return (trimmedFreqReal, trimmedFreqImag, trimmedTimeOutput)
    }

    /// Run inference on stereo spectrogram (legacy single-branch model)
    /// - Parameters:
    ///   - real: Real component [2][freqBins][timeFrames] (stereo, freq, time)
    ///   - imag: Imaginary component [2][freqBins][timeFrames]
    /// - Returns: Masks [6][2][freqBins][timeFrames] (6 stems, stereo, freq, time)
    /// - Throws: InferenceError if prediction fails
    public func predict(real: [[[Float]]], imag: [[[Float]]]) throws -> (real: [[[[Float]]]], imag: [[[[Float]]]]) {
        // Validate input shapes
        guard real.count == 2, imag.count == 2 else {
            throw InferenceError.invalidInputShape("Expected stereo input (2 channels)")
        }

        guard real[0].count == 2049 else {
            throw InferenceError.invalidInputShape("Expected 2049 frequency bins")
        }

        let inputTimeFrames = real[0][0].count

        // Model expects exactly 431 time frames (for 10s chunks at 44.1kHz with hop=1024)
        let modelTimeFrames = 431

        // Pad or trim input to match model's expected size
        let (paddedReal, paddedImag) = padOrTrimSpectrogram(
            real: real,
            imag: imag,
            targetFrames: modelTimeFrames
        )

        // Model expects input "x" with shape [1, 4, 2049, 431]
        // where the 4 channels are: [real_left, imag_left, real_right, imag_right]
        let combinedInput = try createCombinedMLMultiArray(
            real: paddedReal,
            imag: paddedImag,
            shape: [1, 4, 2049, modelTimeFrames]
        )

        // Create input features with the expected input name "x"
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: combinedInput)
        ])

        // Run prediction
        let output = try model.prediction(from: inputFeatures)

        // Extract output - model returns "var_2155" with shape [1, 6, 4, 2048, 431]
        // where 4 channels are [real_left, imag_left, real_right, imag_right] for each stem
        guard let outputArray = output.featureValue(for: "var_2155")?.multiArrayValue else {
            throw InferenceError.invalidOutput("var_2155 output not found")
        }

        // Convert output to separated spectrograms [6][2][2049][modelTimeFrames]
        let (separatedReal, separatedImag) = try convertOutputToSpectrograms(outputArray, timeFrames: modelTimeFrames)

        // Trim spectrograms back to original input size if we padded
        if inputTimeFrames < modelTimeFrames {
            return trimSpectrograms(real: separatedReal, imag: separatedImag, targetFrames: inputTimeFrames)
        }

        return (separatedReal, separatedImag)
    }

    // MARK: - Private Helpers

    /// Create combined MLMultiArray with Complex-as-Channels format.
    ///
    /// ## Memory Layout
    /// CoreML expects row-major (C-style) memory layout. For shape `[1, 4, F, T]`:
    /// - Outermost dimension (batch) varies slowest
    /// - Innermost dimension (time) varies fastest
    ///
    /// ## Channel Ordering
    /// The 4 channels encode stereo complex spectrogram as real-valued tensor:
    /// ```
    /// Channel 0: Left real      (L_re)
    /// Channel 1: Left imaginary (L_im)
    /// Channel 2: Right real     (R_re)
    /// Channel 3: Right imaginary (R_im)
    /// ```
    ///
    /// This "Complex-as-Channels" format allows CoreML to process spectrograms
    /// without native complex number support.
    ///
    /// - Parameters:
    ///   - real: Real component `[2 channels][freq bins][time frames]`
    ///   - imag: Imaginary component `[2 channels][freq bins][time frames]`
    ///   - shape: Target MLMultiArray shape `[1, 4, freq, time]`
    /// - Returns: MLMultiArray ready for CoreML inference
    private func createCombinedMLMultiArray(
        real: [[[Float]]],
        imag: [[[Float]]],
        shape: [Int]
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)

        // Fill array in row-major order: [batch=1][4 channels][freq bins][time frames]
        // Index increments: time (fastest) -> freq -> channel -> batch (slowest)
        var index = 0

        // Channel 0: Left channel real component
        for freq in 0..<shape[2] {
            for time in 0..<shape[3] {
                array[index] = NSNumber(value: real[0][freq][time])
                index += 1
            }
        }

        // Channel 1: Left channel imaginary component
        for freq in 0..<shape[2] {
            for time in 0..<shape[3] {
                array[index] = NSNumber(value: imag[0][freq][time])
                index += 1
            }
        }

        // Channel 2: Right channel real component
        for freq in 0..<shape[2] {
            for time in 0..<shape[3] {
                array[index] = NSNumber(value: real[1][freq][time])
                index += 1
            }
        }

        // Channel 3: Right channel imaginary component
        for freq in 0..<shape[2] {
            for time in 0..<shape[3] {
                array[index] = NSNumber(value: imag[1][freq][time])
                index += 1
            }
        }

        return array
    }

    /// Convert model output to masks
    /// - Parameters:
    ///   - mlArray: Output MLMultiArray [1, 6, 4, 2048, timeFrames]
    ///   - timeFrames: Number of time frames
    /// - Returns: Masks [6 stems][2 channels][2049 bins][timeFrames]
    private func convertOutputToSpectrograms(_ mlArray: MLMultiArray, timeFrames: Int) throws -> (real: [[[[Float]]]], imag: [[[[Float]]]]) {
        // Model output is [1, 6 stems, 4 channels, 2048 freq bins, timeFrames]
        // Channels are: [left_real, left_imag, right_real, right_imag]
        // We need to return [6 stems][2 channels][2049 bins][timeFrames] for both real and imag

        let numStems = 6
        let numChannels = 2
        let numBins = 2049  // We need to pad 2048 → 2049

        var resultReal: [[[[Float]]]] = []
        var resultImag: [[[[Float]]]] = []

        for stemIdx in 0..<numStems {
            var stemDataReal: [[[Float]]] = []
            var stemDataImag: [[[Float]]] = []

            for channelIdx in 0..<numChannels {
                var channelDataReal: [[Float]] = []
                var channelDataImag: [[Float]] = []

                for binIdx in 0..<numBins {
                    var binDataReal: [Float] = []
                    var binDataImag: [Float] = []

                    for timeIdx in 0..<timeFrames {
                        // Calculate index in row-major order
                        // [batch=0][stem][real/imag channel pair][freq][time]
                        // For left channel: use real (channel 0) and imag (channel 1)
                        // For right channel: use real (channel 2) and imag (channel 3)

                        if binIdx < 2048 {
                            let realChannelIdx = channelIdx == 0 ? 0 : 2
                            let imagChannelIdx = channelIdx == 0 ? 1 : 3

                            // Read BOTH real and imaginary components
                            let realIndex = (((stemIdx * 4 + realChannelIdx) * 2048) + binIdx) * timeFrames + timeIdx
                            let imagIndex = (((stemIdx * 4 + imagChannelIdx) * 2048) + binIdx) * timeFrames + timeIdx

                            binDataReal.append(mlArray[realIndex].floatValue)
                            binDataImag.append(mlArray[imagIndex].floatValue)
                        } else {
                            // Pad with 0 for bin 2048 (DC/Nyquist)
                            binDataReal.append(0.0)
                            binDataImag.append(0.0)
                        }
                    }

                    channelDataReal.append(binDataReal)
                    channelDataImag.append(binDataImag)
                }

                stemDataReal.append(channelDataReal)
                stemDataImag.append(channelDataImag)
            }

            resultReal.append(stemDataReal)
            resultImag.append(stemDataImag)
        }

        return (resultReal, resultImag)
    }

    /// Pad or trim spectrogram to match model's expected time frame count.
    ///
    /// ## Why Reflect Padding (Not Zero Padding)?
    /// The model normalizes input by computing mean and std across the spectrogram.
    /// Zero padding would skew these statistics, causing incorrect normalization.
    /// Reflect padding preserves the statistical properties of the signal.
    ///
    /// ## Reflect Padding Algorithm
    /// For a sequence `[a, b, c, d, e]` that needs 3 more elements:
    /// - Mirror from the end: `[d, c, b]`
    /// - Result: `[a, b, c, d, e, d, c, b]`
    ///
    /// This is equivalent to NumPy's `np.pad(x, pad_width, mode='reflect')`.
    ///
    /// - Parameters:
    ///   - real: Real component spectrogram
    ///   - imag: Imaginary component spectrogram
    ///   - targetFrames: Model's expected number of time frames
    /// - Returns: Padded or trimmed spectrograms matching target size
    private func padOrTrimSpectrogram(
        real: [[[Float]]],
        imag: [[[Float]]],
        targetFrames: Int
    ) -> (real: [[[Float]]], imag: [[[Float]]]) {
        let currentFrames = real[0][0].count

        if currentFrames == targetFrames {
            return (real, imag)
        }

        var paddedReal: [[[Float]]] = []
        var paddedImag: [[[Float]]] = []

        for channelIdx in 0..<2 {
            var channelRealData: [[Float]] = []
            var channelImagData: [[Float]] = []

            for binIdx in 0..<2049 {
                var binRealData = real[channelIdx][binIdx]
                var binImagData = imag[channelIdx][binIdx]

                if currentFrames < targetFrames {
                    // REFLECT PADDING (not zeros!) to preserve statistics
                    // This is critical because the model normalizes by mean/std
                    let padNeeded = targetFrames - currentFrames
                    var realPadding = [Float]()
                    var imagPadding = [Float]()

                    for i in 0..<padNeeded {
                        // Reflect from end: use indices n-2, n-3, n-4, ...
                        // wrapping around if needed
                        let reflectIdx = currentFrames - 2 - (i % max(1, currentFrames - 1))
                        let safeIdx = max(0, min(reflectIdx, currentFrames - 1))
                        realPadding.append(binRealData[safeIdx])
                        imagPadding.append(binImagData[safeIdx])
                    }

                    binRealData.append(contentsOf: realPadding)
                    binImagData.append(contentsOf: imagPadding)
                } else {
                    // Trim
                    binRealData = Array(binRealData.prefix(targetFrames))
                    binImagData = Array(binImagData.prefix(targetFrames))
                }

                channelRealData.append(binRealData)
                channelImagData.append(binImagData)
            }

            paddedReal.append(channelRealData)
            paddedImag.append(channelImagData)
        }

        return (paddedReal, paddedImag)
    }

    /// Trim masks to target number of frames
    private func trimSpectrograms(
        real: [[[[Float]]]],
        imag: [[[[Float]]]],
        targetFrames: Int
    ) -> (real: [[[[Float]]]], imag: [[[[Float]]]]) {
        var trimmedReal: [[[[Float]]]] = []
        var trimmedImag: [[[[Float]]]] = []

        for stemIdx in 0..<real.count {
            var stemDataReal: [[[Float]]] = []
            var stemDataImag: [[[Float]]] = []

            for channelIdx in 0..<real[stemIdx].count {
                var channelDataReal: [[Float]] = []
                var channelDataImag: [[Float]] = []

                for binIdx in 0..<real[stemIdx][channelIdx].count {
                    let binDataReal = Array(real[stemIdx][channelIdx][binIdx].prefix(targetFrames))
                    let binDataImag = Array(imag[stemIdx][channelIdx][binIdx].prefix(targetFrames))
                    channelDataReal.append(binDataReal)
                    channelDataImag.append(binDataImag)
                }

                stemDataReal.append(channelDataReal)
                stemDataImag.append(channelDataImag)
            }

            trimmedReal.append(stemDataReal)
            trimmedImag.append(stemDataImag)
        }

        return (trimmedReal, trimmedImag)
    }

    // MARK: - Hybrid Model Helpers

    /// Create audio MLMultiArray from raw audio
    /// - Parameters:
    ///   - audio: Raw audio [2 channels][samples]
    ///   - shape: Target shape [1, 2, samples]
    /// - Returns: MLMultiArray with audio data
    private func createAudioMLMultiArray(
        audio: [[Float]],
        shape: [Int]
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)

        var index = 0
        // Fill in row-major order: [batch=1][2 channels][samples]
        for channel in 0..<shape[1] {
            for sample in 0..<shape[2] {
                array[index] = NSNumber(value: audio[channel][sample])
                index += 1
            }
        }

        return array
    }

    /// Pad or trim audio to target number of samples
    private func padOrTrimAudio(
        audio: [[Float]],
        targetSamples: Int
    ) -> [[Float]] {
        let currentSamples = audio[0].count

        if currentSamples == targetSamples {
            return audio
        }

        var paddedAudio: [[Float]] = []

        for channelIdx in 0..<2 {
            var channelData = audio[channelIdx]

            if currentSamples < targetSamples {
                // Reflect padding (NumPy-compatible: doesn't include edge)
                // For [a, b, c, d, e], reflect at end gives: [d, c, b, a, b, c, ...]
                let padNeeded = targetSamples - currentSamples
                var padding = [Float]()

                for i in 0..<padNeeded {
                    // NumPy reflect mode: starts at n-2, wraps around interior
                    let period = max(1, currentSamples - 1)
                    let reflectIdx = currentSamples - 2 - (i % period)
                    let safeIdx = max(0, min(reflectIdx, currentSamples - 1))
                    padding.append(channelData[safeIdx])
                }

                channelData.append(contentsOf: padding)
            } else {
                // Trim
                channelData = Array(channelData.prefix(targetSamples))
            }

            paddedAudio.append(channelData)
        }

        return paddedAudio
    }

    /// Convert frequency output MLMultiArray to spectrograms
    /// - Parameters:
    ///   - mlArray: Output MLMultiArray [1, 6 stems, 4 channels, 2048 bins, timeFrames]
    ///   - timeFrames: Number of time frames
    /// - Returns: Separated spectrograms (real, imag) [6 stems][2 channels][2049 bins][timeFrames]
    private func convertFreqOutputToSpectrograms(
        _ mlArray: MLMultiArray,
        timeFrames: Int
    ) throws -> (real: [[[[Float]]]], imag: [[[[Float]]]]) {
        let numStems = 6
        let numChannels = 2
        let numBins = 2049  // Pad 2048 → 2049 for iSTFT

        var resultReal: [[[[Float]]]] = []
        var resultImag: [[[[Float]]]] = []

        for stemIdx in 0..<numStems {
            var stemDataReal: [[[Float]]] = []
            var stemDataImag: [[[Float]]] = []

            for channelIdx in 0..<numChannels {
                var channelDataReal: [[Float]] = []
                var channelDataImag: [[Float]] = []

                for binIdx in 0..<numBins {
                    var binDataReal: [Float] = []
                    var binDataImag: [Float] = []

                    for timeIdx in 0..<timeFrames {
                        if binIdx < freqBins {
                            // Channels in model: [L_real, L_imag, R_real, R_imag]
                            let realChannelIdx = channelIdx == 0 ? 0 : 2
                            let imagChannelIdx = channelIdx == 0 ? 1 : 3

                            // Row-major index: [batch][stem][channel][bin][time]
                            let realIndex = (((stemIdx * 4 + realChannelIdx) * freqBins) + binIdx) * timeFrames + timeIdx
                            let imagIndex = (((stemIdx * 4 + imagChannelIdx) * freqBins) + binIdx) * timeFrames + timeIdx

                            binDataReal.append(mlArray[realIndex].floatValue)
                            binDataImag.append(mlArray[imagIndex].floatValue)
                        } else {
                            // Pad with 0 for bin 2048 (Nyquist)
                            binDataReal.append(0.0)
                            binDataImag.append(0.0)
                        }
                    }

                    channelDataReal.append(binDataReal)
                    channelDataImag.append(binDataImag)
                }

                stemDataReal.append(channelDataReal)
                stemDataImag.append(channelDataImag)
            }

            resultReal.append(stemDataReal)
            resultImag.append(stemDataImag)
        }

        return (resultReal, resultImag)
    }

    /// Convert time output MLMultiArray to audio
    /// - Parameters:
    ///   - mlArray: Output MLMultiArray [1, 6 stems, 2 channels, samples]
    ///   - samples: Number of audio samples
    /// - Returns: Separated audio [6 stems][2 channels][samples]
    private func convertTimeOutputToAudio(
        _ mlArray: MLMultiArray,
        samples: Int
    ) throws -> [[[Float]]] {
        let numStems = 6
        let numChannels = 2

        var result: [[[Float]]] = []

        for stemIdx in 0..<numStems {
            var stemData: [[Float]] = []

            for channelIdx in 0..<numChannels {
                var channelData: [Float] = []

                for sampleIdx in 0..<samples {
                    // Row-major index: [batch][stem][channel][sample]
                    let index = ((stemIdx * numChannels + channelIdx) * samples) + sampleIdx
                    channelData.append(mlArray[index].floatValue)
                }

                stemData.append(channelData)
            }

            result.append(stemData)
        }

        return result
    }

    /// Trim audio to target number of samples
    private func trimAudio(
        audio: [[[Float]]],
        targetSamples: Int
    ) -> [[[Float]]] {
        return audio.map { stem in
            stem.map { channel in
                Array(channel.prefix(targetSamples))
            }
        }
    }
}

// MARK: - Error Types

public enum InferenceError: Error, LocalizedError {
    case invalidInputShape(String)
    case invalidOutput(String)
    case predictionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidInputShape(let msg):
            return "Invalid input shape: \(msg)"
        case .invalidOutput(let msg):
            return "Invalid output: \(msg)"
        case .predictionFailed(let msg):
            return "Prediction failed: \(msg)"
        }
    }
}
