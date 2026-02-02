import CoreML
import Foundation

/// Runs CoreML inference on audio spectrograms
public class InferenceEngine {
    private let model: MLModel

    /// Initialize with loaded CoreML model
    /// - Parameter model: The MLModel instance from ModelLoader
    public init(model: MLModel) {
        self.model = model
    }

    /// Run inference on stereo spectrogram
    /// - Parameters:
    ///   - real: Real component [2][freqBins][timeFrames] (stereo, freq, time)
    ///   - imag: Imaginary component [2][freqBins][timeFrames]
    /// - Returns: Masks [6][2][freqBins][timeFrames] (6 stems, stereo, freq, time)
    /// - Throws: InferenceError if prediction fails
    public func predict(real: [[[Float]]], imag: [[[Float]]]) throws -> [[[[Float]]]] {
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

        // Convert output to masks [6][2][2049][modelTimeFrames]
        let masks = try convertOutputToMasks(outputArray, timeFrames: modelTimeFrames)

        // Trim masks back to original input size if we padded
        if inputTimeFrames < modelTimeFrames {
            return trimMasks(masks, targetFrames: inputTimeFrames)
        }

        return masks
    }

    // MARK: - Private Helpers

    /// Create combined MLMultiArray with interleaved real/imag channels
    /// - Parameters:
    ///   - real: Real component [2 channels][2049 bins][timeFrames]
    ///   - imag: Imaginary component [2 channels][2049 bins][timeFrames]
    ///   - shape: Target shape [1, 4, 2049, timeFrames]
    /// - Returns: MLMultiArray with real/imag interleaved
    private func createCombinedMLMultiArray(
        real: [[[Float]]],
        imag: [[[Float]]],
        shape: [Int]
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)

        // Fill array in row-major order: [batch=1][4 channels][2049 bins][timeFrames]
        // Channels are: [real_left, imag_left, real_right, imag_right]
        var index = 0

        // Channel 0: real_left
        for freq in 0..<shape[2] {
            for time in 0..<shape[3] {
                array[index] = NSNumber(value: real[0][freq][time])
                index += 1
            }
        }

        // Channel 1: imag_left
        for freq in 0..<shape[2] {
            for time in 0..<shape[3] {
                array[index] = NSNumber(value: imag[0][freq][time])
                index += 1
            }
        }

        // Channel 2: real_right
        for freq in 0..<shape[2] {
            for time in 0..<shape[3] {
                array[index] = NSNumber(value: real[1][freq][time])
                index += 1
            }
        }

        // Channel 3: imag_right
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
    private func convertOutputToMasks(_ mlArray: MLMultiArray, timeFrames: Int) throws -> [[[[Float]]]] {
        // Model output is [1, 6 stems, 4 channels, 2048 freq bins, timeFrames]
        // We need [6 stems][2 channels][2049 bins][timeFrames]

        let numStems = 6
        let numChannels = 2
        let numBins = 2049  // We need to pad 2048 → 2049

        var result: [[[[Float]]]] = []

        for stemIdx in 0..<numStems {
            var stemData: [[[Float]]] = []

            for channelIdx in 0..<numChannels {
                var channelData: [[Float]] = []

                for binIdx in 0..<numBins {
                    var binData: [Float] = []

                    for timeIdx in 0..<timeFrames {
                        // Calculate index in row-major order
                        // [batch=0][stem][real/imag channel pair][freq][time]
                        // For left channel: use real (channel 0) and imag (channel 1)
                        // For right channel: use real (channel 2) and imag (channel 3)

                        if binIdx < 2048 {
                            // Model output has 2048 bins, we need to read the appropriate one
                            let realChannelIdx = channelIdx == 0 ? 0 : 2
                            let imagChannelIdx = channelIdx == 0 ? 1 : 3

                            // For masks, we typically use magnitude: sqrt(real^2 + imag^2)
                            // But let's just use real component as mask for now
                            let index = (((stemIdx * 4 + realChannelIdx) * 2048) + binIdx) * timeFrames + timeIdx
                            binData.append(mlArray[index].floatValue)
                        } else {
                            // Pad with 0 for bin 2048 (DC/Nyquist)
                            binData.append(0.0)
                        }
                    }

                    channelData.append(binData)
                }

                stemData.append(channelData)
            }

            result.append(stemData)
        }

        return result
    }

    /// Pad or trim spectrogram to target number of frames
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
                    // Pad with zeros
                    binRealData.append(contentsOf: [Float](repeating: 0, count: targetFrames - currentFrames))
                    binImagData.append(contentsOf: [Float](repeating: 0, count: targetFrames - currentFrames))
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
    private func trimMasks(_ masks: [[[[Float]]]], targetFrames: Int) -> [[[[Float]]]] {
        var trimmedMasks: [[[[Float]]]] = []

        for stemIdx in 0..<masks.count {
            var stemData: [[[Float]]] = []

            for channelIdx in 0..<masks[stemIdx].count {
                var channelData: [[Float]] = []

                for binIdx in 0..<masks[stemIdx][channelIdx].count {
                    let binData = Array(masks[stemIdx][channelIdx][binIdx].prefix(targetFrames))
                    channelData.append(binData)
                }

                stemData.append(channelData)
            }

            trimmedMasks.append(stemData)
        }

        return trimmedMasks
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
