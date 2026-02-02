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

        let timeFrames = real[0][0].count

        // Convert to MLMultiArray [1, 2, 2049, timeFrames]
        let realInput = try createMLMultiArray(from: real, shape: [1, 2, 2049, timeFrames])
        let imagInput = try createMLMultiArray(from: imag, shape: [1, 2, 2049, timeFrames])

        // Create input features
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "spectrogram_real": MLFeatureValue(multiArray: realInput),
            "spectrogram_imag": MLFeatureValue(multiArray: imagInput)
        ])

        // Run prediction
        let output = try model.prediction(from: inputFeatures)

        // Extract masks
        guard let masksArray = output.featureValue(for: "masks")?.multiArrayValue else {
            throw InferenceError.invalidOutput("masks output not found")
        }

        // Convert back to Swift arrays [6][2][2049][timeFrames]
        return try convertToSwiftArray(masksArray, shape: [6, 2, 2049, timeFrames])
    }

    // MARK: - Private Helpers

    private func createMLMultiArray(from data: [[[Float]]], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)

        // Fill array in row-major order
        var index = 0
        for channel in 0..<shape[1] {
            for freq in 0..<shape[2] {
                for time in 0..<shape[3] {
                    array[index] = NSNumber(value: data[channel][freq][time])
                    index += 1
                }
            }
        }

        return array
    }

    private func convertToSwiftArray(_ mlArray: MLMultiArray, shape: [Int]) throws -> [[[[Float]]]] {
        var result = [[[[Float]]]]()

        var index = 0
        for _ in 0..<shape[0] {
            var stemData = [[[Float]]]()
            for _ in 0..<shape[1] {
                var channelData = [[Float]]()
                for _ in 0..<shape[2] {
                    var freqData = [Float]()
                    for _ in 0..<shape[3] {
                        freqData.append(mlArray[index].floatValue)
                        index += 1
                    }
                    channelData.append(freqData)
                }
                stemData.append(channelData)
            }
            result.append(stemData)
        }

        return result
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
