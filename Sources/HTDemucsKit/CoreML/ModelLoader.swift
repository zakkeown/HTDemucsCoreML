import CoreML
import Foundation

/// Loads and manages CoreML models from .mlpackage files
public class ModelLoader {
    private let modelURL: URL
    private var model: MLModel?

    /// Initialize with path to .mlpackage file
    /// - Parameter modelPath: Path to the CoreML model package
    /// - Throws: ModelError if file doesn't exist
    public init(modelPath: String) throws {
        self.modelURL = URL(fileURLWithPath: modelPath)

        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw ModelError.modelNotFound(modelPath)
        }
    }

    /// Load the CoreML model with configuration
    /// - Returns: Loaded MLModel instance
    /// - Throws: ModelError if loading fails
    public func load() throws -> MLModel {
        // Return cached model if already loaded
        if let cached = model {
            return cached
        }

        // Configure for CPU+GPU (Phase 1 target)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        // Load model
        let loadedModel = try MLModel(contentsOf: modelURL, configuration: config)
        self.model = loadedModel
        return loadedModel
    }
}

// MARK: - Error Types

public enum ModelError: Error, LocalizedError {
    case modelNotFound(String)
    case loadFailed(String)
    case incompatibleVersion(model: String, required: String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Model not found at path: \(path)"
        case .loadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .incompatibleVersion(let model, let required):
            return "Model '\(model)' is incompatible (required: \(required))"
        }
    }
}
