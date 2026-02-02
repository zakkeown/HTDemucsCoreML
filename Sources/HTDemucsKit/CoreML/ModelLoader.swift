import CoreML
import Foundation

/// Loads and manages CoreML models from .mlpackage files
public class ModelLoader {
    private let modelURL: URL
    private var model: MLModel?

    /// Initialize with model name from Resources/Models directory
    /// - Parameter modelName: Name of the model (default: "htdemucs_6s")
    /// - Throws: ModelError if model not found in Resources/Models
    public init(modelName: String = "htdemucs_6s") throws {
        let modelPath = try Self.resolveModelPath(modelName: modelName)
        self.modelURL = URL(fileURLWithPath: modelPath)
    }

    /// Initialize with explicit path to .mlpackage file
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

    // MARK: - Private Helpers

    /// Resolve model path from Resources/Models directory
    /// - Parameter modelName: Name of the model
    /// - Returns: Full path to the model
    /// - Throws: ModelError if model not found
    private static func resolveModelPath(modelName: String) throws -> String {
        // Find project root by searching for Package.swift
        var currentURL = URL(fileURLWithPath: #file)
        var projectRoot: URL?

        while currentURL.path != "/" {
            currentURL = currentURL.deletingLastPathComponent()
            let packagePath = currentURL.appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: packagePath.path) {
                projectRoot = currentURL
                break
            }
        }

        guard let root = projectRoot else {
            throw ModelError.modelNotFound("Could not find project root (Package.swift)")
        }

        // Build path to model in Resources/Models
        let modelFileName = modelName.hasSuffix(".mlpackage") ? modelName : "\(modelName).mlpackage"
        let modelPath = root
            .appendingPathComponent("Resources/Models")
            .appendingPathComponent(modelFileName)
            .path

        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw ModelError.modelNotFound("Model not found at: \(modelPath)")
        }

        return modelPath
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
