import XCTest
@testable import HTDemucsKit
import CoreML
import Foundation

final class ModelLoaderTests: XCTestCase {

    // MARK: - Initialization Tests

    func testInitWithInvalidPath() {
        // Test that initialization throws for non-existent path
        let invalidPath = "/tmp/nonexistent_model_\(UUID().uuidString).mlpackage"

        XCTAssertThrowsError(try ModelLoader(modelPath: invalidPath)) { error in
            guard let modelError = error as? ModelError else {
                XCTFail("Expected ModelError, got \(type(of: error))")
                return
            }

            if case .modelNotFound(let path) = modelError {
                XCTAssertEqual(path, invalidPath)
                XCTAssertTrue(modelError.errorDescription?.contains(invalidPath) ?? false)
            } else {
                XCTFail("Expected modelNotFound error, got \(modelError)")
            }
        }
    }

    func testInitWithValidPath() throws {
        // Create a temporary directory that exists
        let tempDir = FileManager.default.temporaryDirectory
        let validPath = tempDir.appendingPathComponent("test_model_\(UUID().uuidString).mlpackage")

        // Create the directory to make it a valid path
        try FileManager.default.createDirectory(at: validPath, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: validPath)
        }

        // Should not throw since path exists
        XCTAssertNoThrow(try ModelLoader(modelPath: validPath.path))
    }

    // MARK: - File Existence Tests

    func testErrorDescriptionForModelNotFound() {
        let testPath = "/test/path/model.mlpackage"
        let error = ModelError.modelNotFound(testPath)

        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains(testPath) ?? false)
        XCTAssertTrue(error.errorDescription?.contains("Model not found") ?? false)
    }

    func testErrorDescriptionForLoadFailed() {
        let testReason = "Invalid model format"
        let error = ModelError.loadFailed(testReason)

        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains(testReason) ?? false)
        XCTAssertTrue(error.errorDescription?.contains("Failed to load") ?? false)
    }

    // MARK: - Model Loading Tests (with mock)

    func testLoadThrowsForInvalidModel() throws {
        // Create a temporary directory that looks like a model but isn't valid
        let tempDir = FileManager.default.temporaryDirectory
        let modelPath = tempDir.appendingPathComponent("invalid_model_\(UUID().uuidString).mlpackage")

        try FileManager.default.createDirectory(at: modelPath, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: modelPath)
        }

        let loader = try ModelLoader(modelPath: modelPath.path)

        // Loading should throw because the directory isn't a valid .mlpackage
        XCTAssertThrowsError(try loader.load()) { error in
            // CoreML will throw its own error for invalid models
            // We just verify that an error is thrown
            XCTAssertNotNil(error)
        }
    }

    // MARK: - Path Validation Tests

    func testInitWithEmptyPath() {
        XCTAssertThrowsError(try ModelLoader(modelPath: "")) { error in
            guard let modelError = error as? ModelError else {
                XCTFail("Expected ModelError, got \(type(of: error))")
                return
            }

            if case .modelNotFound = modelError {
                // Expected error
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }

    func testInitWithRelativePath() throws {
        // Create a temporary directory with a relative path reference
        let tempDir = FileManager.default.temporaryDirectory
        let modelDir = tempDir.appendingPathComponent("test_models_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: modelDir)
        }

        // Change to temp directory and test relative path
        let currentDir = FileManager.default.currentDirectoryPath
        defer {
            FileManager.default.changeCurrentDirectoryPath(currentDir)
        }

        FileManager.default.changeCurrentDirectoryPath(tempDir.path)

        // Should work with relative path if file exists
        let relativePath = modelDir.lastPathComponent
        XCTAssertNoThrow(try ModelLoader(modelPath: relativePath))
    }

    // MARK: - Edge Cases

    func testMultipleInitializations() throws {
        // Test that we can create multiple loaders for the same path
        let tempDir = FileManager.default.temporaryDirectory
        let modelPath = tempDir.appendingPathComponent("shared_model_\(UUID().uuidString).mlpackage")

        try FileManager.default.createDirectory(at: modelPath, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: modelPath)
        }

        let loader1 = try ModelLoader(modelPath: modelPath.path)
        let loader2 = try ModelLoader(modelPath: modelPath.path)

        // Both should be valid instances
        XCTAssertNotNil(loader1)
        XCTAssertNotNil(loader2)
    }

    // MARK: - Configuration Tests

    func testModelLoaderUsesCorrectConfiguration() throws {
        // We can't test actual model loading without a real .mlpackage,
        // but we can verify that the ModelLoader is initialized correctly
        let tempDir = FileManager.default.temporaryDirectory
        let modelPath = tempDir.appendingPathComponent("config_test_\(UUID().uuidString).mlpackage")

        try FileManager.default.createDirectory(at: modelPath, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: modelPath)
        }

        let loader = try ModelLoader(modelPath: modelPath.path)

        // Verify the loader was created successfully
        // (Configuration will be tested when we have actual models)
        XCTAssertNotNil(loader)
    }
}
