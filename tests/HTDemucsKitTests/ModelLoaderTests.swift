import Testing
import Foundation
@testable import HTDemucsKit

@Suite("ModelLoader Tests")
struct ModelLoaderTests {
    @Test("Load model from explicit path")
    func testLoadFromExplicitPath() throws {
        // This test requires a model file to be present
        let modelPath = try resolveModelPath("htdemucs_6s.mlpackage")

        let loader = try ModelLoader(modelPath: modelPath)
        let model = try loader.load()

        // Just verify we got a model back without error
        _ = model
    }

    @Test("Load model from Resources/Models directory")
    func testLoadFromResourcesDirectory() throws {
        // This test skips if model not present
        let loader = try ModelLoader(modelName: "htdemucs_6s")
        let model = try loader.load()

        // Just verify we got a model back without error
        _ = model
    }

    @Test("Load model with default name")
    func testLoadWithDefaultName() throws {
        // This test skips if model not present
        let loader = try ModelLoader()
        let model = try loader.load()

        // Just verify we got a model back without error
        _ = model
    }

    @Test("Throw error for non-existent model path")
    func testThrowsForNonExistentPath() {
        #expect(throws: ModelError.self) {
            _ = try ModelLoader(modelPath: "/tmp/nonexistent.mlpackage")
        }
    }

    @Test("Throw error for non-existent model name")
    func testThrowsForNonExistentName() {
        #expect(throws: ModelError.self) {
            _ = try ModelLoader(modelName: "nonexistent_model")
        }
    }

    @Test("Cache loaded model")
    func testCacheLoadedModel() throws {
        let modelPath = try resolveModelPath("htdemucs_6s.mlpackage")

        let loader = try ModelLoader(modelPath: modelPath)
        let model1 = try loader.load()
        let model2 = try loader.load()

        // Both should return the same cached instance
        #expect(model1 === model2)
    }

    // Helper to resolve model path
    private func resolveModelPath(_ name: String) throws -> String {
        var projectRoot = URL(fileURLWithPath: #file)
        while projectRoot.path != "/" {
            projectRoot = projectRoot.deletingLastPathComponent()
            let packagePath = projectRoot.appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: packagePath.path) {
                break
            }
        }

        let modelPath = projectRoot
            .appendingPathComponent("Resources/Models")
            .appendingPathComponent(name)
            .path

        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw SkipTest("Model not found: \(modelPath)")
        }

        return modelPath
    }
}
