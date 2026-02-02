import Foundation

/// Custom error for skipping tests when required resources are not available
struct SkipTest: Error {
    let message: String
    init(_ message: String) {
        self.message = message
    }
}

/// Error for test helper functions
enum TestHelperError: Error {
    case fixtureNotFound(String)
}

/// Resolve path to test fixture in Resources/TestAudio/
func resolveFixturePath(_ name: String) throws -> String {
    var projectRoot = URL(fileURLWithPath: #filePath)
    while projectRoot.path != "/" {
        projectRoot = projectRoot.deletingLastPathComponent()
        let packagePath = projectRoot.appendingPathComponent("Package.swift")
        if FileManager.default.fileExists(atPath: packagePath.path) {
            break
        }
    }

    let fixturePath = projectRoot
        .appendingPathComponent("Resources")
        .appendingPathComponent("TestAudio")
        .appendingPathComponent(name)

    guard FileManager.default.fileExists(atPath: fixturePath.path) else {
        throw TestHelperError.fixtureNotFound(name)
    }

    return fixturePath.path
}
