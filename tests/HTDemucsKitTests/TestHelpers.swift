import Foundation

/// Custom error for skipping tests when required resources are not available
struct SkipTest: Error {
    let message: String
    init(_ message: String) {
        self.message = message
    }
}
