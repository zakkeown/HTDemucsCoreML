import XCTest
@testable import HTDemucsKit

final class AudioErrorsTests: XCTestCase {
    func testFileNotFoundErrorDescription() {
        let error = AudioError.fileNotFound(path: "/test/file.mp3")
        XCTAssertTrue(error.localizedDescription.contains("/test/file.mp3"))
    }

    func testUnsupportedFormatErrorDescription() {
        let error = AudioError.unsupportedFormat(format: "DRM-MP3", reason: "DRM protected")
        XCTAssertTrue(error.localizedDescription.contains("DRM-MP3"))
        XCTAssertTrue(error.localizedDescription.contains("DRM protected"))
    }

    func testDecodeFailedErrorDescription() {
        let underlying = NSError(domain: "test", code: 1, userInfo: nil)
        let error = AudioError.decodeFailed(underlyingError: underlying)
        XCTAssertTrue(error.localizedDescription.contains("decode"))
    }

    func testEncodeFailedErrorDescription() {
        let error = AudioError.encodeFailed(stem: .drums, reason: "disk full")
        XCTAssertTrue(error.localizedDescription.contains("drums"))
        XCTAssertTrue(error.localizedDescription.contains("disk full"))
    }
}
