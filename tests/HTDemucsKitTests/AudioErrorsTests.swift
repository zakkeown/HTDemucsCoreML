import Testing
import Foundation
@testable import HTDemucsKit

@Suite("AudioError Tests")
struct AudioErrorsTests {
    @Test("File not found error description")
    func testFileNotFoundErrorDescription() {
        let error = AudioError.fileNotFound(path: "/test/file.mp3")
        #expect(error.localizedDescription.contains("/test/file.mp3"))
    }

    @Test("Unsupported format error description")
    func testUnsupportedFormatErrorDescription() {
        let error = AudioError.unsupportedFormat(format: "DRM-MP3", reason: "DRM protected")
        #expect(error.localizedDescription.contains("DRM-MP3"))
        #expect(error.localizedDescription.contains("DRM protected"))
    }

    @Test("Decode failed error description")
    func testDecodeFailedErrorDescription() {
        let underlying = NSError(domain: "test", code: 1, userInfo: nil)
        let error = AudioError.decodeFailed(underlyingError: underlying)
        #expect(error.localizedDescription.contains("decode"))
    }

    @Test("Encode failed error description")
    func testEncodeFailedErrorDescription() {
        let error = AudioError.encodeFailed(stem: .drums, reason: "disk full")
        #expect(error.localizedDescription.contains("drums"))
        #expect(error.localizedDescription.contains("disk full"))
    }
}
