import Foundation

/// Represents decoded audio data in PCM float format
public struct DecodedAudio {
    public let leftChannel: [Float]
    public let rightChannel: [Float]
    public let sampleRate: Double
    public let duration: Double

    public init(leftChannel: [Float], rightChannel: [Float], sampleRate: Double, duration: Double) {
        self.leftChannel = leftChannel
        self.rightChannel = rightChannel
        self.sampleRate = sampleRate
        self.duration = duration
    }

    /// Number of audio channels (always 2 for stereo)
    public var channelCount: Int { 2 }

    /// Number of frames per channel
    public var frameCount: Int { leftChannel.count }

    /// Convert to [[left samples], [right samples]] format
    public var stereoArray: [[Float]] {
        [leftChannel, rightChannel]
    }
}

/// Supported audio output formats
public enum AudioFormat: String {
    case wav
    case mp3
    case flac

    public var fileExtension: String { rawValue }
}
