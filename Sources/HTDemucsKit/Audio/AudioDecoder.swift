import Foundation
import SwiftFFmpeg

/// Decodes audio files to PCM float format using FFmpeg
public class AudioDecoder: @unchecked Sendable {
    public init() {}

    /// Decode an audio file to PCM float format
    /// - Parameter fileURL: URL of the audio file to decode
    /// - Returns: DecodedAudio containing stereo PCM float data
    /// - Throws: AudioError if decoding fails
    public func decode(fileURL: URL) throws -> DecodedAudio {
        let filePath = fileURL.path

        // Check file exists
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw AudioError.fileNotFound(path: filePath)
        }

        do {
            // Open the input file
            let formatContext = try AVFormatContext(url: filePath)
            try formatContext.findStreamInfo()

            // Find the audio stream
            guard let audioStreamIndex = formatContext.findBestStream(type: .audio) else {
                throw AudioError.unsupportedFormat(
                    format: fileURL.pathExtension,
                    reason: "No audio stream found"
                )
            }

            let audioStream = formatContext.streams[audioStreamIndex]
            let codecParameters = audioStream.codecParameters

            // Find decoder for the audio stream
            guard let codec = AVCodec.findDecoderById(codecParameters.codecId) else {
                throw AudioError.unsupportedFormat(
                    format: fileURL.pathExtension,
                    reason: "Codec not found for \(codecParameters.codecId.name)"
                )
            }

            // Create codec context
            let codecContext = AVCodecContext(codec: codec)
            codecContext.setParameters(codecParameters)
            try codecContext.openCodec()

            // Get audio properties
            let sampleRate = Double(codecParameters.sampleRate)
            let channelCount = Int(codecParameters.channelLayout.nb_channels)

            // Calculate duration
            let duration: Double
            if audioStream.duration != AVTimestamp.noPTS {
                duration = Double(audioStream.duration) * audioStream.timebase.toDouble
            } else {
                duration = 0.0
            }

            // Decode all packets
            var leftSamples: [Float] = []
            var rightSamples: [Float] = []

            let packet = AVPacket()
            let frame = AVFrame()

            // Read frames from the file
            while let _ = try? formatContext.readFrame(into: packet) {
                // Only process packets from the audio stream
                guard packet.streamIndex == audioStreamIndex else {
                    packet.unref()
                    continue
                }

                // Decode this packet
                try decodePacket(
                    codecContext: codecContext,
                    packet: packet,
                    frame: frame,
                    channelCount: channelCount,
                    leftSamples: &leftSamples,
                    rightSamples: &rightSamples
                )

                packet.unref()
            }

            // Flush decoder (send nil packet)
            try decodePacket(
                codecContext: codecContext,
                packet: nil,
                frame: frame,
                channelCount: channelCount,
                leftSamples: &leftSamples,
                rightSamples: &rightSamples
            )

            // Ensure stereo output (duplicate mono to both channels if needed)
            if channelCount == 1 && !leftSamples.isEmpty {
                rightSamples = leftSamples
            } else if leftSamples.isEmpty && rightSamples.isEmpty {
                throw AudioError.decodeFailed(
                    underlyingError: NSError(
                        domain: "AudioDecoder",
                        code: -1,
                        userInfo: [NSLocalizedDescriptionKey: "No audio data decoded"]
                    )
                )
            }

            // Calculate actual duration if not available from metadata
            let actualDuration = duration > 0 ? duration : Double(leftSamples.count) / sampleRate

            return DecodedAudio(
                leftChannel: leftSamples,
                rightChannel: rightSamples,
                sampleRate: sampleRate,
                duration: actualDuration
            )

        } catch let error as AudioError {
            throw error
        } catch {
            throw AudioError.decodeFailed(underlyingError: error)
        }
    }

    /// Decode a packet and append samples to the arrays
    private func decodePacket(
        codecContext: AVCodecContext,
        packet: AVPacket?,
        frame: AVFrame,
        channelCount: Int,
        leftSamples: inout [Float],
        rightSamples: inout [Float]
    ) throws {
        try codecContext.sendPacket(packet)

        while true {
            do {
                try codecContext.receiveFrame(frame)

                // Convert frame data to PCM float
                let (left, right) = try convertFrameToFloat(frame: frame, channelCount: channelCount)
                leftSamples.append(contentsOf: left)
                rightSamples.append(contentsOf: right)

                frame.unref()

            } catch let err as AVError where err == .tryAgain || err == .eof {
                break
            }
        }
    }

    /// Convert AVFrame to Float PCM samples
    private func convertFrameToFloat(frame: AVFrame, channelCount: Int) throws -> ([Float], [Float]) {
        let sampleCount = frame.sampleCount
        guard sampleCount > 0 else {
            return ([], [])
        }

        var leftChannel: [Float] = []
        var rightChannel: [Float] = []

        // Get sample format
        let format = frame.sampleFormat

        // SwiftFFmpeg provides access to frame data through data/extendedData buffers
        // The format determines how to interpret the data

        switch format {
        case .floatPlanar: // Planar float (separate channels)
            // Left channel is in extendedData[0]
            if let leftData = frame.extendedData[0] {
                leftChannel = leftData.withMemoryRebound(to: Float.self, capacity: sampleCount) { ptr in
                    Array(UnsafeBufferPointer(start: ptr, count: sampleCount))
                }
            }

            // Right channel is in extendedData[1] for stereo
            if channelCount >= 2, let rightData = frame.extendedData[1] {
                rightChannel = rightData.withMemoryRebound(to: Float.self, capacity: sampleCount) { ptr in
                    Array(UnsafeBufferPointer(start: ptr, count: sampleCount))
                }
            } else {
                rightChannel = leftChannel // Mono
            }

        case .float: // Interleaved float
            if let audioData = frame.extendedData[0] {
                audioData.withMemoryRebound(to: Float.self, capacity: sampleCount * channelCount) { ptr in
                    let buffer = UnsafeBufferPointer(start: ptr, count: sampleCount * channelCount)

                    // Deinterleave channels
                    for i in 0..<sampleCount {
                        leftChannel.append(buffer[i * channelCount])
                        if channelCount >= 2 {
                            rightChannel.append(buffer[i * channelCount + 1])
                        }
                    }
                }

                if channelCount == 1 {
                    rightChannel = leftChannel
                }
            }

        case .int16Planar: // Planar 16-bit signed integer
            // Left channel
            if let leftData = frame.extendedData[0] {
                leftChannel = leftData.withMemoryRebound(to: Int16.self, capacity: sampleCount) { ptr in
                    let buffer = UnsafeBufferPointer(start: ptr, count: sampleCount)
                    return buffer.map { Float($0) / 32768.0 }
                }
            }

            // Right channel
            if channelCount >= 2, let rightData = frame.extendedData[1] {
                rightChannel = rightData.withMemoryRebound(to: Int16.self, capacity: sampleCount) { ptr in
                    let buffer = UnsafeBufferPointer(start: ptr, count: sampleCount)
                    return buffer.map { Float($0) / 32768.0 }
                }
            } else {
                rightChannel = leftChannel
            }

        case .int16: // Interleaved 16-bit signed integer
            if let audioData = frame.extendedData[0] {
                audioData.withMemoryRebound(to: Int16.self, capacity: sampleCount * channelCount) { ptr in
                    let buffer = UnsafeBufferPointer(start: ptr, count: sampleCount * channelCount)

                    for i in 0..<sampleCount {
                        leftChannel.append(Float(buffer[i * channelCount]) / 32768.0)
                        if channelCount >= 2 {
                            rightChannel.append(Float(buffer[i * channelCount + 1]) / 32768.0)
                        }
                    }
                }

                if channelCount == 1 {
                    rightChannel = leftChannel
                }
            }

        default:
            // For unsupported formats, throw an error
            throw AudioError.unsupportedFormat(
                format: String(describing: format),
                reason: "Sample format not yet supported, needs resampling"
            )
        }

        return (leftChannel, rightChannel)
    }
}
