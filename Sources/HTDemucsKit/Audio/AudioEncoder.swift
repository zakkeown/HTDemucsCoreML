import Foundation
import SwiftFFmpeg

/// Encodes PCM float audio data to various output formats using FFmpeg
public class AudioEncoder {
    public init() {}

    /// Encode PCM float audio to a file
    /// - Parameters:
    ///   - leftChannel: Left channel PCM float samples
    ///   - rightChannel: Right channel PCM float samples
    ///   - sampleRate: Sample rate in Hz
    ///   - format: Output audio format (WAV, MP3, FLAC)
    ///   - destination: Destination file URL
    /// - Throws: AudioError if encoding fails
    public func encode(
        leftChannel: [Float],
        rightChannel: [Float],
        sampleRate: Int,
        format: AudioFormat,
        destination: URL
    ) throws {
        guard leftChannel.count == rightChannel.count else {
            throw AudioError.encodeFailed(
                stem: .other,
                reason: "Channel lengths must match"
            )
        }

        let filePath = destination.path

        // Create output directory if it doesn't exist
        let directory = destination.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: nil
            )
        } catch {
            throw AudioError.encodeFailed(
                stem: .other,
                reason: "Failed to create output directory: \(error.localizedDescription)"
            )
        }

        do {
            // Determine codec ID based on format
            let codecId: AVCodecID
            switch format {
            case .wav:
                codecId = .PCM_S16LE  // 16-bit PCM for WAV
            case .mp3:
                codecId = .MP3
            case .flac:
                codecId = .FLAC
            }

            // Find encoder
            guard let codec = AVCodec.findEncoderById(codecId) else {
                throw AudioError.unsupportedFormat(
                    format: format.rawValue,
                    reason: "Encoder not found for codec ID"
                )
            }

            // Create format context for output
            let formatContext = try AVFormatContext(
                format: nil,
                formatName: format.rawValue,
                filename: filePath
            )

            // Add audio stream
            guard let stream = formatContext.addStream(codec: codec) else {
                throw AudioError.encodeFailed(
                    stem: .other,
                    reason: "Failed to create audio stream"
                )
            }

            // Create and configure codec context
            let codecContext = AVCodecContext(codec: codec)
            codecContext.sampleRate = sampleRate
            codecContext.channelLayout = AVChannelLayoutStereo
            codecContext.sampleFormat = codec.supportedSampleFormats?.first ?? .int16

            // Set bitrate for compressed formats
            if format == .mp3 {
                codecContext.bitRate = 192_000  // 192 kbps
            }

            // Open the codec
            try codecContext.openCodec()

            // Copy codec parameters to stream
            stream.codecParameters.copy(from: codecContext)

            // Set stream time base
            stream.timebase = AVRational(num: 1, den: Int32(sampleRate))

            // Open output file
            try formatContext.openOutput(url: filePath, flags: .write)

            // Write header
            try formatContext.writeHeader()

            // Encode audio
            try encodeAudio(
                formatContext: formatContext,
                codecContext: codecContext,
                stream: stream,
                leftChannel: leftChannel,
                rightChannel: rightChannel
            )

            // Write trailer and close
            try formatContext.writeTrailer()

        } catch let error as AudioError {
            throw error
        } catch {
            throw AudioError.encodeFailed(
                stem: .other,
                reason: "Encoding failed: \(error.localizedDescription)"
            )
        }
    }

    /// Encode audio samples and write to format context
    private func encodeAudio(
        formatContext: AVFormatContext,
        codecContext: AVCodecContext,
        stream: AVStream,
        leftChannel: [Float],
        rightChannel: [Float]
    ) throws {
        let frameSize = Int(codecContext.frameSize > 0 ? codecContext.frameSize : 1024)
        let sampleCount = leftChannel.count

        var pts: Int64 = 0
        var offset = 0

        while offset < sampleCount {
            let frame = AVFrame()
            let currentFrameSize = min(frameSize, sampleCount - offset)

            frame.sampleCount = currentFrameSize
            frame.sampleFormat = codecContext.sampleFormat
            frame.channelLayout = codecContext.channelLayout
            frame.sampleRate = codecContext.sampleRate
            frame.pts = pts

            // Allocate frame buffers
            try frame.allocBuffer()

            // Fill frame with audio data
            try fillFrame(
                frame: frame,
                leftChannel: Array(leftChannel[offset..<offset + currentFrameSize]),
                rightChannel: Array(rightChannel[offset..<offset + currentFrameSize]),
                sampleFormat: codecContext.sampleFormat
            )

            // Send frame for encoding
            try codecContext.sendFrame(frame)

            // Receive and write packets
            try receivePackets(
                codecContext: codecContext,
                formatContext: formatContext,
                stream: stream
            )

            frame.unref()

            offset += currentFrameSize
            pts += Int64(currentFrameSize)
        }

        // Flush encoder
        try codecContext.sendFrame(nil)
        try receivePackets(
            codecContext: codecContext,
            formatContext: formatContext,
            stream: stream
        )
    }

    /// Fill an AVFrame with audio samples
    private func fillFrame(
        frame: AVFrame,
        leftChannel: [Float],
        rightChannel: [Float],
        sampleFormat: AVSampleFormat
    ) throws {
        let sampleCount = leftChannel.count

        switch sampleFormat {
        case .int16:
            // Interleaved 16-bit PCM
            if let audioData = frame.extendedData[0] {
                audioData.withMemoryRebound(to: Int16.self, capacity: sampleCount * 2) { ptr in
                    for i in 0..<sampleCount {
                        ptr[i * 2] = Int16(max(-1.0, min(1.0, leftChannel[i])) * 32767.0)
                        ptr[i * 2 + 1] = Int16(max(-1.0, min(1.0, rightChannel[i])) * 32767.0)
                    }
                }
            }

        case .int16Planar:
            // Planar 16-bit PCM
            if let leftData = frame.extendedData[0] {
                leftData.withMemoryRebound(to: Int16.self, capacity: sampleCount) { ptr in
                    for i in 0..<sampleCount {
                        ptr[i] = Int16(max(-1.0, min(1.0, leftChannel[i])) * 32767.0)
                    }
                }
            }
            if let rightData = frame.extendedData[1] {
                rightData.withMemoryRebound(to: Int16.self, capacity: sampleCount) { ptr in
                    for i in 0..<sampleCount {
                        ptr[i] = Int16(max(-1.0, min(1.0, rightChannel[i])) * 32767.0)
                    }
                }
            }

        case .float:
            // Interleaved float
            if let audioData = frame.extendedData[0] {
                audioData.withMemoryRebound(to: Float.self, capacity: sampleCount * 2) { ptr in
                    for i in 0..<sampleCount {
                        ptr[i * 2] = leftChannel[i]
                        ptr[i * 2 + 1] = rightChannel[i]
                    }
                }
            }

        case .floatPlanar:
            // Planar float
            if let leftData = frame.extendedData[0] {
                leftData.withMemoryRebound(to: Float.self, capacity: sampleCount) { ptr in
                    for i in 0..<sampleCount {
                        ptr[i] = leftChannel[i]
                    }
                }
            }
            if let rightData = frame.extendedData[1] {
                rightData.withMemoryRebound(to: Float.self, capacity: sampleCount) { ptr in
                    for i in 0..<sampleCount {
                        ptr[i] = rightChannel[i]
                    }
                }
            }

        default:
            throw AudioError.unsupportedFormat(
                format: String(describing: sampleFormat),
                reason: "Sample format not supported for encoding"
            )
        }
    }

    /// Receive encoded packets and write to format context
    private func receivePackets(
        codecContext: AVCodecContext,
        formatContext: AVFormatContext,
        stream: AVStream
    ) throws {
        let packet = AVPacket()

        while true {
            do {
                try codecContext.receivePacket(packet)

                // Set packet stream index
                packet.streamIndex = stream.index

                // Write packet
                try formatContext.interleavedWriteFrame(packet)

                packet.unref()

            } catch let err as AVError where err == .tryAgain || err == .eof {
                break
            }
        }
    }
}
