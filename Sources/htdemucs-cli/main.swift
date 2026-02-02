import Foundation
import HTDemucsKit
import AVFoundation

@main
struct HTDemucsCLI {
    static func main() async {
        let args = CommandLine.arguments

        guard args.count >= 2 else {
            printUsage()
            exit(1)
        }

        do {
            switch args[1] {
            case "stft":
                try await runSTFT(args: Array(args.dropFirst(2)))
            case "separate":
                try await runSeparate(args: Array(args.dropFirst(2)))
            case "validate":
                try runValidate()
            case "version":
                printVersion()
            default:
                print("Unknown command: \(args[1])")
                printUsage()
                exit(1)
            }
        } catch {
            print("Error: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - Commands

    static func runSTFT(args: [String]) async throws {
        guard args.count >= 2 else {
            print("Usage: htdemucs-cli stft <input.wav> --output <output.npz>")
            exit(1)
        }

        let inputPath = args[0]
        let _ = args.count >= 4 ? args[3] : "output.npz"

        print("Computing STFT for: \(inputPath)")
        print("Note: NPZ export not yet implemented")
        print("This requires Python bridge (Task 9)")

        // Placeholder for STFT computation
        let fft = try AudioFFT()
        print("✓ AudioFFT initialized (fftSize: \(fft.fftSize), hop: \(fft.hopLength))")
    }

    static func runSeparate(args: [String]) async throws {
        // Parse arguments
        var inputPath: String?
        var outputDir: String = "stems"
        var format: AudioFormat = .wav

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--output":
                if i + 1 < args.count {
                    outputDir = args[i + 1]
                    i += 2
                } else {
                    throw CLIError.missingValue("--output")
                }
            case "--format":
                if i + 1 < args.count {
                    let formatStr = args[i + 1].lowercased()
                    switch formatStr {
                    case "wav":
                        format = .wav
                    case "mp3":
                        format = .mp3
                    case "flac":
                        format = .flac
                    default:
                        throw CLIError.invalidFormat(formatStr)
                    }
                    i += 2
                } else {
                    throw CLIError.missingValue("--format")
                }
            default:
                if inputPath == nil {
                    inputPath = args[i]
                }
                i += 1
            }
        }

        guard let input = inputPath else {
            throw CLIError.missingArgument("input file")
        }

        // Verify input file exists
        guard FileManager.default.fileExists(atPath: input) else {
            throw CLIError.invalidPath(input)
        }

        print("HTDemucs Audio Separation")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("Input:  \(input)")
        print("Output: \(outputDir)/")
        print("Format: \(format.rawValue)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print()

        // Initialize coordinator (uses default htdemucs_6s model)
        print("Loading model...")
        let coordinator = try SeparationCoordinator()
        print("✓ Model loaded: htdemucs_6s")
        print()

        // Run separation with progress tracking
        let inputURL = URL(fileURLWithPath: input)
        let outputURL = URL(fileURLWithPath: outputDir)

        let progressStream = coordinator.separate(
            input: inputURL,
            outputDir: outputURL,
            format: format
        )

        // Track progress state
        var currentLine = ""

        for await event in progressStream {
            switch event {
            case .decoding(let progress):
                clearCurrentLine()
                currentLine = String(format: "Decoding audio... %3.0f%%", progress * 100)
                print(currentLine, terminator: "")
                fflush(stdout)

            case .processing(let chunk, let total):
                clearCurrentLine()
                let progressBar = makeProgressBar(current: chunk, total: total)
                currentLine = "Processing: \(progressBar) \(chunk)/\(total) chunks"
                print(currentLine, terminator: "")
                fflush(stdout)

            case .encoding(let stem, let progress):
                clearCurrentLine()
                let progressBar = makeProgressBar(progress: progress)
                currentLine = "Encoding \(stem.rawValue)... \(progressBar)"
                print(currentLine, terminator: "")
                fflush(stdout)

            case .complete(let outputPaths):
                clearCurrentLine()
                print("✓ Separation complete!")
                print()
                print("Generated stems:")
                for stemType in StemType.allCases.sorted(by: { $0.rawValue < $1.rawValue }) {
                    if let path = outputPaths[stemType] {
                        print("  ✓ \(stemType.rawValue.padding(toLength: 7, withPad: " ", startingAt: 0)) → \(path.path)")
                    }
                }
                print()

            case .failed(let error):
                clearCurrentLine()
                print("✗ Separation failed: \(error.localizedDescription)")
                throw error
            }
        }
    }

    // MARK: - Progress Display Helpers

    static func clearCurrentLine() {
        print("\r\u{001B}[K", terminator: "")
        fflush(stdout)
    }

    static func makeProgressBar(current: Int, total: Int) -> String {
        let progress = total > 0 ? Float(current) / Float(total) : 0.0
        return makeProgressBar(progress: progress)
    }

    static func makeProgressBar(progress: Float) -> String {
        let barWidth = 20
        let filled = Int(progress * Float(barWidth))
        let empty = barWidth - filled
        let percent = Int(progress * 100)

        let bar = String(repeating: "█", count: filled) + String(repeating: "░", count: empty)
        return "[\(bar)] \(String(format: "%3d%%", percent))"
    }

    static func runValidate() throws {
        print("HTDemucs Test Suite")
        print("Note: Run tests with: swift test")
        print()

        // Could integrate test runner here
        print("Available test suites:")
        print("  - AudioFFTTests (STFT/iSTFT)")
        print("  - RoundTripTests (reconstruction)")
        print("  - STFTPropertyTests (Parseval, COLA, symmetry)")
        print("  - EdgeCaseTests (edge cases)")
        print("  - ModelLoaderTests (CoreML loading)")
        print("  - InferenceEngineTests (CoreML inference)")
        print("  - ChunkProcessorTests (chunking)")
        print("  - SeparationPipelineTests (integration)")
    }

    static func printVersion() {
        print("HTDemucs CLI v0.3.0")
        print("Phase 3: Complete Audio I/O Pipeline")
    }

    static func printUsage() {
        print("""
        HTDemucs CLI - Audio Source Separation

        Commands:
          separate <input> [--output <dir>] [--format <format>]
              Separate audio into 6 stems (drums, bass, vocals, other, piano, guitar)
              Options:
                --output <dir>      Output directory (default: stems/)
                --format <format>   Output format: wav, mp3, flac (default: wav)

          stft <input.wav> --output <stft.npz>
              Compute STFT for PyTorch validation

          validate
              Display test suite information

          version
              Show version information

        Examples:
          htdemucs-cli separate song.wav --output my_stems/
          htdemucs-cli separate song.mp3 --output stems/ --format flac
          htdemucs-cli stft song.wav --output stft.npz
          htdemucs-cli validate
        """)
    }
}

// MARK: - Error Types

enum CLIError: Error, LocalizedError {
    case missingArgument(String)
    case missingValue(String)
    case invalidPath(String)
    case invalidFormat(String)

    var errorDescription: String? {
        switch self {
        case .missingArgument(let arg):
            return "Missing required argument: \(arg)"
        case .missingValue(let flag):
            return "Missing value for flag: \(flag)"
        case .invalidPath(let path):
            return "Invalid path: \(path)"
        case .invalidFormat(let format):
            return "Invalid format: \(format). Supported formats: wav, mp3, flac"
        }
    }
}
