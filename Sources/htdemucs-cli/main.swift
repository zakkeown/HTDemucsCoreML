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
        let outputPath = args.count >= 4 ? args[3] : "output.npz"

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
        var modelPath: String?
        var outputDir: String = "stems"

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--model":
                if i + 1 < args.count {
                    modelPath = args[i + 1]
                    i += 2
                } else {
                    throw CLIError.missingValue("--model")
                }
            case "--output":
                if i + 1 < args.count {
                    outputDir = args[i + 1]
                    i += 2
                } else {
                    throw CLIError.missingValue("--output")
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

        guard let model = modelPath else {
            throw CLIError.missingArgument("--model <path>")
        }

        print("HTDemucs Audio Separation")
        print("Input: \(input)")
        print("Model: \(model)")
        print("Output: \(outputDir)")
        print()

        // Create output directory
        try FileManager.default.createDirectory(
            atPath: outputDir,
            withIntermediateDirectories: true
        )

        // Initialize pipeline
        print("Loading model...")
        let pipeline = try SeparationPipeline(modelPath: model)
        print("✓ Model loaded")

        // Note: Actual audio I/O requires AVFoundation integration
        print()
        print("Note: Audio file I/O not yet implemented")
        print("This requires AVFoundation integration")
        print()
        print("Pipeline ready! Would process:")
        print("  1. Load WAV from: \(input)")
        print("  2. Separate into 6 stems")
        print("  3. Save stems to: \(outputDir)/")
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
        print("HTDemucs CLI v0.1.0")
        print("Phase 2B: Swift STFT/iSTFT + CoreML Integration")
    }

    static func printUsage() {
        print("""
        HTDemucs CLI - Audio Source Separation

        Commands:
          stft <input.wav> --output <stft.npz>
              Compute STFT for PyTorch validation

          separate <input.wav> --model <model.mlpackage> --output <dir/>
              Separate audio into 6 stems (drums, bass, vocals, other, piano, guitar)

          validate
              Display test suite information

          version
              Show version information

        Examples:
          htdemucs-cli separate song.wav --model htdemucs.mlpackage --output stems/
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

    var errorDescription: String? {
        switch self {
        case .missingArgument(let arg):
            return "Missing required argument: \(arg)"
        case .missingValue(let flag):
            return "Missing value for flag: \(flag)"
        case .invalidPath(let path):
            return "Invalid path: \(path)"
        }
    }
}
