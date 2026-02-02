import Foundation

enum TestSignals {
    static let sampleRate: Float = 44100

    /// Generate a sine wave test signal
    static func sine(frequency: Float, duration: Float, sampleRate: Float = 44100) -> [Float] {
        let numSamples = Int(duration * sampleRate)
        return (0..<numSamples).map { i in
            let t = Float(i) / sampleRate
            return sin(2.0 * .pi * frequency * t)
        }
    }

    /// Generate silence (all zeros)
    static func silence(samples: Int) -> [Float] {
        return [Float](repeating: 0, count: samples)
    }

    /// Generate white noise
    static func whiteNoise(samples: Int) -> [Float] {
        return (0..<samples).map { _ in
            Float.random(in: -1.0...1.0)
        }
    }

    /// Load NumPy .npz fixture via Python bridge
    ///
    /// - Parameter name: Name of the fixture file (without .npz extension)
    /// - Returns: Tuple of (audio, stft_real, stft_imag)
    /// - Throws: Error if Python script fails or JSON parsing fails
    static func loadNPZFixture(name: String) throws -> (audio: [Float], real: [[Float]], imag: [[Float]]) {
        // Locate the .npz file in Resources/GoldenOutputs/
        // Find project root by searching upward for Package.swift
        var projectRoot = URL(fileURLWithPath: #file)
        while projectRoot.path != "/" {
            projectRoot = projectRoot.deletingLastPathComponent()
            if FileManager.default.fileExists(atPath: projectRoot.appendingPathComponent("Package.swift").path) {
                break
            }
        }

        let npzPath = projectRoot
            .appendingPathComponent("Resources")
            .appendingPathComponent("GoldenOutputs")
            .appendingPathComponent("\(name).npz")
            .path

        let scriptPath = projectRoot
            .appendingPathComponent("scripts")
            .appendingPathComponent("npz_to_json.py")
            .path

        // Verify files exist
        guard FileManager.default.fileExists(atPath: npzPath) else {
            throw NSError(
                domain: "TestSignals",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Fixture not found: \(npzPath)"]
            )
        }

        guard FileManager.default.fileExists(atPath: scriptPath) else {
            throw NSError(
                domain: "TestSignals",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Python script not found: \(scriptPath)"]
            )
        }

        // Run Python script to convert .npz to JSON
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", scriptPath, npzPath]

        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        try process.run()
        process.waitUntilExit()

        // Read output synchronously after process completes
        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()

        guard process.terminationStatus == 0 else {
            let errorMessage = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            throw NSError(
                domain: "TestSignals",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Python script failed: \(errorMessage)"]
            )
        }

        // Parse JSON output

        struct NPZData: Codable {
            let audio: [Float]
            let stft_real: [[Float]]
            let stft_imag: [[Float]]
        }

        let decoder = JSONDecoder()
        let data = try decoder.decode(NPZData.self, from: outputData)

        return (audio: data.audio, real: data.stft_real, imag: data.stft_imag)
    }
}
