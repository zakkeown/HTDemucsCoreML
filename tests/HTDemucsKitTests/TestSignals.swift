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
}
