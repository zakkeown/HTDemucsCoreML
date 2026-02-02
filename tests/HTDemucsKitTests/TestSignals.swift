import Foundation

enum TestSignals {
    /// Generate a sine wave test signal
    static func sine(frequency: Float, duration: Float, sampleRate: Float) -> [Float] {
        let numSamples = Int(duration * sampleRate)
        return (0..<numSamples).map { i in
            let t = Float(i) / sampleRate
            return sin(2.0 * .pi * frequency * t)
        }
    }
}
