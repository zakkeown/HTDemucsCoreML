import Foundation

/// Common audio types and test utilities
public enum AudioTypes {
    /// Standard sample rate for testing
    public static let sampleRate = 44100
}

/// Test signal generators
public enum TestSignals {
    /// Generate sine wave
    public static func sine(frequency: Float, duration: Float, sampleRate: Int = AudioTypes.sampleRate) -> [Float] {
        let samples = Int(duration * Float(sampleRate))
        let angularFreq = 2 * Float.pi * frequency / Float(sampleRate)
        return (0..<samples).map { Float(sin(angularFreq * Float($0))) }
    }

    /// Generate white noise
    public static func whiteNoise(samples: Int) -> [Float] {
        return (0..<samples).map { _ in Float.random(in: -1...1) }
    }

    /// Generate silence
    public static func silence(samples: Int) -> [Float] {
        return [Float](repeating: 0, count: samples)
    }
}
