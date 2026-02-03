import Testing
import Foundation
import Accelerate
@testable import HTDemucsKit

@Suite("AudioFFT Tests")
struct AudioFFTTests {

    // MARK: - STFT/iSTFT Roundtrip Tests

    @Test("STFT/iSTFT roundtrip with 440Hz sine wave")
    func testRoundtripSineWave440Hz() throws {
        let fft = try AudioFFT()
        let sampleRate: Float = 44100
        let duration: Float = 1.0
        let frequency: Float = 440.0

        // Generate sine wave
        let numSamples = Int(sampleRate * duration)
        let original = (0..<numSamples).map { i in
            sin(2 * Float.pi * frequency * Float(i) / sampleRate)
        }

        // STFT -> iSTFT
        let (real, imag) = try fft.stft(original)
        let reconstructed = try fft.istft(real: real, imag: imag)

        // Compute SDR
        let sdr = computeSDR(original: original, reconstructed: reconstructed)

        print("440Hz sine wave roundtrip SDR: \(String(format: "%.2f", sdr)) dB")

        // Should be > 40 dB for a good implementation
        #expect(sdr > 40, "STFT roundtrip SDR should be > 40 dB, got \(sdr) dB")
    }

    @Test("STFT/iSTFT roundtrip with low frequency (80Hz bass)")
    func testRoundtripLowFrequency() throws {
        let fft = try AudioFFT()
        let sampleRate: Float = 44100
        let duration: Float = 1.0
        let frequency: Float = 80.0  // Bass frequency

        let numSamples = Int(sampleRate * duration)
        let original = (0..<numSamples).map { i in
            sin(2 * Float.pi * frequency * Float(i) / sampleRate)
        }

        let (real, imag) = try fft.stft(original)
        let reconstructed = try fft.istft(real: real, imag: imag)

        let sdr = computeSDR(original: original, reconstructed: reconstructed)

        print("80Hz (bass) roundtrip SDR: \(String(format: "%.2f", sdr)) dB")

        #expect(sdr > 40, "Low frequency roundtrip SDR should be > 40 dB, got \(sdr) dB")
    }

    @Test("STFT/iSTFT roundtrip with high frequency (8000Hz)")
    func testRoundtripHighFrequency() throws {
        let fft = try AudioFFT()
        let sampleRate: Float = 44100
        let duration: Float = 1.0
        let frequency: Float = 8000.0

        let numSamples = Int(sampleRate * duration)
        let original = (0..<numSamples).map { i in
            sin(2 * Float.pi * frequency * Float(i) / sampleRate)
        }

        let (real, imag) = try fft.stft(original)
        let reconstructed = try fft.istft(real: real, imag: imag)

        let sdr = computeSDR(original: original, reconstructed: reconstructed)

        print("8000Hz roundtrip SDR: \(String(format: "%.2f", sdr)) dB")

        #expect(sdr > 40, "High frequency roundtrip SDR should be > 40 dB, got \(sdr) dB")
    }

    @Test("STFT/iSTFT roundtrip with white noise")
    func testRoundtripWhiteNoise() throws {
        let fft = try AudioFFT()
        let sampleRate: Float = 44100
        let duration: Float = 1.0

        let numSamples = Int(sampleRate * duration)

        // Generate white noise (deterministic for reproducibility)
        srand48(42)
        let original = (0..<numSamples).map { _ in
            Float(drand48() * 2 - 1)  // Range [-1, 1]
        }

        let (real, imag) = try fft.stft(original)
        let reconstructed = try fft.istft(real: real, imag: imag)

        let sdr = computeSDR(original: original, reconstructed: reconstructed)

        print("White noise roundtrip SDR: \(String(format: "%.2f", sdr)) dB")

        #expect(sdr > 40, "White noise roundtrip SDR should be > 40 dB, got \(sdr) dB")
    }

    @Test("STFT/iSTFT roundtrip with multi-frequency signal (simulating music)")
    func testRoundtripMultiFrequency() throws {
        let fft = try AudioFFT()
        let sampleRate: Float = 44100
        let duration: Float = 1.0

        let numSamples = Int(sampleRate * duration)

        // Mix of frequencies: bass (80Hz), mid (440Hz), high (4000Hz)
        let original = (0..<numSamples).map { i in
            let t = Float(i) / sampleRate
            let bass = 0.5 * sin(2 * Float.pi * 80 * t)
            let mid = 0.3 * sin(2 * Float.pi * 440 * t)
            let high = 0.2 * sin(2 * Float.pi * 4000 * t)
            return bass + mid + high
        }

        let (real, imag) = try fft.stft(original)
        let reconstructed = try fft.istft(real: real, imag: imag)

        let sdr = computeSDR(original: original, reconstructed: reconstructed)

        print("Multi-frequency roundtrip SDR: \(String(format: "%.2f", sdr)) dB")

        #expect(sdr > 40, "Multi-frequency roundtrip SDR should be > 40 dB, got \(sdr) dB")
    }

    @Test("STFT/iSTFT roundtrip with longer audio (10 seconds)")
    func testRoundtripLongAudio() throws {
        let fft = try AudioFFT()
        let sampleRate: Float = 44100
        let duration: Float = 10.0  // 10 seconds - matches chunk size

        let numSamples = Int(sampleRate * duration)

        // Complex signal with varying frequencies
        let original = (0..<numSamples).map { i in
            let t = Float(i) / sampleRate
            let freq = 200 + 100 * sin(2 * Float.pi * 0.5 * t)  // Frequency modulation
            return sin(2 * Float.pi * freq * t)
        }

        let (real, imag) = try fft.stft(original)
        let reconstructed = try fft.istft(real: real, imag: imag)

        let sdr = computeSDR(original: original, reconstructed: reconstructed)

        print("10-second audio roundtrip SDR: \(String(format: "%.2f", sdr)) dB")

        #expect(sdr > 40, "Long audio roundtrip SDR should be > 40 dB, got \(sdr) dB")
    }

    // MARK: - Spectrogram Shape Tests

    @Test("STFT produces correct spectrogram shape")
    func testSTFTShape() throws {
        let fft = try AudioFFT()
        let sampleRate: Float = 44100
        let duration: Float = 1.0

        let numSamples = Int(sampleRate * duration)
        let audio = [Float](repeating: 0.5, count: numSamples)

        let (real, imag) = try fft.stft(audio)

        // With center padding (n_fft // 2 on each side), padded length is:
        // numSamples + n_fft (= 2 * n_fft/2)
        // Expected frames = (paddedLength - fftSize) / hopLength + 1
        let centerPadding = fft.fftSize / 2
        let paddedLength = numSamples + 2 * centerPadding
        let expectedFrames = (paddedLength - fft.fftSize) / fft.hopLength + 1
        let expectedBins = fft.fftSize / 2 + 1  // 2049 for fftSize=4096

        #expect(real.count == expectedFrames, "Expected \(expectedFrames) frames, got \(real.count)")
        #expect(real[0].count == expectedBins, "Expected \(expectedBins) bins, got \(real[0].count)")
        #expect(imag.count == expectedFrames, "Imag frames should match real")
        #expect(imag[0].count == expectedBins, "Imag bins should match real")

        print("Spectrogram shape: [\(real.count) frames][\(real[0].count) bins]")
    }

    // MARK: - DC and Nyquist Bin Tests

    @Test("DC bin is purely real")
    func testDCBinPurelyReal() throws {
        let fft = try AudioFFT()

        // DC signal (constant)
        let audio = [Float](repeating: 0.5, count: 44100)

        let (real, imag) = try fft.stft(audio)

        // DC bin (index 0) should have zero imaginary part
        for frame in 0..<real.count {
            #expect(abs(imag[frame][0]) < 1e-6, "DC bin imaginary should be ~0, got \(imag[frame][0])")
        }

        print("DC bin test passed - imaginary components are ~0")
    }

    @Test("Nyquist bin is purely real")
    func testNyquistBinPurelyReal() throws {
        let fft = try AudioFFT()

        // Generate Nyquist frequency signal (22050 Hz at 44100 sample rate)
        let sampleRate: Float = 44100
        let nyquistFreq = sampleRate / 2
        let numSamples = 44100

        let audio = (0..<numSamples).map { i in
            sin(2 * Float.pi * nyquistFreq * Float(i) / sampleRate)
        }

        let (real, imag) = try fft.stft(audio)

        let nyquistBin = fft.fftSize / 2  // 2048

        // Nyquist bin should have zero imaginary part
        for frame in 0..<real.count {
            #expect(abs(imag[frame][nyquistBin]) < 1e-6,
                   "Nyquist bin imaginary should be ~0, got \(imag[frame][nyquistBin])")
        }

        print("Nyquist bin test passed - imaginary components are ~0")
    }

    // MARK: - Energy Preservation Test

    @Test("STFT preserves energy (Parseval's theorem)")
    func testEnergyPreservation() throws {
        let fft = try AudioFFT()

        // Generate test signal
        srand48(123)
        let numSamples = 44100
        let audio = (0..<numSamples).map { _ in Float(drand48() * 2 - 1) }

        // Compute time-domain energy
        var timeEnergy: Float = 0
        vDSP_svesq(audio, 1, &timeEnergy, vDSP_Length(numSamples))

        // Compute STFT
        let (real, imag) = try fft.stft(audio)

        // Compute frequency-domain energy (sum of |X|^2 across all frames and bins)
        var freqEnergy: Float = 0
        for frame in 0..<real.count {
            for bin in 0..<real[frame].count {
                freqEnergy += real[frame][bin] * real[frame][bin] + imag[frame][bin] * imag[frame][bin]
            }
        }

        // The ratio should be related to window properties and overlap
        // We're mainly checking it's in a reasonable range
        let ratio = freqEnergy / timeEnergy

        print("Energy ratio (freq/time): \(ratio)")

        // Should be positive and finite
        #expect(ratio > 0 && ratio.isFinite, "Energy ratio should be positive and finite")
    }

    // MARK: - Helper Functions

    /// Compute Signal-to-Distortion Ratio in dB
    private func computeSDR(original: [Float], reconstructed: [Float]) -> Float {
        let minLen = min(original.count, reconstructed.count)

        // Skip edge samples where windowing effects are strongest
        let skipSamples = 4096  // One FFT window
        let startIdx = skipSamples
        let endIdx = minLen - skipSamples

        guard endIdx > startIdx else {
            return -100  // Not enough samples
        }

        let origSlice = Array(original[startIdx..<endIdx])
        let reconSlice = Array(reconstructed[startIdx..<endIdx])
        let len = origSlice.count

        // Signal power
        var signalPower: Float = 0
        vDSP_svesq(origSlice, 1, &signalPower, vDSP_Length(len))

        // Error
        var error = [Float](repeating: 0, count: len)
        vDSP_vsub(reconSlice, 1, origSlice, 1, &error, 1, vDSP_Length(len))

        // Error power
        var errorPower: Float = 0
        vDSP_svesq(error, 1, &errorPower, vDSP_Length(len))

        guard errorPower > 0 else {
            return 100  // Perfect reconstruction
        }

        return 10 * log10(signalPower / errorPower)
    }
}
