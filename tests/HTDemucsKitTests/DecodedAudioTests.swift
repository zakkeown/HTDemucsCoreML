import Testing
@testable import HTDemucsKit

@Suite("DecodedAudio Tests")
struct DecodedAudioTests {
    @Test("Initialization with channels and metadata")
    func testInitialization() {
        let leftChannel: [Float] = [0.1, 0.2, 0.3]
        let rightChannel: [Float] = [0.4, 0.5, 0.6]
        let decoded = DecodedAudio(
            leftChannel: leftChannel,
            rightChannel: rightChannel,
            sampleRate: 44100,
            duration: 0.068
        )

        #expect(decoded.leftChannel == leftChannel)
        #expect(decoded.rightChannel == rightChannel)
        #expect(decoded.sampleRate == 44100)
        #expect(abs(decoded.duration - 0.068) < 0.001)
        #expect(decoded.channelCount == 2)
        #expect(decoded.frameCount == 3)
    }

    @Test("Stereo array conversion")
    func testStereoArray() {
        let decoded = DecodedAudio(
            leftChannel: [0.1, 0.2],
            rightChannel: [0.3, 0.4],
            sampleRate: 44100,
            duration: 0.045
        )

        let stereo = decoded.stereoArray
        #expect(stereo.count == 2)
        #expect(stereo[0] == [0.1, 0.2])
        #expect(stereo[1] == [0.3, 0.4])
    }
}
