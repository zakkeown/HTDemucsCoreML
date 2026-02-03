# Swift API Guide

This guide covers using HTDemucsKit in your Swift projects.

## Installation

Add HTDemucsKit to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/youruser/HTDemucsCoreML.git", from: "1.0.0")
],
targets: [
    .target(
        name: "YourApp",
        dependencies: ["HTDemucsKit"]
    )
]
```

## Basic Usage

### Using SeparationCoordinator (Recommended)

The simplest way to separate audio is with `SeparationCoordinator`, which handles decoding, separation, and encoding:

```swift
import HTDemucsKit

let coordinator = try SeparationCoordinator()

for await event in coordinator.separate(
    input: inputURL,
    outputDir: outputDir,
    format: .wav
) {
    switch event {
    case .decoding(let progress):
        print("Decoding: \(Int(progress * 100))%")

    case .processing(let chunk, let total):
        print("Processing chunk \(chunk + 1) of \(total)")

    case .encoding(let stem, let progress):
        print("Encoding \(stem.rawValue): \(Int(progress * 100))%")

    case .complete(let paths):
        print("Done! Stems saved to:")
        for (stem, url) in paths {
            print("  \(stem.rawValue): \(url.path)")
        }

    case .failed(let error):
        print("Error: \(error.localizedDescription)")
    }
}
```

### Using SeparationPipeline Directly

For more control, use `SeparationPipeline` with raw audio data:

```swift
import HTDemucsKit

// Load the CoreML model
let loader = try ModelLoader()  // Uses default "htdemucs_6s"
let model = try loader.load()

// Create pipeline
let pipeline = try SeparationPipeline(model: model)

// Separate stereo audio (44.1 kHz float samples)
let stems = try pipeline.separate(
    stereoAudio: [leftChannel, rightChannel],
    progressCallback: { chunk, total in
        print("Chunk \(chunk + 1)/\(total)")
    }
)

// Access individual stems
let drums = stems[.drums]!   // [[Float]] - [left, right]
let vocals = stems[.vocals]!
let bass = stems[.bass]!
```

## Progress Tracking

`SeparationCoordinator.separate()` returns an `AsyncStream<ProgressEvent>`. Events are:

| Event | Description |
|-------|-------------|
| `.decoding(progress:)` | Audio file being decoded (0.0 to 1.0) |
| `.processing(chunk:total:)` | Model inference on chunk N of M |
| `.encoding(stem:progress:)` | Stem being written to file |
| `.complete(outputPaths:)` | All stems written successfully |
| `.failed(error:)` | An error occurred |

### SwiftUI Integration

```swift
struct SeparationView: View {
    @State private var progress: String = "Ready"
    @State private var isProcessing = false

    var body: some View {
        VStack {
            Text(progress)
            Button("Separate") {
                Task { await separate() }
            }
            .disabled(isProcessing)
        }
    }

    func separate() async {
        isProcessing = true
        let coordinator = try! SeparationCoordinator()

        for await event in coordinator.separate(
            input: inputURL,
            outputDir: outputDir,
            format: .wav
        ) {
            progress = event.description

            if case .complete = event {
                isProcessing = false
            }
            if case .failed = event {
                isProcessing = false
            }
        }
    }
}
```

## Accessing Stems

The 6 stems are accessed via the `StemType` enum:

```swift
public enum StemType: String, CaseIterable {
    case drums
    case bass
    case other
    case vocals
    case guitar
    case piano
}
```

Each stem is stereo: `[[Float]]` where index 0 is left channel, index 1 is right channel.

```swift
let stems = try pipeline.separate(stereoAudio: audio)

// Get vocals as stereo pair
let vocalsLeft = stems[.vocals]![0]
let vocalsRight = stems[.vocals]![1]

// Iterate all stems
for stemType in StemType.allCases {
    let stemAudio = stems[stemType]!
    print("\(stemType.rawValue): \(stemAudio[0].count) samples")
}
```

## Output Formats

`SeparationCoordinator` supports three output formats:

```swift
public enum AudioFormat: String {
    case wav   // Uncompressed, highest quality
    case mp3   // Compressed, smaller files
    case flac  // Lossless compression
}
```

Example:

```swift
// Output as FLAC
for await event in coordinator.separate(
    input: inputURL,
    outputDir: outputDir,
    format: .flac
) { ... }
```

## Configuration

### Custom Model Path

```swift
// Load from explicit path
let loader = try ModelLoader(modelPath: "/path/to/model.mlpackage")
let model = try loader.load()
let pipeline = try SeparationPipeline(model: model)
```

### Sample Rate

HTDemucsKit expects **44.1 kHz** audio. If your audio has a different sample rate, resample it first. The pipeline will throw `PipelineError.invalidSampleRate` if the rate doesn't match.

## Error Handling

### Pipeline Errors

```swift
public enum PipelineError: Error {
    case invalidChannelCount(Int)      // Expected stereo (2 channels)
    case channelLengthMismatch         // L/R channels different lengths
    case emptyAudio                    // Zero-length audio
    case invalidSampleRate(got:expected:)  // Wrong sample rate
}
```

### Model Errors

```swift
public enum ModelError: Error {
    case modelNotFound(String)         // Model file doesn't exist
    case loadFailed(String)            // CoreML loading failed
}
```

### Example Error Handling

```swift
do {
    let coordinator = try SeparationCoordinator()

    for await event in coordinator.separate(input: url, outputDir: dir, format: .wav) {
        if case .failed(let error) = event {
            switch error {
            case let e as PipelineError:
                print("Pipeline error: \(e.localizedDescription)")
            case let e as ModelError:
                print("Model error: \(e.localizedDescription)")
            default:
                print("Unknown error: \(error)")
            }
        }
    }
} catch {
    print("Initialization error: \(error)")
}
```

## Memory Considerations

### Large Files

For long audio files, the pipeline processes in ~7.8 second chunks with overlap. Memory usage is proportional to chunk size, not total file length. Typical peak usage is ~500 MB.

### Releasing Resources

The CoreML model is cached after first load. To release memory, let the `ModelLoader` and `SeparationPipeline` instances go out of scope.

### Tips

- Process one file at a time to avoid memory pressure
- On iOS, consider using background processing for long files
- The model runs on Neural Engine when available, which is more power-efficient than GPU
