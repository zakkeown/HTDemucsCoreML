# Phase 3: Audio I/O & Model Integration Design

**Date**: 2026-02-02
**Status**: Approved
**Target**: Production-ready end-to-end audio separation with extensive testing

## Overview

Phase 3 completes the HTDemucsKit pipeline by adding audio I/O and full integration with the htdemucs_6s CoreML model. This enables real-world audio separation with progress monitoring and comprehensive testing infrastructure.

## Goals

- Support multiple audio formats (WAV, MP3, FLAC, OGG, M4A) for input and output
- Integrate htdemucs_6s model (6-source separation: drums, bass, vocals, other, piano, guitar)
- Provide progress reporting to detect hangs vs. slow processing
- Build extensive testing infrastructure for quality validation
- Create production-ready CLI tool for testing and validation

## Architecture

### System Layers

**Audio I/O Layer**: Swift FFmpeg package wrapper handles audio decoding/encoding. Provides maximum format support without requiring system FFmpeg installation. Decodes to PCM float arrays, encodes separated stems back to user's chosen format.

**Model Management**: Simple approach - htdemucs_6s CoreML model committed to `Resources/Models/htdemucs_6s.mlpackage/` in the repository. `ModelLoader` loads from fixed path. No download, versioning, or checksum complexity.

**Processing Pipeline**: Existing `SeparationPipeline` from Phase 2B integrates seamlessly. Receives decoded audio, chunks it (10s chunks, 1s overlap), runs STFT → CoreML inference → iSTFT, outputs 6 separated stems. No changes needed to core logic.

**Progress Reporting**: AsyncStream provides real-time progress updates through the pipeline. Yields progress events (decoding, chunk N of M, encoding stems) that CLI displays. Critical for debugging long-running separations and detecting hangs.

## Components

### AudioDecoder / AudioEncoder

Thin wrappers around Swift FFmpeg package:

- `AudioDecoder.decode(fileURL:) throws -> DecodedAudio` - Returns PCM float arrays with sample rate and channel info
- `AudioEncoder.encode(audio:format:destination:) throws` - Writes separated stems to disk
- Both handle format detection automatically
- Throw descriptive errors for unsupported/corrupted files

### ModelLoader (Simplified)

Loads CoreML model from fixed repository path:

- `init(modelName: String = "htdemucs_6s") throws` - Loads from `Resources/Models/{modelName}.mlpackage`
- `load() throws -> MLModel` - Returns cached CoreML model
- No download, verification, or version management complexity

### SeparationCoordinator

New orchestration layer coordinating the full pipeline:

```swift
class SeparationCoordinator {
    func separate(input: URL, outputDir: URL, format: AudioFormat?) async throws
        -> AsyncStream<ProgressEvent>
}
```

**Progress Events**:
- `.decoding(progress: Float)` - Audio file being decoded
- `.processing(chunk: Int, total: Int)` - Current chunk being processed
- `.encoding(stem: StemType, progress: Float)` - Stem being written to disk
- `.complete(outputPaths: [StemType: URL])` - All stems written successfully
- `.failed(error: Error)` - Operation failed with error

### CLI Updates

**Enhanced separate command**:
```bash
htdemucs-cli separate input.mp3 --output ./stems/ [--format wav]
```

Features:
- Progress bar showing decoding → processing (chunk N/M) → encoding
- Default output format matches input (MP3 → MP3), overridable with `--format`
- Clear error messages with actionable diagnostics

## Data Flow

### Input Processing

1. User runs `htdemucs-cli separate song.mp3 --output ./stems/`
2. `AudioDecoder.decode()` reads file via FFmpeg package → stereo PCM float arrays at native sample rate
3. System validates 44.1kHz (required for model) - resamples if needed via vDSP
4. Stereo arrays `[leftChannel, rightChannel]` flow into `SeparationPipeline`

### Core Processing (Phase 2B Pipeline)

5. `ChunkProcessor` splits audio into 10s chunks with 1s overlap
6. For each chunk:
   - `AudioFFT.stft()` → `[2][2049][timeFrames]` complex spectrogram
   - `InferenceEngine` runs CoreML → `[6][2][2049][timeFrames]` separation masks
   - Apply masks to input spectrogram → 6 separated spectrograms
   - `AudioFFT.istft()` on each → 6 time-domain audio chunks
7. Overlap-add with crossfade blending → continuous separated streams

### Output Flow

8. 6 complete separated audio arrays (drums, bass, vocals, other, piano, guitar)
9. `AudioEncoder.encode()` writes each stem to `{outputDir}/{stemName}.{format}`
10. Default format matches input, overridable with `--format` flag

### Progress Events

Emitted at:
- Decoding start/complete
- Each chunk start: `.processing(chunk: 5, total: 23)`
- Each stem encoding: `.encoding(stem: .drums, progress: 0.6)`
- Final completion with output paths

## Error Handling

### Fail-Fast Philosophy

Every operation validates inputs and fails immediately with actionable error messages. No silent fallbacks, no partial state corruption. User gets clear diagnostics.

### Error Types

**Audio I/O Errors**:
- `AudioError.fileNotFound(path:)` - Missing input file
- `AudioError.unsupportedFormat(format:, reason:)` - FFmpeg can't decode (DRM, corruption, unknown codec)
- `AudioError.decodeFailed(underlyingError:)` - FFmpeg internal error with details
- `AudioError.encodeFailed(stem:, reason:)` - Write permission, disk full, invalid output path

**Model Errors**:
- `ModelError.notFound(name:)` - Model file missing from Resources/Models/
- `ModelError.loadFailed(reason:)` - CoreML loading error
- `ModelError.incompatibleVersion(model:, required:)` - Model format mismatch

**Processing Errors**:
- `ProcessingError.invalidSampleRate(actual:, required:)` - Non-44.1kHz after resample attempt
- `ProcessingError.invalidChannelCount(actual:, required:)` - Mono/multichannel when stereo required
- `ProcessingError.inferenceFailed(chunk:, reason:)` - CoreML error with chunk context
- `ProcessingError.outOfMemory` - Audio too long, exceeds available RAM

### Error Recovery

None. When error occurs:
- Operation stops immediately
- AsyncStream yields `.failed(error:)`
- Temp files cleaned up
- Detailed error logged
- User must fix root cause and retry

## Testing Strategy

### Unit Tests (~51 new tests)

**AudioDecoder/Encoder Tests** (15 tests):
- Decode WAV, MP3, FLAC fixtures → validate sample rate, channel count, audio length
- Encode PCM arrays → decode result → verify round-trip accuracy
- Error handling: missing files, corrupted headers, invalid formats
- Use small test files (~1-2 seconds) for speed

**ModelLoader Tests** (8 tests):
- Load from valid path → verify model loads
- Missing model file → proper error
- Corrupted .mlpackage → proper error
- Model compatibility check

**SeparationCoordinator Tests** (10 tests):
- Mock decoder/encoder/pipeline, verify orchestration flow
- Validate progress event sequence and values
- Error propagation from each layer
- Verify temp file cleanup on failure

**Integration Tests** (12 tests):
- End-to-end: real audio file → separation → verify 6 output files exist
- Use existing STFT/CoreML test fixtures (validated in Phase 2B)
- Progress event completeness (all expected events emitted)
- Multi-format test: MP3 in → WAV out, FLAC in → MP3 out
- Real htdemucs_6s model inference on test audio

**CLI Tests** (6 tests):
- Argument parsing for all flags
- Separate command integration with mocked components
- Error message formatting
- Progress bar rendering

### Manual Validation

- Run on known reference tracks
- Verify separated stems sound correct (drums isolated, vocals clean)
- Quick subjective quality check (not automated)
- Test with diverse audio: music, speech, ambient sounds

### Total Test Coverage

- **Phase 2B Foundation**: 64 tests (STFT, CoreML, Pipeline)
- **Phase 3 New Tests**: 51 tests (Audio I/O, Integration)
- **Grand Total**: 115 tests

## Implementation Notes

### Model Setup (One-Time)

1. Download htdemucs_6s CoreML model from HuggingFace
2. Place in `Resources/Models/htdemucs_6s.mlpackage/`
3. Commit to repository (git handles integrity)
4. Model size: ~100-200MB (acceptable for repo)

### Swift FFmpeg Package

Use existing Swift package wrapper (e.g., swift-ffmpeg or similar). Provides:
- Pure Swift API (no command-line wrapper)
- Bundled FFmpeg binaries (no system dependency)
- Format auto-detection
- PCM float output for processing
- Encoding to common formats

### iOS Bundling (Future)

- Model file already in repo → copy to iOS app bundle resources
- ModelLoader adapts to load from Bundle.main.path() instead of fixed path
- Same model, same pipeline, zero download complexity

## Success Criteria

1. ✅ CLI successfully separates real audio files (MP3, WAV, FLAC) into 6 stems
2. ✅ Progress reporting works, helps identify hangs
3. ✅ 115 total tests passing (64 existing + 51 new)
4. ✅ Error messages are clear and actionable
5. ✅ Manual validation: separated stems sound clean and isolated
6. ✅ Ready for iOS app integration (model bundled, pipeline reusable)

## Dependencies

- **Phase 2A**: STFT/iSTFT with vDSP (complete ✅)
- **Phase 2B**: CoreML integration and pipeline (complete ✅)
- **New**: Swift FFmpeg package
- **New**: htdemucs_6s CoreML model file

## Timeline Note

No time estimates provided per project policy. Work broken into actionable steps, user judges timing.
