# Architecture Overview

This document explains how HTDemucs CoreML works, from audio input to separated stems.

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Audio File                                      │
│                        (MP3, WAV, FLAC, etc.)                               │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  AudioDecoder (FFmpeg)                                                       │
│  ─────────────────────                                                       │
│  Decodes any audio format to raw PCM float samples                          │
│  Output: Stereo float audio at 44.1 kHz                                     │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ChunkProcessor                                                              │
│  ──────────────                                                              │
│  Splits long audio into ~7.8s segments with 1s overlap                      │
│  Each chunk processed independently, then blended with crossfade            │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
           ┌──────────────────┴──────────────────┐
           │         For each chunk:             │
           ▼                                     ▼
┌──────────────────────────┐      ┌──────────────────────────┐
│  AudioFFT.stft()         │      │  Raw Audio               │
│  ────────────────        │      │  ──────────              │
│  vDSP FFT with Hann      │      │  Passed directly to      │
│  window                  │      │  model's time branch     │
│  4096-point, 1024 hop    │      │                          │
│  Output: Complex spectrogram    │                          │
└────────────┬─────────────┘      └────────────┬─────────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CoreML Model (HTDemucs "Inner Model")                                      │
│  ─────────────────────────────────────                                      │
│                                                                              │
│  Inputs:                                                                     │
│    • spectrogram: [1, 4, 2048, 336] — real/imag for L/R channels           │
│    • raw_audio:   [1, 2, 343980]    — stereo waveform                       │
│                                                                              │
│  The model has two branches:                                                 │
│    Frequency branch: Processes spectrogram → 6 separated spectrograms       │
│    Time branch:      Processes raw audio → 6 separated waveforms            │
│                                                                              │
│  Outputs:                                                                    │
│    • add_66: [1, 6, 4, 2048, 336] — frequency-domain stems                  │
│    • add_67: [1, 6, 2, 343980]    — time-domain stems                       │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
           ┌──────────────────┴──────────────────┐
           │         For each stem:              │
           ▼                                     ▼
┌──────────────────────────┐      ┌──────────────────────────┐
│  AudioFFT.istft()        │      │  Time Branch Output      │
│  ─────────────────       │      │  ───────────────────     │
│  Inverse FFT with        │      │  Already in time domain  │
│  overlap-add             │      │                          │
│  Output: Time-domain audio      │                          │
└────────────┬─────────────┘      └────────────┬─────────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Combine: time_output + istft(freq_output)                                  │
│  ─────────────────────────────────────────                                  │
│  The hybrid model's final output is the sum of both branches                │
│  Output: 6 separated stereo stems                                           │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Overlap-Add Blending                                                        │
│  ────────────────────                                                        │
│  Chunks are blended with linear crossfade in overlap regions                │
│  Output: Full-length separated stems                                        │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  AudioEncoder (FFmpeg)                                                       │
│  ─────────────────────                                                       │
│  Encodes stems to output format (WAV, FLAC, MP3)                            │
│  Output: 6 audio files (drums, bass, vocals, other, piano, guitar)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why This Architecture?

**The problem:** HTDemucs uses STFT/iSTFT operations that operate on complex numbers. CoreML doesn't support complex tensors natively, and attempts to convert these operations produce incorrect results.

**The solution:** Extract the "inner model" that operates purely on real-valued tensors (spectrograms with real/imaginary as separate channels), and implement STFT/iSTFT natively using Apple's vDSP framework.

This matches the approach used by successful ONNX conversions of similar models.

## Component Details

### AudioDecoder / AudioEncoder

**Location:** `Sources/HTDemucsKit/Audio/`

FFmpeg-based audio I/O supporting any format FFmpeg can handle. The decoder outputs 44.1 kHz stereo float PCM regardless of input format. The encoder writes stems in the user's requested format.

### AudioFFT

**Location:** `Sources/HTDemucsKit/Audio/AudioFFT.swift`

Implements STFT and iSTFT using Apple's vDSP Accelerate framework. Key parameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| FFT size | 4096 | ~93ms window at 44.1 kHz |
| Hop length | 1024 | 75% overlap |
| Window | Hann | Standard for audio |
| Normalization | `0.5/sqrt(N)` | Matches PyTorch `normalized=True` |

The implementation matches HTDemucs' specific padding scheme:
- Left pad: `hop_length // 2 * 3` = 1536 samples
- Frame trimming: Skip first 2 frames, keep `ceil(length / hop_length)` frames

### ChunkProcessor

**Location:** `Sources/HTDemucsKit/Pipeline/ChunkProcessor.swift`

Handles audio longer than the model's native segment size (~7.8 seconds). Chunks overlap by 1 second on each side. The overlap regions are blended using a linear crossfade to avoid discontinuities at chunk boundaries.

### InferenceEngine

**Location:** `Sources/HTDemucsKit/CoreML/InferenceEngine.swift`

Wraps the CoreML model for inference. Handles:
- Input tensor preparation (spectrogram + raw audio)
- Padding/trimming to match model's fixed input size
- Output tensor unpacking (6 stems × 2 channels × freq/time branches)

**Tensor shapes:**
- Spectrogram input: `[1, 4, 2048, 336]` — batch, channels (L_real, L_imag, R_real, R_imag), freq bins, time frames
- Audio input: `[1, 2, 343980]` — batch, channels, samples
- Frequency output: `[1, 6, 4, 2048, 336]` — batch, stems, channels, freq bins, time frames
- Time output: `[1, 6, 2, 343980]` — batch, stems, channels, samples

### SeparationPipeline

**Location:** `Sources/HTDemucsKit/Pipeline/SeparationPipeline.swift`

Orchestrates the full separation flow. Takes stereo audio, coordinates chunking, runs inference, combines branches, and returns separated stems.

### SeparationCoordinator

**Location:** `Sources/HTDemucsKit/Pipeline/SeparationCoordinator.swift`

Provides an async interface with progress streaming via `AsyncStream<ProgressEvent>`. Useful for UI integration where you need real-time progress updates.

## Data Flow Details

### Sample Rates and Sizes

| Parameter | Value |
|-----------|-------|
| Sample rate | 44,100 Hz |
| Segment duration | ~7.8 seconds |
| Segment samples | 343,980 |
| Spectrogram frames | 336 |
| Frequency bins | 2048 (Nyquist excluded) |

### Stem Output Order

The model outputs stems in this order (matching PyTorch):

| Index | Stem |
|-------|------|
| 0 | drums |
| 1 | bass |
| 2 | other |
| 3 | vocals |
| 4 | guitar |
| 5 | piano |

Note: "other" comes before "vocals" in the model output.

### Hybrid Model Combination

The final stem audio is computed as:

```
stem_audio = time_branch_output + istft(freq_branch_output)
```

Both branches contribute to the final result. The time branch captures transients well; the frequency branch captures tonal content well.
