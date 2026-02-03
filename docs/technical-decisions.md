# Technical Decisions

This document explains the "why" behind the architectural decisions in HTDemucs CoreML. It's written for ML engineers, curious Swift developers, and future-me who needs to remember why things work this way.

## Why Strip STFT/iSTFT from the Model?

### The Problem

HTDemucs performs STFT and iSTFT internally using PyTorch's `torch.stft()` and `torch.istft()` functions. These operate on complex-valued tensors (`torch.complex64`).

CoreML doesn't support complex number types natively. When coremltools encounters complex operations during conversion, it either:
1. Fails with unsupported operation errors
2. Produces incorrect numerical results due to improper handling of the real/imaginary components

### The Solution

Extract the "inner model" that operates purely on real-valued tensors, and implement STFT/iSTFT externally using Apple's vDSP Accelerate framework.

This approach:
- Avoids CoreML's complex number limitations entirely
- Gives us precise control over the signal processing
- Matches the approach used by successful ONNX conversions of similar audio models

### How It Works

1. **Model Surgery**: The `FullHybridHTDemucs` wrapper class accepts spectrograms (real/imaginary as separate channels) instead of raw audio. It runs the encoder, cross-transformer, and decoder—the actual neural network—but skips the STFT/iSTFT operations.

2. **Complex-as-Channels Format**: Instead of a complex tensor `[batch, channels, freq, time]` with complex64 dtype, we use `[batch, channels*2, freq, time]` with float32 dtype, where the channel dimension contains `[real_L, imag_L, real_R, imag_R]` for stereo.

3. **Swift STFT/iSTFT**: The AudioFFT class in Swift performs the same STFT/iSTFT operations using vDSP, with careful attention to matching PyTorch's exact padding, windowing, and normalization.

## Why vDSP Instead of Other Options?

### Options Considered

1. **Metal Compute Shaders**: Maximum flexibility, but significant development effort
2. **Accelerate vDSP**: Apple's optimized signal processing library
3. **AVAudioEngine FFT**: Higher-level API but less control
4. **Third-party libraries**: Additional dependencies

### Why vDSP Won

**Performance**: vDSP is hardware-optimized by Apple. It automatically uses SIMD instructions and dispatches to the best available hardware path.

**Availability**: vDSP is available on all Apple platforms (macOS, iOS, tvOS, watchOS) with consistent API. No deployment target concerns.

**Precision Control**: vDSP gives us direct control over normalization factors, which is critical for matching PyTorch's output exactly.

**No Dependencies**: Part of the system Accelerate framework, already linked by CoreML.

### Implementation Details

The key challenge was matching PyTorch's STFT output exactly. Differences we had to account for:

| Parameter | PyTorch | vDSP |
|-----------|---------|------|
| Normalization | `normalized=True` divides by `sqrt(N)` | No automatic normalization |
| FFT scaling | Returns unscaled magnitudes | Returns scaled by 2.0 |
| Packed format | Complex tensor | Split real/imag or packed |

The solution: Apply manual scaling factor of `0.5 / sqrt(N)` to vDSP output to match PyTorch's `normalized=True` behavior.

## Mixed Precision Strategy

### The Problem

Pure FP16 inference produces numerical instability in HTDemucs. Symptoms include:
- NaN values in normalization layers
- Overflow in attention softmax
- Accumulated precision loss in reductions

FP16's maximum value is 65,504. HTDemucs' internal activations can exceed this during normalization and attention operations.

### The Solution

Selective mixed precision: FP32 for precision-sensitive operations, FP16 for everything else.

**Operations kept in FP32:**
- `pow` — Power operations amplify numerical errors
- `sqrt` — Square root requires high precision for small values
- `real_div` — Division operations compound precision errors
- `l2_norm` — Normalization is precision-critical
- `reduce_mean`, `reduce_sum` — Reductions accumulate errors
- `softmax` — Attention mechanism is quality-critical
- `matmul` — Matrix multiplication benefits from precision

**Operations using FP16:**
- Convolutions (bulk of computation)
- Activations (ReLU, Sigmoid, Tanh)
- Concatenation, pooling, etc.

### How It Was Determined

1. **Baseline**: Convert everything to FP16, observe failures
2. **Selective promotion**: Identify operations causing NaN/Inf, promote to FP32
3. **Validation**: Compare CoreML output to PyTorch reference
4. **Iteration**: Repeat until SNR > 60 dB (perceptually identical)

The final precision selector is defined in `src/htdemucs_coreml/coreml_converter.py`:

```python
PRECISION_SENSITIVE_OPS = {
    "pow", "sqrt", "real_div", "l2_norm",
    "reduce_mean", "reduce_sum", "softmax", "matmul",
}
```

## Chunking and Overlap-Add

### Why Chunking Is Necessary

1. **Memory**: Processing a 5-minute song as a single tensor would require gigabytes of memory
2. **CoreML constraints**: The model has fixed input dimensions (343,980 samples = ~7.8 seconds)
3. **Latency**: Chunking enables streaming-style processing with intermediate progress updates

### Chunk Size

The chunk size is **343,980 samples** (~7.8 seconds at 44.1 kHz). This matches HTDemucs' training segment size, which is important because:

- The model's receptive field was trained on this duration
- Normalization statistics are computed per-segment during training
- Shorter chunks may not provide enough context for accurate separation

### Overlap Strategy

Chunks overlap by **1 second** (44,100 samples) on each side.

Without overlap, chunk boundaries would produce audible discontinuities. The overlap region allows us to crossfade between chunks for seamless reconstruction.

**Crossfade function:**
```
For a 1-second overlap at sample rate 44100:
- Fade-in: weight = sample_index / 44100  (0.0 → 1.0)
- Fade-out: weight = 1.0 - sample_index / 44100  (1.0 → 0.0)
```

### Edge Cases

**Short files (< 7.8 seconds):**
The audio is padded using reflect mode to reach the model's expected input size. After inference, the output is trimmed back to the original length.

**Exact chunk boundaries:**
The last chunk may be shorter than the full segment. It's padded, processed, then trimmed. The crossfade handles the transition smoothly.

## Model Surgery Approach

### What "Inner Model" Means

HTDemucs has this high-level structure:

```
Raw Audio
    ↓
STFT (complex-valued)
    ↓
┌───────────────────────────────┐
│ Encoder                       │
│ Cross-Transformer             │  ← "Inner Model" (what we extract)
│ Decoder                       │
└───────────────────────────────┘
    ↓
iSTFT
    ↓
Separated Stems
```

The "inner model" is the neural network portion—everything between STFT and iSTFT. It operates on real-valued spectrograms (using Complex-as-Channels format) and can be converted to CoreML.

### The Hybrid Architecture

HTDemucs is actually a hybrid model with two branches:

1. **Frequency branch**: Processes spectrograms through encoder/decoder
2. **Time branch**: Processes raw audio through a parallel encoder/decoder

The final output combines both: `output = time_branch + istft(freq_branch)`

Our `FullHybridHTDemucs` wrapper preserves both branches. The CoreML model accepts both spectrogram and raw audio inputs, and returns both frequency-domain and time-domain outputs. Swift then performs the iSTFT on the frequency output and sums the two branches.

### Normalization Layers

HTDemucs normalizes inputs before processing:

```python
# Frequency branch
mean = spectrogram.mean(dim=(1, 2, 3), keepdim=True)
std = spectrogram.std(dim=(1, 2, 3), keepdim=True)
x = (spectrogram - mean) / (1e-5 + std)

# Time branch (similar)
meant = raw_audio.mean(dim=(1, 2), keepdim=True)
stdt = raw_audio.std(dim=(1, 2), keepdim=True)
xt = (raw_audio - meant) / (1e-5 + stdt)
```

These normalizations stay in the model (they're in the FullHybridHTDemucs wrapper). The `1e-5` epsilon prevents division by zero. This is one of the precision-sensitive operations that must remain in FP32.

### Validation Methodology

To verify the CoreML model matches PyTorch:

1. **Generate test fixtures**: Run PyTorch HTDemucs on test audio (silence, sine wave, white noise), save intermediate spectrograms and final outputs as NumPy arrays

2. **Unit tests**: Compare Swift STFT output to PyTorch STFT output on the same audio

3. **Integration tests**: Run full pipeline (Swift STFT → CoreML → Swift iSTFT) and compare to PyTorch end-to-end output

4. **Quality metrics**: Compute SDR (Signal-to-Distortion Ratio), SIR (Signal-to-Interference Ratio), SAR (Signal-to-Artifacts Ratio) between CoreML and PyTorch outputs

**Target tolerances:**
- Model surgery layer: `torch.allclose(rtol=1e-5, atol=1e-7)`
- CoreML conversion: `np.allclose(rtol=1e-3, atol=1e-4)`
- End-to-end: SNR > 60 dB (perceptually identical)

The parity tests in `tests/parity/` automate this validation.
