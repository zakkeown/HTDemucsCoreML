# Porting HT Demucs to CoreML: A Technical Guide

**Direct CoreML conversion of HT Demucs remains unsolved as of February 2026.** The fundamental blocker is CoreML's lack of support for complex-valued operations required by STFT/iSTFT. However, a viable path exists by stripping these operations from the model and implementing them natively, following patterns established by successful ONNX conversions and the Whisper.cpp CoreML implementation.

## The architecture you're trying to convert

HT Demucs (Hybrid Transformer Demucs v4) is a **dual-branch U-Net** that processes audio simultaneously in time and frequency domains. The model uses a **cross-domain Transformer encoder** at its core, connecting 4 CNN encoder/decoder layers in each branch with 5 interleaved self-attention and cross-attention layers.

The 6-source model (htdemucs_6s) outputs stems for drums, bass, vocals, guitar, piano, and other instruments. Key architectural parameters include **4096-point FFT** with **1024-sample hop length**, **48 base channels** growing to **512 at the bottleneck**, **8 attention heads** with **384-dimensional embeddings**, and approximately **27M parameters**. The model processes audio at 44.1kHz and uses 10-second segments during inference with 25% overlap for reconstruction.

The critical operations for conversion include `torch.stft`/`torch.istft` for time-frequency conversion, `torch.nn.MultiheadAttention` for cross-domain attention, `torch.nn.GroupNorm` and `torch.nn.LayerNorm` for normalization, and Complex-as-Channels (CaC) representation where real and imaginary STFT components are concatenated along the channel dimension.

## Why direct conversion fails

### Complex number operations are the primary blocker

CoreML and coremltools fundamentally **do not support complex numbers**. This affects every operation in the STFT/iSTFT pipeline:

- `torch.complex(real, imag)` — NOT SUPPORTED
- `torch.fft.fft/rfft/stft` — NOT SUPPORTED
- `torch.view_as_real/view_as_complex` — NOT SUPPORTED
- Complex tensor dtype (`torch.complex64`) — NOT SUPPORTED

When attempting direct conversion, you'll encounter errors like: `ValueError: Op "slice_by_index" expects tensor of dtype from ['fp16', 'fp32', 'int32', 'bool'] but got tensor[1,2,2049,440,complex64]` as documented in coremltools GitHub issue #2112.

### Transformer conversion has additional challenges

The `scaled_dot_product_attention` operation wasn't supported until **coremltools 8.0** (requiring iOS 18/macOS 15). PyTorch's `torch._native_multi_head_attention` fast path causes conversion failures—you must disable it with `torch.backends.mha.set_fastpath_enabled(False)` before tracing. Deep transformer networks cause the ANECompilerService to take exponentially longer, sometimes requiring model splitting.

### No existing CoreML implementation exists

Extensive search reveals **no working Demucs CoreML implementations** on GitHub or in community discussions. The closest related work includes spleeter-pytorch (partial CoreML conversion with STFT handled externally), Whisper.cpp with CoreML encoder acceleration, and djay Pro AI's "Neural Mix" feature (proprietary, built from ground up for CoreML).

## The viable conversion strategy

The only workable approach mirrors what Intel's OpenVINO conversion and the Mixxx GSoC 2025 ONNX project accomplished: **strip STFT/iSTFT from the neural network and implement them natively**.

```
┌─────────────────────────────────────────────────────────────┐
│                    Host Application (Swift)                  │
├─────────────────────────────────────────────────────────────┤
│  1. STFT (vDSP/Accelerate)                                  │
│     - Input: Raw audio waveform (Float32)                   │
│     - Output: Magnitude + Phase OR Real/Imag channels       │
│     - Use vDSP_fft_zrip for forward FFT                     │
├─────────────────────────────────────────────────────────────┤
│  2. CoreML Model (Neural Network portion only)              │
│     - Input: Real/Imaginary channels (stacked)              │
│     - Output: 6 separation masks                            │
│     - Runs on ANE/GPU/CPU (hybrid)                          │
├─────────────────────────────────────────────────────────────┤
│  3. Mask Application + iSTFT (vDSP/Accelerate)              │
│     - Apply masks to complex spectrogram                    │
│     - Overlap-add reconstruction                            │
│     - Output: 6 separated audio stems                       │
└─────────────────────────────────────────────────────────────┘
```

### Modifying Demucs for CoreML conversion

The Mixxx GSoC 2025 project provides a reference for rewriting STFT/iSTFT using **real-valued convolutions** instead of complex FFT. Their approach achieved **17.94% faster** CPU inference than PyTorch with SI-SDR difference of less than **0.1 dB**—essentially equivalent audio quality.

The key technique stores the DFT matrix as 1D convolution weights. Adobe Research's **convmelspec** library provides a ready implementation:

```python
from convmelspec.stft import ConvertibleSpectrogram as Spectrogram

melspec = Spectrogram(sr=44100, n_fft=4096, hop_size=1024)
melspec.set_mode("DFT", "store")  # Fastest: stores DFT matrix as weights

traced = torch.jit.trace(melspec, example_audio)
mlmodel = ct.convert(traced, inputs=[ct.TensorType(shape=input_shape)])
```

However, convmelspec only handles forward STFT. For the inverse transform, you'll need to implement overlap-add reconstruction in Swift using vDSP, or contribute an iSTFT implementation to the library.

## Implementing STFT/iSTFT natively in Swift

The Accelerate framework provides efficient FFT operations that maintain Float32 precision:

```swift
import Accelerate

class AudioFFT {
    private var fftSetup: FFTSetup
    private let log2n: vDSP_Length
    private let fftSize: Int
    private let hopLength: Int
    private var window: [Float]
    
    init(fftSize: Int = 4096, hopLength: Int = 1024) {
        self.fftSize = fftSize
        self.hopLength = hopLength
        self.log2n = vDSP_Length(log2(Double(fftSize)))
        self.fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
        
        // Create Hann window (matching Demucs)
        self.window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
    }
    
    func stft(_ audio: [Float]) -> (real: [[Float]], imag: [[Float]]) {
        // Window, FFT, and organize into frames
        // Returns real and imaginary components separately
    }
    
    func istft(real: [[Float]], imag: [[Float]]) -> [Float] {
        // Inverse FFT each frame
        // Apply overlap-add reconstruction
        // Apply COLA normalization
    }
}
```

For GPU-accelerated FFT, **MPSGraph now supports batched FFT operations** (introduced WWDC 2024). This enables fully GPU-accelerated audio ML pipelines without falling back to CPU for transforms.

## Handling the transformer layers

### Preparing the model for tracing

```python
import torch
import coremltools as ct

# CRITICAL: Disable fast attention path before tracing
torch.backends.mha.set_fastpath_enabled(False)

model = load_htdemucs_model()
model.eval()

# Remove STFT/iSTFT operations (model surgery required)
# The inner model processes spectrograms directly
inner_model = extract_inner_model(model)

with torch.no_grad():
    traced_model = torch.jit.trace(inner_model, example_spectrogram)
```

### CoreML conversion with precision controls

```python
# Define precision-sensitive ops that should stay in FP32
PRECISION_SENSITIVE_OPS = {
    "pow", "real_div", "l2_norm", "sqrt", 
    "reduce_mean", "reduce_sum", "softmax"
}

def op_selector(op):
    return op.op_type not in PRECISION_SENSITIVE_OPS

mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="spectrogram_real", shape=(1, 2, 2049, 431)),
        ct.TensorType(name="spectrogram_imag", shape=(1, 2, 2049, 431))
    ],
    outputs=[
        ct.TensorType(name="mask_drums"),
        ct.TensorType(name="mask_bass"),
        ct.TensorType(name="mask_vocals"),
        ct.TensorType(name="mask_guitar"),
        ct.TensorType(name="mask_piano"),
        ct.TensorType(name="mask_other")
    ],
    compute_precision=ct.transform.FP16ComputePrecision(op_selector=op_selector),
    compute_units=ct.ComputeUnit.CPU_AND_GPU,  # Exclude ANE for precision
    minimum_deployment_target=ct.target.iOS18  # Required for SDPA support
)
```

## Achieving near-bit-exact parity

**True bit-exact parity is mathematically impossible** due to floating-point non-associativity and implementation differences. However, you can achieve near-parity sufficient for audio quality.

### Precision targets by compute unit

| Compute Unit | Native Precision | Typical SNR vs PyTorch |
|--------------|-----------------|------------------------|
| Neural Engine | FP16 only | 60-80 dB |
| GPU | FP16/FP32 | 70-90 dB |
| CPU | FP32 | 90-100 dB |

### Layer normalization is the primary precision concern

LayerNorm operations use `pow` and `sqrt` which can overflow in FP16 (max value 65,504). The solution is selective mixed precision:

```python
layer_norm_tracker = False

def layernorm_aware_selector(op):
    global layer_norm_tracker
    if op.op_type == "pow":
        layer_norm_tracker = True
    is_fp16 = not layer_norm_tracker
    if op.op_type == "sqrt":
        layer_norm_tracker = False
    return is_fp16

mlmodel = ct.convert(
    traced_model,
    compute_precision=ct.transform.FP16ComputePrecision(
        op_selector=layernorm_aware_selector
    )
)
```

### Validation methodology

```python
def validate_parity(pytorch_model, coreml_model, test_inputs, 
                    rtol=1e-3, atol=1e-4):
    with torch.no_grad():
        pt_out = pytorch_model(test_inputs).numpy()
    
    coreml_out = coreml_model.predict({"input": test_inputs.numpy()})
    
    max_diff = np.max(np.abs(pt_out - coreml_out))
    snr = 10 * np.log10(np.mean(pt_out**2) / np.mean((pt_out - coreml_out)**2))
    
    # For audio: SNR > 60dB generally indistinguishable
    # For masks: relative tolerance of 1e-3 acceptable
    return {
        "max_absolute_diff": max_diff,
        "snr_db": snr,
        "within_tolerance": np.allclose(pt_out, coreml_out, rtol=rtol, atol=atol)
    }
```

## Memory management and chunking for iOS

### Device memory constraints

| Device | RAM | Practical App Limit | Recommendation |
|--------|-----|---------------------|----------------|
| iPhone 12/13 | 4GB | ~1.5GB | 5-7 sec chunks, quantized model |
| iPhone 14/15 Pro | 6GB | ~2-3GB | 8-10 sec chunks, FP16 model |
| iPad Pro M-series | 8-16GB | ~5GB | 10-12 sec chunks, full precision |

### Chunking implementation

HT Demucs uses **10-second segments** with **25% overlap** during inference. For iOS, implement overlap-add reconstruction:

```swift
class DemucsProcessor {
    let chunkDuration: Float = 10.0
    let overlapDuration: Float = 1.0
    let sampleRate: Int = 44100
    
    func process(audio: [Float]) async -> [[Float]] {  // 6 stems
        let chunkSamples = Int(chunkDuration * Float(sampleRate))
        let overlapSamples = Int(overlapDuration * Float(sampleRate))
        let stride = chunkSamples - 2 * overlapSamples
        
        var outputs = [[Float]](repeating: [Float](repeating: 0, count: audio.count), count: 6)
        
        for start in stride(from: 0, to: audio.count, by: stride) {
            let chunk = Array(audio[start..<min(start + chunkSamples, audio.count)])
            
            // STFT
            let (real, imag) = stft(chunk)
            
            // CoreML inference
            let masks = try await coremlModel.prediction(
                spectrogram_real: real, 
                spectrogram_imag: imag
            )
            
            // Apply masks and iSTFT for each stem
            for (i, mask) in masks.enumerated() {
                let stem = istft(real: real * mask, imag: imag * mask)
                
                // Linear fade for overlap regions
                let fadedStem = applyFade(stem, 
                    fadeInLength: overlapSamples, 
                    fadeOutLength: overlapSamples,
                    isFirst: start == 0,
                    isLast: start + stride >= audio.count
                )
                
                // Overlap-add
                for (j, sample) in fadedStem.enumerated() {
                    outputs[i][start + j] += sample
                }
            }
        }
        
        return outputs
    }
}
```

### Memory optimization techniques

Apple recommends **6-bit palettization** for Neural Engine deployment, reducing model size by 5x while maintaining quality. For htdemucs (~80MB weights), this yields ~16MB on-device:

```python
from coremltools.optimize.coreml import (
    OpPalettizerConfig, 
    OptimizationConfig,
    palettize_weights
)

config = OptimizationConfig(
    global_config=OpPalettizerConfig(mode="kmeans", nbits=6)
)
compressed_model = palettize_weights(mlmodel, config)
```

## Alternative approaches if direct conversion proves intractable

### ONNX Runtime with CoreML Execution Provider

Rather than converting ONNX to CoreML, use ONNX Runtime directly with CoreML as the backend. The Mixxx GSoC project's ONNX model works with this approach:

```cpp
Ort::SessionOptions so;
std::unordered_map<std::string, std::string> provider_options;
provider_options["ModelFormat"] = "MLProgram";
provider_options["MLComputeUnits"] = "ALL";
so.AppendExecutionProvider("CoreML", provider_options);

Ort::Session session(env, "demucs.onnx", so);
```

**Caveat**: Only ~25% of transformer operations may actually run on CoreML; the rest fall back to CPU. This is documented in ONNX Runtime GitHub issue #19887.

### demucs.cpp with Metal acceleration

The demucs.cpp project (github.com/sevagh/demucs.cpp) provides a low-memory C++ implementation using Eigen. Compiling for iOS with Metal compute shaders for matrix operations bypasses CoreML entirely while achieving good performance:

- Designed for memory-constrained environments
- Supports htdemucs, htdemucs_6s, htdemucs_ft variants
- Already deployed in Android app (freemusicdemixer.com)
- WebAssembly version exists, demonstrating portability

### Hybrid pipeline: CoreML for CNN, native for transformer

Given transformer conversion challenges, consider:
1. Convert only the CNN encoder/decoder portions to CoreML
2. Implement the 5-layer transformer in optimized Swift/Metal
3. Shuttle data between CoreML and native code

This trades implementation complexity for guaranteed precision and performance characteristics.

## Reference implementations and resources

### Essential repositories

- **facebook/demucs** — Original PyTorch implementation (archived Jan 2025)
- **sevagh/demucs.cpp** — C++ Eigen implementation, low-memory design
- **sevagh/demucs.onnx** — ONNX Runtime inference, STFT stripped
- **mixxxdj/demucs** — Mixxx fork with full ONNX export (STFT rewritten)
- **adobe-research/convmelspec** — Convertible STFT via 1D convolutions
- **apple/ml-ane-transformers** — Apple's ANE-optimized transformer patterns
- **ggml-org/whisper.cpp** — Reference for CoreML audio encoder integration

### Key documentation

- coremltools supported operations: Lists all convertible PyTorch ops
- Apple Technical Note TN3151: ANE deployment debugging
- WWDC 2024 "Accelerate machine learning with Metal": MPSGraph FFT
- arXiv:2211.08553: Original HT Demucs paper with architecture details

## Recommended development path

1. **Start with the Mixxx modified Demucs** that has STFT/iSTFT rewritten as real-valued convolutions
2. **Extract the inner model** (post-STFT to pre-iSTFT) for CoreML conversion
3. **Target iOS 18+** for scaled_dot_product_attention support
4. **Use CPU_AND_GPU compute units** initially (exclude ANE for precision debugging)
5. **Implement STFT/iSTFT in Swift** using vDSP for Float32 precision
6. **Validate with multiple audio samples** at various lengths and genres
7. **Apply 6-bit palettization** once functional for deployment optimization
8. **Test on lowest-spec target device** (iPhone 12 mini with 4GB RAM)

The path to deployment is challenging but tractable. The key insight is that complex number support isn't coming to CoreML—the solution is architectural, moving FFT operations outside the neural network boundary where they belong in signal processing terms anyway.