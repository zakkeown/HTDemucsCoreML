# HTDemucs CoreML Models

## htdemucs_6s.mlpackage

**Source:** Facebook Research Demucs (converted from PyTorch)
**Original Model:** [facebookresearch/demucs](https://github.com/facebookresearch/demucs)
**Version:** HTDemucs v4 (6-stem)
**License:** MIT License (see [Demucs repository](https://github.com/facebookresearch/demucs/blob/main/LICENSE))
**Size:** 59 MB

### Model Details

- **Architecture:** Hybrid Transformer Demucs (HTDemucs)
- **Stems:** 6-stem separation
  - drums
  - bass
  - vocals
  - other
  - piano
  - guitar
- **Input:** Stereo spectrogram in Complex-as-Channels format `[1, 4, 2049, 431]`
  - Channels: `[real_L, imag_L, real_R, imag_R]`
  - Frequency bins: 2049 (from 4096-point FFT)
  - Time frames: 431 (approximately 10 seconds at 44.1kHz with 1024 hop length)
- **Output:** 6 stem masks in CaC format `[1, 6, 4, 2049, 431]`
  - 6 sources
  - Stereo (4 channels in CaC format per source)
  - Same frequency/time dimensions as input
- **Sample Rate:** 44.1 kHz
- **STFT Configuration:**
  - FFT size: 4096
  - Hop length: 1024
  - Window: Hann

### Conversion Details

This model was converted using the project's conversion pipeline:
- **Script:** `scripts/convert_htdemucs.py`
- **Converter:** `src/htdemucs_coreml/coreml_converter.py`
- **Precision Strategy:**
  - Sensitive operations (pow, sqrt, div, norm, softmax, matmul): FP32
  - Other operations: FP16 for compression
- **Compute Units:** ALL (CPU + Neural Engine)
- **Deployment Target:** iOS 18
- **Format:** ML Program (.mlpackage)

### Usage

The model is loaded automatically by the `ModelLoader` class:

```swift
let loader = ModelLoader(modelName: "htdemucs_6s")
let model = try await loader.loadModel()
```

The model expects spectrograms in Complex-as-Channels format from the `AudioFFT` STFT implementation and outputs separation masks that are applied by the `SeparationPipeline`.

### Quality

The CoreML conversion maintains high quality:
- **SNR:** > 60 dB (perceptually identical to PyTorch)
- **Tolerance:** Within rtol=1e-3, atol=1e-4
- All validation tests passed during Phase 1

### Updating the Model

To regenerate or update the model:

```bash
cd /path/to/HTDemucsCoreML
python3 scripts/convert_htdemucs.py \
  --output Resources/Models/htdemucs_6s.mlpackage \
  --compute-units ALL
```

This will:
1. Download the latest htdemucs_6s model from PyTorch Hub (cached in `~/.cache/torch/hub/`)
2. Extract the inner neural network (encoder/decoder without STFT/iSTFT)
3. Trace to TorchScript
4. Convert to CoreML with mixed precision
5. Save as .mlpackage

### References

- [HTDemucs Paper (arXiv:2211.08553)](https://arxiv.org/abs/2211.08553) - "Hybrid Transformers for Music Source Separation"
- [Demucs GitHub Repository](https://github.com/facebookresearch/demucs)
- [CoreML Tools Documentation](https://coremltools.readme.io/)
- Project conversion pipeline: `docs/plans/2026-02-01-phase1-python-foundation.md`
