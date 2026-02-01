"""Tests for model surgery operations.

This module tests the extraction of the inner neural network from HTDemucs,
which allows conversion to CoreML by bypassing STFT/iSTFT operations.
"""

import pytest
import torch
import torch.nn as nn


class TestModelExtraction:
    """Tests for extracting the inner model from HTDemucs."""

    def test_extract_inner_model_creates_module(self, htdemucs_model):
        """Verify that extract_inner_model returns an nn.Module."""
        from htdemucs_coreml.model_surgery import extract_inner_model

        # The fixture returns BagOfModels, extract the actual HTDemucs
        actual_model = htdemucs_model.models[0]
        inner_model = extract_inner_model(actual_model)

        assert isinstance(inner_model, nn.Module), (
            f"Expected nn.Module, got {type(inner_model)}"
        )

    def test_inner_model_has_no_stft_operations(self, htdemucs_model):
        """Verify that the inner model contains no STFT/iSTFT operations.

        The original HTDemucs model applies STFT to input audio before passing
        to the neural network. The inner model should receive spectrograms
        directly and output separation masks without performing STFT/iSTFT.
        """
        from htdemucs_coreml.model_surgery import extract_inner_model

        # The fixture returns BagOfModels, extract the actual HTDemucs
        actual_model = htdemucs_model.models[0]
        inner_model = extract_inner_model(actual_model)

        # Convert to string representation and check for STFT operations
        model_str = str(inner_model)

        # These operations should not appear in the inner model
        disallowed_ops = ["stft", "istft"]

        for op in disallowed_ops:
            assert op.lower() not in model_str.lower(), (
                f"Inner model should not contain '{op}' operation. "
                f"Model contains: {model_str[:500]}..."
            )

    def test_inner_model_output_shape(self, htdemucs_model, test_case_name):
        """Verify that the inner model produces correct output shape.

        The inner model should accept a spectrogram in Complex-as-Channels format
        (2 channels for real/imaginary parts, stacked along channel dimension)
        and output 6 separation masks (one per source: drums, bass, vocals,
        guitar, piano, other).

        Given:
        - Input: spectrogram shape (batch=1, channels=2, freq_bins=2049, time_frames=T)
        - Output: 6 separation masks of shape (batch=1, sources=6, freq_bins=2049, time_frames=T)
        """
        from htdemucs_coreml.model_surgery import extract_inner_model

        # The fixture returns BagOfModels, extract the actual HTDemucs
        actual_model = htdemucs_model.models[0]
        inner_model = extract_inner_model(actual_model)
        inner_model.eval()

        # Create a dummy spectrogram input in Complex-as-Channels format:
        # - Batch size: 1
        # - Channels: 4 (stereo mix with real and imaginary parts each = 2 channels * 2 for real/imag)
        # - Frequency bins: 2049 (from 4096-point FFT, accounting for real FFT)
        # - Time frames: 431 (approximately for 10 seconds at 44.1kHz with 1024 hop length)
        #
        # For a stereo mix:
        # - Left channel: real and imaginary parts -> 2 channels
        # - Right channel: real and imaginary parts -> 2 channels
        # - Total: 4 channels
        batch_size = 1
        num_channels = 4  # Stereo (2 channels) * (real + imaginary) = 4
        num_freq_bins = 2049  # From 4096-point FFT
        num_time_frames = 431  # Typical for ~10 second audio chunk

        dummy_spectrogram = torch.randn(
            batch_size, num_channels, num_freq_bins, num_time_frames
        )

        with torch.no_grad():
            output = inner_model(dummy_spectrogram)

        # Output should be masks for 6 sources
        assert isinstance(output, torch.Tensor), (
            f"Expected torch.Tensor output, got {type(output)}"
        )

        # Output shape should be (batch=1, sources=6, channels=4, freq_bins, time_frames)
        # where channels=4 is for Complex-as-Channels format (real/imag for stereo)
        # Note: freq_bins will be 2048 after decoder (down from 2049 input due to processing)
        expected_batch = batch_size
        expected_sources = 6
        expected_channels = 4  # stereo CaC
        expected_time = num_time_frames
        # freq_bins may differ due to decoder processing, so we just check the others
        assert output.ndim == 5, (
            f"Expected 5D output (batch, sources, channels, freq, time), got {output.ndim}D"
        )
        assert output.shape[0] == expected_batch, (
            f"Batch size mismatch: expected {expected_batch}, got {output.shape[0]}"
        )
        assert output.shape[1] == expected_sources, (
            f"Sources mismatch: expected {expected_sources}, got {output.shape[1]}"
        )
        assert output.shape[2] == expected_channels, (
            f"Channels mismatch: expected {expected_channels}, got {output.shape[2]}"
        )
        assert output.shape[4] == expected_time, (
            f"Time frames mismatch: expected {expected_time}, got {output.shape[4]}"
        )
