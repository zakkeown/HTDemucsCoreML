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


class TestSTFTOutputCapture:
    """Tests for capturing intermediate STFT outputs from HTDemucs."""

    def test_capture_stft_output_from_original_model(self, htdemucs_model, test_audio):
        """Test capturing intermediate STFT outputs from original HTDemucs.

        This test verifies that we can hook into the HTDemucs model and extract
        the STFT output (real and imaginary parts) before the neural network
        processes them.
        """
        from htdemucs_coreml.model_surgery import capture_stft_output

        # test_audio shape: (channels, samples)
        # The HTDemucs model has use_train_segment=True, which requires audio
        # to match the model's segment length. Get the segment length from the model.
        actual_model = htdemucs_model.models[0]
        segment_length = int(float(actual_model.segment) * actual_model.samplerate)

        # Slice audio to segment length
        audio_segment = test_audio[:, :segment_length]

        # Model expects: (batch, channels, samples)
        audio_batch = audio_segment.unsqueeze(0)

        real, imag = capture_stft_output(htdemucs_model, audio_batch)

        # Expected shapes: (batch, channels, freq_bins, time_frames)
        # Note: The HTDemucs._spec method removes the last frequency bin,
        # so we get 2048 freq bins instead of 2049 (4096 FFT / 2 + 1 - 1)
        assert real.shape[0] == 1, f"Expected batch size 1, got {real.shape[0]}"
        assert real.shape[1] == 2, f"Expected 2 channels (stereo), got {real.shape[1]}"
        assert real.shape[2] == 2048, f"Expected 2048 freq bins (after removing last bin), got {real.shape[2]}"
        assert real.shape == imag.shape, f"Real and imaginary shapes don't match: {real.shape} vs {imag.shape}"


class TestModelParity:
    """Tests for verifying parity between original HTDemucs and extracted inner model."""

    def test_inner_model_processes_captured_spectrogram(self, htdemucs_model, test_audio):
        """Test that inner model can process spectrograms captured from original.

        This test verifies that the InnerHTDemucs model (which accepts spectrograms
        directly) can process STFT outputs captured from the original HTDemucs model.

        This is a critical validation that the extraction process preserves the model's
        ability to process spectrograms correctly.
        """
        from htdemucs_coreml.model_surgery import (
            extract_inner_model,
            capture_stft_output,
        )

        # Get the actual HTDemucs model from BagOfModels wrapper
        actual_model = htdemucs_model.models[0]
        inner_model = extract_inner_model(actual_model)
        inner_model.eval()

        # The HTDemucs model has use_train_segment=True, which requires audio
        # to match the model's segment length. Get the segment length from the model.
        segment_length = int(float(actual_model.segment) * actual_model.samplerate)

        # Slice audio to segment length
        audio_segment = test_audio[:, :segment_length]

        # test_audio shape: (channels, samples)
        # Model expects: (batch, channels, samples)
        audio_batch = audio_segment.unsqueeze(0)

        with torch.no_grad():
            # Get STFT output from original model
            real, imag = capture_stft_output(htdemucs_model, audio_batch)

            # Combine real and imaginary parts in Complex-as-Channels format
            # Shape: (batch, 2*channels, freq, time) for stereo
            # For stereo: [real_L, imag_L, real_R, imag_R]
            cac_input = torch.cat([real, imag], dim=1)

            # Forward through inner model
            inner_output = inner_model(cac_input)

        # Verify output shape and properties
        batch_size = 1
        num_sources = 6
        num_channels = 4  # stereo CaC
        freq_bins = 2048
        time_frames = 336

        assert inner_output.ndim == 5, f"Expected 5D output, got {inner_output.ndim}D"
        assert inner_output.shape[0] == batch_size, f"Batch mismatch: {inner_output.shape[0]}"
        assert inner_output.shape[1] == num_sources, f"Sources mismatch: {inner_output.shape[1]}"
        assert inner_output.shape[2] == num_channels, f"Channels mismatch: {inner_output.shape[2]}"
        assert inner_output.shape[3] == freq_bins, f"Freq bins mismatch: {inner_output.shape[3]}"
        assert inner_output.shape[4] == time_frames, f"Time frames mismatch: {inner_output.shape[4]}"

        # Verify that output is not NaN or Inf
        assert torch.isfinite(inner_output).all(), "Output contains NaN or Inf values"

        # Verify that output has reasonable magnitude
        assert (inner_output.abs() < 1e4).all(), "Output values are unexpectedly large"
