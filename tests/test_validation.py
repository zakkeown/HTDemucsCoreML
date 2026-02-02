"""Tests for numerical validation utilities.

This module tests the numerical comparison and validation metrics for comparing
PyTorch outputs with CoreML outputs, ensuring numerical accuracy across
different precision levels.
"""

import pytest
import torch
import numpy as np
from dataclasses import is_dataclass


class TestNumericalValidation:
    """Tests for numerical comparison and validation utilities."""

    def test_compute_numerical_diff_identical_tensors(self):
        """Verify that identical tensors produce zero differences.

        When comparing identical tensors, all numerical metrics should reflect
        perfect agreement (zero differences, SNR should be very high).
        """
        from htdemucs_coreml.validation import compute_numerical_diff

        # Create identical tensors
        tensor = torch.randn(10, 20, 30)
        pytorch_output = tensor
        coreml_output = tensor

        metrics = compute_numerical_diff(pytorch_output, coreml_output)

        # Verify all metrics reflect perfect match
        assert metrics.max_absolute_diff == 0.0, (
            f"Max absolute diff should be 0 for identical tensors, got {metrics.max_absolute_diff}"
        )
        assert metrics.mean_absolute_diff == 0.0, (
            f"Mean absolute diff should be 0 for identical tensors, got {metrics.mean_absolute_diff}"
        )
        assert metrics.mean_relative_error == 0.0, (
            f"Mean relative error should be 0 for identical tensors, got {metrics.mean_relative_error}"
        )
        # Very high SNR (infinite in theory, very large in practice)
        assert metrics.snr_db >= 100, (
            f"SNR should be very high for identical tensors, got {metrics.snr_db}"
        )

    def test_compute_numerical_diff_different_tensors(self):
        """Verify that different tensors produce non-zero differences.

        When comparing different tensors, metrics should reflect the differences
        in a meaningful way (positive max/mean diffs, reasonable SNR).
        """
        from htdemucs_coreml.validation import compute_numerical_diff

        # Create different tensors
        pytorch_output = torch.randn(10, 20, 30)
        coreml_output = pytorch_output + torch.randn(10, 20, 30) * 0.01

        metrics = compute_numerical_diff(pytorch_output, coreml_output)

        # Verify metrics are reasonable
        assert metrics.max_absolute_diff > 0, (
            "Max absolute diff should be positive for different tensors"
        )
        assert metrics.mean_absolute_diff > 0, (
            "Mean absolute diff should be positive for different tensors"
        )
        assert metrics.mean_relative_error >= 0, (
            "Mean relative error should be non-negative"
        )
        assert metrics.snr_db > 0, (
            "SNR should be positive"
        )
        assert metrics.max_absolute_diff >= metrics.mean_absolute_diff, (
            "Max absolute diff should be >= mean absolute diff"
        )

    def test_compute_numerical_diff_with_custom_tolerances(self):
        """Verify that custom tolerance parameters work correctly.

        The compute_numerical_diff function should accept rtol and atol parameters
        and return whether values are within tolerance.
        """
        from htdemucs_coreml.validation import compute_numerical_diff

        pytorch_output = torch.randn(5, 10)
        coreml_output = pytorch_output + 1e-6  # Small difference

        # Tight tolerance should fail
        metrics_tight = compute_numerical_diff(pytorch_output, coreml_output, rtol=1e-8, atol=1e-9)
        assert not metrics_tight.within_tolerance, (
            "Should not be within tight tolerance for small differences"
        )

        # Loose tolerance should pass
        metrics_loose = compute_numerical_diff(pytorch_output, coreml_output, rtol=1e-3, atol=1e-3)
        assert metrics_loose.within_tolerance, (
            "Should be within loose tolerance for small differences"
        )

    def test_validation_metrics_dataclass(self):
        """Verify that ValidationMetrics is a proper dataclass with expected fields.

        The ValidationMetrics should be a dataclass that can be instantiated
        and contains all required metric fields.
        """
        from htdemucs_coreml.validation import ValidationMetrics

        # Verify it's a dataclass
        assert is_dataclass(ValidationMetrics), (
            "ValidationMetrics should be a dataclass"
        )

        # Create an instance with expected values
        metrics = ValidationMetrics(
            max_absolute_diff=1e-3,
            mean_absolute_diff=5e-4,
            mean_relative_error=0.001,
            snr_db=60.0,
            within_tolerance=True,
        )

        # Verify all fields are accessible
        assert metrics.max_absolute_diff == 1e-3
        assert metrics.mean_absolute_diff == 5e-4
        assert metrics.mean_relative_error == 0.001
        assert metrics.snr_db == 60.0
        assert metrics.within_tolerance is True

    def test_compute_numerical_diff_handles_various_shapes(self):
        """Verify that compute_numerical_diff works with various tensor shapes.

        The function should handle tensors with different dimensionalities
        (1D, 2D, 3D, 4D, etc.) without errors.
        """
        from htdemucs_coreml.validation import compute_numerical_diff

        test_shapes = [
            (100,),
            (10, 20),
            (5, 10, 15),
            (2, 3, 4, 5),
            (1, 2, 3, 4, 5),
        ]

        for shape in test_shapes:
            pytorch_output = torch.randn(*shape)
            coreml_output = pytorch_output + torch.randn(*shape) * 1e-5

            # Should not raise an exception
            metrics = compute_numerical_diff(pytorch_output, coreml_output)

            # Verify metrics exist and are reasonable
            assert metrics.max_absolute_diff >= 0
            assert metrics.mean_absolute_diff >= 0
            assert metrics.mean_relative_error >= 0
            assert metrics.snr_db > 0

    def test_compute_numerical_diff_with_zeros_in_reference(self):
        """Verify correct handling when reference tensor contains zeros.

        Computing relative error when the reference contains zeros should be
        handled gracefully to avoid division by zero.
        """
        from htdemucs_coreml.validation import compute_numerical_diff

        # Create tensors with zeros
        pytorch_output = torch.zeros(5, 5)
        pytorch_output[0, 0] = 1.0
        pytorch_output[1, 1] = 2.0

        coreml_output = pytorch_output + 0.01

        # Should not raise an exception
        metrics = compute_numerical_diff(pytorch_output, coreml_output)

        # Metrics should exist and be finite
        assert np.isfinite(metrics.max_absolute_diff)
        assert np.isfinite(metrics.mean_absolute_diff)
        assert np.isfinite(metrics.mean_relative_error)
        assert np.isfinite(metrics.snr_db)

    def test_compute_numerical_diff_large_difference_detection(self):
        """Verify that large differences are properly detected and reported.

        When comparing tensors with significant differences, the metrics should
        reflect this appropriately (high error, low SNR).
        """
        from htdemucs_coreml.validation import compute_numerical_diff

        pytorch_output = torch.randn(10, 10)
        coreml_output = pytorch_output + pytorch_output * 0.5  # 50% difference

        metrics = compute_numerical_diff(pytorch_output, coreml_output)

        # Difference should be significant
        assert metrics.max_absolute_diff > 0.1, (
            "Max absolute diff should be large for 50% difference"
        )
        # Relative error should also be significant
        assert metrics.mean_relative_error > 0.1, (
            "Mean relative error should be large for 50% difference"
        )
        # SNR should be lower
        assert metrics.snr_db < 20, (
            "SNR should be lower for large differences"
        )

    def test_compute_numerical_diff_default_tolerances(self):
        """Verify that default tolerance values produce reasonable results.

        The function should have sensible default rtol and atol values that
        are appropriate for neural network comparisons.
        """
        from htdemucs_coreml.validation import compute_numerical_diff

        pytorch_output = torch.randn(10, 10)
        # Add small FP32 rounding noise
        coreml_output = pytorch_output + torch.randn(10, 10) * 1e-7

        # Should work with defaults
        metrics = compute_numerical_diff(pytorch_output, coreml_output)

        # Metrics should be reasonable
        assert metrics.max_absolute_diff >= 0
        assert metrics.mean_absolute_diff >= 0
        assert metrics.mean_relative_error >= 0
        assert metrics.snr_db > 0

    def test_compute_numerical_diff_preserves_tensor_device(self):
        """Verify that compute_numerical_diff works with tensors on different devices.

        The function should handle CPU tensors and potentially other devices
        without requiring device conversion by the caller.
        """
        from htdemucs_coreml.validation import compute_numerical_diff

        # Create CPU tensors
        pytorch_output = torch.randn(5, 5)
        coreml_output = pytorch_output + torch.randn(5, 5) * 1e-5

        # Should work without error
        metrics = compute_numerical_diff(pytorch_output, coreml_output)

        # Verify metrics exist
        assert metrics.max_absolute_diff >= 0
        assert metrics.mean_absolute_diff >= 0
