"""Tests for CoreML conversion utilities.

This module tests the CoreML conversion pipeline, including precision
selection for FP16/FP32 conversion to maintain numerical stability for
sensitive operations.
"""

import pytest
import torch
import torch.nn as nn


class TestPrecisionSelector:
    """Tests for precision selector that keeps sensitive ops in FP32."""

    def test_precision_sensitive_ops_stay_fp32(self):
        """Verify that precision-sensitive operations are marked FP32.

        Operations like pow, sqrt, division, and reductions are sensitive
        to precision loss and should remain in FP32 even when converting
        to FP16 for other operations.
        """
        from htdemucs_coreml.coreml_converter import (
            create_precision_selector,
            PRECISION_SENSITIVE_OPS,
        )

        # Create a precision selector
        selector = create_precision_selector()

        # Verify that sensitive operations are identified correctly
        sensitive_ops = [
            "pow",
            "sqrt",
            "real_div",
            "l2_norm",
            "reduce_mean",
            "reduce_sum",
            "softmax",
            "matmul",
        ]

        for op_name in sensitive_ops:
            # The selector should return FP32 for sensitive operations
            result = selector(op_name)
            assert result == torch.float32, (
                f"Operation '{op_name}' should be FP32, got {result}"
            )

        # Verify that the set is consistent with expected values
        assert len(PRECISION_SENSITIVE_OPS) >= len(sensitive_ops), (
            f"PRECISION_SENSITIVE_OPS should contain at least {len(sensitive_ops)} "
            f"operations, got {len(PRECISION_SENSITIVE_OPS)}"
        )

    def test_other_ops_use_fp16(self):
        """Verify that non-sensitive operations can use FP16.

        Most operations can safely run in FP16 to reduce model size.
        Only precision-sensitive operations should be forced to FP32.
        """
        from htdemucs_coreml.coreml_converter import (
            create_precision_selector,
            PRECISION_SENSITIVE_OPS,
        )

        # Create a precision selector
        selector = create_precision_selector()

        # Operations that are NOT precision-sensitive
        non_sensitive_ops = [
            "add",
            "mul",
            "conv2d",
            "relu",
            "batch_norm",
            "max_pool",
            "concatenate",
            "reshape",
        ]

        for op_name in non_sensitive_ops:
            # These operations should be FP16 by default
            # (or return None, indicating they can use any precision)
            result = selector(op_name)

            # For non-sensitive operations, the selector can either:
            # 1. Return FP16 (explicitly allow FP16)
            # 2. Return None (use default/auto precision)
            # Both are acceptable - we just verify it's NOT FP32
            assert result != torch.float32, (
                f"Operation '{op_name}' should not be forced to FP32, got {result}"
            )

    def test_precision_sensitive_ops_set_contains_required_operations(self):
        """Verify that PRECISION_SENSITIVE_OPS contains all required operations.

        This test ensures the set of precision-sensitive operations is complete
        and includes all operations that need FP32 precision.
        """
        from htdemucs_coreml.coreml_converter import PRECISION_SENSITIVE_OPS

        required_ops = {
            "pow",
            "sqrt",
            "real_div",
            "l2_norm",
            "reduce_mean",
            "reduce_sum",
            "softmax",
            "matmul",
        }

        missing_ops = required_ops - PRECISION_SENSITIVE_OPS

        assert len(missing_ops) == 0, (
            f"PRECISION_SENSITIVE_OPS is missing required operations: {missing_ops}"
        )

    def test_create_precision_selector_returns_callable(self):
        """Verify that create_precision_selector returns a callable."""
        from htdemucs_coreml.coreml_converter import create_precision_selector

        selector = create_precision_selector()

        assert callable(selector), (
            f"create_precision_selector should return a callable, got {type(selector)}"
        )

    def test_precision_selector_accepts_operation_name(self):
        """Verify that the selector accepts operation names as strings."""
        from htdemucs_coreml.coreml_converter import create_precision_selector

        selector = create_precision_selector()

        # Should not raise an exception for valid operation names
        result = selector("pow")
        assert result is not None, "Selector should return a value for 'pow'"

        result = selector("add")
        # For non-sensitive ops, result may be None or FP16, but shouldn't raise

    def test_precision_selector_with_complex_operation_names(self):
        """Verify the selector handles various operation naming conventions."""
        from htdemucs_coreml.coreml_converter import create_precision_selector

        selector = create_precision_selector()

        # Test various naming conventions
        test_cases = [
            ("pow", torch.float32),  # Should be sensitive
            ("sqrt", torch.float32),  # Should be sensitive
            ("matmul", torch.float32),  # Should be sensitive
            ("relu", None),  # Not sensitive - may return None or FP16
        ]

        for op_name, expected_precision in test_cases:
            result = selector(op_name)

            if expected_precision is not None:
                assert (
                    result == expected_precision
                ), f"Operation '{op_name}' returned {result}, expected {expected_precision}"
