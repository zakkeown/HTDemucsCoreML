"""Numerical validation utilities for comparing PyTorch vs CoreML outputs.

This module provides tools for validating the numerical accuracy of CoreML models
compared to their PyTorch counterparts. It computes various metrics to quantify
the differences between outputs, including absolute differences, relative errors,
and signal-to-noise ratios.
"""

from dataclasses import dataclass
from typing import Tuple
import torch
import numpy as np


@dataclass
class ValidationMetrics:
    """Metrics for numerical comparison between PyTorch and CoreML outputs.

    Attributes:
        max_absolute_diff: Maximum absolute difference between outputs.
        mean_absolute_diff: Mean absolute difference between outputs.
        mean_relative_error: Mean relative error (MAD / mean absolute reference).
        snr_db: Signal-to-noise ratio in decibels (higher is better).
        within_tolerance: Boolean indicating if differences are within specified tolerances.
    """

    max_absolute_diff: float
    mean_absolute_diff: float
    mean_relative_error: float
    snr_db: float
    within_tolerance: bool


def compute_numerical_diff(
    pytorch_output: torch.Tensor,
    coreml_output: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> ValidationMetrics:
    """Compute numerical differences between PyTorch and CoreML outputs.

    This function computes various metrics to quantify the differences between
    PyTorch and CoreML model outputs, useful for validating conversion accuracy.

    Args:
        pytorch_output: Output tensor from PyTorch model.
        coreml_output: Output tensor from CoreML model (as PyTorch tensor).
        rtol: Relative tolerance for allclose check (default 1e-5).
        atol: Absolute tolerance for allclose check (default 1e-7).

    Returns:
        ValidationMetrics dataclass containing:
            - max_absolute_diff: Maximum |pytorch - coreml|
            - mean_absolute_diff: Mean |pytorch - coreml|
            - mean_relative_error: Mean |pytorch - coreml| / mean(|pytorch|)
            - snr_db: Signal-to-noise ratio in dB
            - within_tolerance: Whether torch.allclose(pytorch, coreml, rtol, atol)

    Notes:
        SNR is computed as: 10 * log10(signal_power / noise_power) where:
        - signal_power = mean(pytorch_output^2)
        - noise_power = mean((pytorch_output - coreml_output)^2)
    """
    # Ensure tensors are on CPU for computation
    pytorch_output = pytorch_output.cpu().detach().float()
    coreml_output = coreml_output.cpu().detach().float()

    # Compute absolute differences
    abs_diff = torch.abs(pytorch_output - coreml_output)
    max_absolute_diff = float(torch.max(abs_diff).item())
    mean_absolute_diff = float(torch.mean(abs_diff).item())

    # Compute relative error (handle division by zero gracefully)
    # Relative error = |diff| / |reference|, where reference is pytorch_output
    pytorch_abs = torch.abs(pytorch_output)
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-8
    relative_errors = abs_diff / (pytorch_abs + epsilon)
    mean_relative_error = float(torch.mean(relative_errors).item())

    # Compute SNR (Signal-to-Noise Ratio) in dB
    # SNR = 10 * log10(signal_power / noise_power)
    # Signal power: mean(pytorch_output^2)
    # Noise power: mean((pytorch_output - coreml_output)^2)
    signal_power = torch.mean(pytorch_output**2).item()
    noise_power = torch.mean((pytorch_output - coreml_output)**2).item()

    # Handle edge cases where noise power is zero or signal power is zero
    if noise_power < epsilon:
        # Perfect match or nearly perfect
        snr_db = 100.0  # Return a very high value
    elif signal_power < epsilon:
        # Signal is essentially zero, use mean absolute diff as metric
        snr_db = -10.0 * np.log10(noise_power + epsilon)
    else:
        snr_db = 10.0 * np.log10(signal_power / (noise_power + epsilon))

    # Check if within tolerance
    within_tolerance = torch.allclose(
        pytorch_output, coreml_output, rtol=rtol, atol=atol
    )

    return ValidationMetrics(
        max_absolute_diff=max_absolute_diff,
        mean_absolute_diff=mean_absolute_diff,
        mean_relative_error=mean_relative_error,
        snr_db=snr_db,
        within_tolerance=within_tolerance,
    )
