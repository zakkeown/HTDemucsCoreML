"""CoreML conversion utilities for HTDemucs models.

This module provides tools for converting PyTorch HTDemucs models to CoreML format,
with specific handling for precision-sensitive operations that need to remain in FP32
while other operations can be quantized to FP16 for model size reduction.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional


# Operations that require FP32 precision for numerical stability
# These operations are sensitive to precision loss and should not be converted to FP16
PRECISION_SENSITIVE_OPS = {
    "pow",              # Power operations can amplify numerical errors
    "sqrt",             # Square root requires high precision
    "real_div",         # Division operations need precision
    "l2_norm",          # Normalization operations
    "reduce_mean",      # Mean reduction can lose precision
    "reduce_sum",       # Sum reduction can accumulate errors
    "softmax",          # Softmax with extreme values needs precision
    "matmul",           # Matrix multiplication benefits from FP32
}


def create_precision_selector() -> Callable[[str], Optional[torch.dtype]]:
    """Create a precision selector function for CoreML conversion.

    The returned function determines whether an operation should use FP32 or can
    safely use FP16. Precision-sensitive operations like pow, sqrt, division,
    normalization, and softmax remain in FP32 to maintain numerical stability
    and avoid potential NaN/Inf issues. Other operations can use FP16 to reduce
    model size.

    Returns:
        A callable that takes an operation name (string) and returns:
        - torch.float32 for operations that must stay in FP32
        - torch.float16 or None for operations that can use reduced precision

    Example:
        >>> selector = create_precision_selector()
        >>> selector("pow")  # torch.float32
        >>> selector("add")   # torch.float16 or None
    """

    def selector(op_name: str) -> Optional[torch.dtype]:
        """Select precision for a given operation.

        Args:
            op_name: Name of the operation (e.g., "pow", "add", "conv2d")

        Returns:
            torch.float32 if operation is precision-sensitive, otherwise torch.float16
        """
        if op_name in PRECISION_SENSITIVE_OPS:
            return torch.float32
        else:
            return torch.float16

    return selector


def trace_inner_model(inner_model: nn.Module) -> torch.jit.ScriptModule:
    """Trace an InnerHTDemucs model to create a TorchScript module.

    TorchScript tracing converts a PyTorch model into a ScriptModule that can be
    executed independently without Python. This is essential for CoreML conversion
    as it provides a static computational graph.

    The function disables fast attention to ensure tracing compatibility and uses
    example inputs matching the expected spectrogram format (batch=1, channels=2,
    freq_bins=2049, time_frames=431).

    Args:
        inner_model: An InnerHTDemucs model or compatible nn.Module that accepts
            spectrograms in Complex-as-Channels format.

    Returns:
        A torch.jit.ScriptModule that can run inference without Python.

    Example:
        >>> from htdemucs_coreml.model_surgery import extract_inner_model
        >>> from demucs.pretrained import get_model
        >>> model = get_model('htdemucs_6s')
        >>> inner_model = extract_inner_model(model.models[0])
        >>> traced_model = trace_inner_model(inner_model)
        >>> # The traced model can now be converted to CoreML
    """
    # Disable fast attention for compatibility
    torch.backends.mha.set_fastpath_enabled(False)

    # Create example inputs with the expected spectrogram shape
    # (batch, channels=2, freq_bins=2049, time_frames=431)
    example_input = torch.randn(1, 2, 2049, 431)

    # Trace the model using the example input
    # torch.jit.trace records the operations executed during the forward pass
    traced_model = torch.jit.trace(inner_model, example_input)

    return traced_model
