"""CoreML conversion utilities for HTDemucs models.

This module provides tools for converting PyTorch HTDemucs models to CoreML format,
with specific handling for precision-sensitive operations that need to remain in FP32
while other operations can be quantized to FP16 for model size reduction.
"""

import torch
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
