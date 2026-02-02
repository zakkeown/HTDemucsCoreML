#!/usr/bin/env python3
"""CLI tool for converting HTDemucs models to CoreML format.

This script loads the HTDemucs model, extracts the inner neural network,
converts it to TorchScript, and then to CoreML format for iOS deployment.

Usage:
    python scripts/convert_htdemucs.py --output model.mlmodel
    python scripts/convert_htdemucs.py --output model.mlmodel --compute-units CPU_ONLY
    python scripts/convert_htdemucs.py --output model.mlpackage --compute-units ALL
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

# Add src directory to path to import our modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from htdemucs_coreml.model_surgery import extract_inner_model
from htdemucs_coreml.coreml_converter import trace_inner_model, convert_to_coreml


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with output path and compute units options.
    """
    parser = argparse.ArgumentParser(
        description="Convert HTDemucs model to CoreML format for iOS deployment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to CoreML with default settings (ALL compute units)
  python scripts/convert_htdemucs.py --output htdemucs.mlmodel

  # Convert with CPU-only compute units
  python scripts/convert_htdemucs.py --output htdemucs.mlmodel --compute-units CPU_ONLY

  # Convert to mlpackage format
  python scripts/convert_htdemucs.py --output htdemucs.mlpackage --compute-units ALL
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for the CoreML model (.mlmodel or .mlpackage)",
    )

    parser.add_argument(
        "--compute-units",
        "-c",
        type=str,
        default="ALL",
        choices=["ALL", "CPU_ONLY", "CPU_AND_NE"],
        help="Compute units for CoreML model (default: ALL)",
    )

    return parser.parse_args()


def print_summary(output_path: str, compute_units: str) -> None:
    """Print a summary of the conversion.

    Args:
        output_path: Path where the CoreML model was saved.
        compute_units: Compute units used during conversion.
    """
    output_path_obj = Path(output_path)
    file_size_mb = output_path_obj.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print("CoreML Conversion Summary")
    print("=" * 60)
    print(f"Model:                HTDemucs (htdemucs_6s)")
    print(f"Output Path:          {output_path}")
    print(f"Output Format:        {output_path_obj.suffix}")
    print(f"File Size:            {file_size_mb:.2f} MB")
    print(f"Compute Units:        {compute_units}")
    print(f"Deployment Target:    iOS 18")
    print("-" * 60)
    print("Model Specifications:")
    print(f"  Input Shape:        (1, 4, 2049, 431)")
    print(f"  Input Format:       Stereo CaC spectrogram [real_L, imag_L, real_R, imag_R]")
    print(f"  Output Shape:       (1, 6, 4, 2049, 431)")
    print(f"  Output Format:      Separation masks (6 sources, stereo CaC)")
    print(f"  Sources:            [drums, bass, vocals, guitar, piano, other]")
    print("-" * 60)
    print("Precision Configuration:")
    print(f"  Sensitive Ops:      FP32 (pow, sqrt, div, norm, softmax, matmul)")
    print(f"  Other Ops:          FP16 (for model compression)")
    print("=" * 60 + "\n")


def load_model():
    """Load the HTDemucs model.

    Returns:
        The loaded HTDemucs model.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    try:
        from demucs.pretrained import get_model
        print("Loading htdemucs_6s model...")
        model = get_model('htdemucs_6s')
        model.eval()
        print("✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}", file=sys.stderr)
        print("  Ensure you have an internet connection for the first run.", file=sys.stderr)
        print("  The model will be downloaded to ~/.cache/torch/hub/", file=sys.stderr)
        raise RuntimeError(f"Failed to load HTDemucs model: {e}") from e


def extract_model(model):
    """Extract the inner neural network from HTDemucs.

    Args:
        model: The HTDemucs model (may be BagOfModels wrapper).

    Returns:
        The extracted InnerHTDemucs model.
    """
    try:
        print("Extracting inner neural network...")
        # Handle both BagOfModels wrapper and raw HTDemucs
        if hasattr(model, 'models'):
            htdemucs_model = model.models[0]
        else:
            htdemucs_model = model

        inner_model = extract_inner_model(htdemucs_model)
        print("✓ Inner model extracted successfully")
        return inner_model
    except Exception as e:
        print(f"✗ Error extracting inner model: {e}", file=sys.stderr)
        raise RuntimeError(f"Failed to extract inner model: {e}") from e


def trace_model(inner_model):
    """Trace the inner model to TorchScript.

    Args:
        inner_model: The InnerHTDemucs model to trace.

    Returns:
        The traced TorchScript model.
    """
    try:
        print("Tracing model to TorchScript...")
        traced_model = trace_inner_model(inner_model)
        print("✓ Model traced successfully")
        return traced_model
    except Exception as e:
        print(f"✗ Error tracing model: {e}", file=sys.stderr)
        raise RuntimeError(f"Failed to trace model: {e}") from e


def convert_model(
    traced_model: torch.jit.ScriptModule,
    output_path: str,
    compute_units: str,
) -> None:
    """Convert traced model to CoreML format.

    Args:
        traced_model: The TorchScript-traced model.
        output_path: Path where the CoreML model will be saved.
        compute_units: Compute units configuration for CoreML.

    Raises:
        RuntimeError: If conversion fails.
    """
    try:
        output_path_obj = Path(output_path)

        # Validate output path extension
        if output_path_obj.suffix not in [".mlmodel", ".mlpackage"]:
            raise ValueError(
                f"Output path must end with .mlmodel or .mlpackage, "
                f"got {output_path_obj.suffix}"
            )

        print(f"Converting to CoreML format ({output_path_obj.suffix})...")
        convert_to_coreml(traced_model, output_path, compute_units=compute_units)
        print("✓ Model converted successfully")

        # Print summary
        print_summary(output_path, compute_units)

    except ValueError as e:
        print(f"✗ Validation error: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"✗ Error converting model: {e}", file=sys.stderr)
        raise RuntimeError(f"Failed to convert model to CoreML: {e}") from e


def main():
    """Main entry point for the CLI tool."""
    try:
        # Parse arguments
        args = parse_arguments()

        print("\n" + "=" * 60)
        print("HTDemucs to CoreML Converter")
        print("=" * 60 + "\n")

        # Load model
        model = load_model()

        # Extract inner model
        inner_model = extract_model(model)

        # Trace to TorchScript
        traced_model = trace_model(inner_model)

        # Convert to CoreML
        convert_model(traced_model, args.output, args.compute_units)

        print("✓ Conversion completed successfully!\n")
        return 0

    except KeyboardInterrupt:
        print("\n✗ Conversion cancelled by user")
        return 130
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
