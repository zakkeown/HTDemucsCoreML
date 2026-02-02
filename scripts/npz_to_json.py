#!/usr/bin/env python3
"""
Load NumPy .npz fixture and output as compact JSON.

Usage:
    python npz_to_json.py <path_to_npz_file>

Output:
    JSON with keys: audio (1D array), stft_real (2D array), stft_imag (2D array)
"""

import sys
import json
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Error: Expected path to .npz file", file=sys.stderr)
        sys.exit(1)

    npz_path = sys.argv[1]

    try:
        # Load .npz file
        data = np.load(npz_path)

        # Extract arrays
        audio = data['audio'].tolist()
        stft_real = data['stft_real'].tolist()
        stft_imag = data['stft_imag'].tolist()

        # Output compact JSON to stdout
        result = {
            'audio': audio,
            'stft_real': stft_real,
            'stft_imag': stft_imag
        }

        json.dump(result, sys.stdout, separators=(',', ':'))

    except FileNotFoundError:
        print(f"Error: File not found: {npz_path}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key in .npz file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
