#!/bin/bash
# Run CoreML HTDemucs separation for parity comparison

set -e

INPUT="$1"
OUTPUT_DIR="${2}"

if [ -z "$INPUT" ]; then
    echo "Usage: $0 <input_audio> [output_dir]"
    exit 1
fi

# Get absolute path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR_ABS="$SCRIPT_DIR/outputs"
else
    # Get absolute path to output directory
    if [[ "$OUTPUT_DIR" = /* ]]; then
        # Already absolute
        OUTPUT_DIR_ABS="$OUTPUT_DIR"
    else
        # Make it relative to current directory, not script directory
        OUTPUT_DIR_ABS="$(pwd)/$OUTPUT_DIR"
    fi
fi

# Get absolute path to the repository root (2 levels up from tests/parity)
REPO_ROOT="$SCRIPT_DIR/../.."

echo "Running CoreML HTDemucs separation..."

# Use the CLI we built in Phase 3
"$REPO_ROOT/.build/release/htdemucs-cli" separate "$INPUT" --output "$OUTPUT_DIR_ABS" --format wav

# Rename outputs to match PyTorch naming
cd "$OUTPUT_DIR_ABS"
for stem in drums bass vocals other piano guitar; do
    if [ -f "${stem}.wav" ]; then
        mv "${stem}.wav" "${stem}_coreml.wav"
        echo "  ✓ ${stem}_coreml.wav"
    fi
done

echo ""
echo "✓ CoreML separation complete"
