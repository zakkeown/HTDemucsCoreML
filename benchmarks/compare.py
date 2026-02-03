#!/usr/bin/env python3
"""
Compare benchmark results against baseline to detect performance regressions.

Compares current benchmark results against historical baseline data,
flagging any significant regressions in throughput or memory usage.
"""

import argparse
import json
import sys
from pathlib import Path

# Regression thresholds
THROUGHPUT_THRESHOLD = 0.10  # 10% regression
MEMORY_THRESHOLD = 0.15  # 15% increase


def load_json(path: Path) -> dict | None:
    """Load JSON file, returning None if file doesn't exist."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def get_most_recent_baseline(baseline: dict) -> dict | None:
    """Get the most recent entry from baseline (sorted by date keys)."""
    if not baseline:
        return None

    # Sort by date keys (YYYY-MM-DD format sorts correctly as strings)
    sorted_dates = sorted(baseline.keys(), reverse=True)
    if not sorted_dates:
        return None

    most_recent_date = sorted_dates[0]
    entry = baseline[most_recent_date]
    entry["date"] = most_recent_date
    return entry


def compare_metrics(
    current: dict, baseline: dict
) -> tuple[list[tuple[str, float, float, float]], list[str]]:
    """Compare current vs baseline metrics.

    Returns:
        (comparisons, regressions) where:
        - comparisons: list of (metric_name, current_value, baseline_value, delta_pct)
        - regressions: list of warning messages for detected regressions
    """
    comparisons = []
    regressions = []

    # Compare CoreML metrics (primary target)
    current_coreml = current.get("coreml", {})
    baseline_coreml = baseline.get("coreml", {})

    # Full pipeline time
    current_time = current_coreml.get("full_pipeline_sec")
    baseline_time = baseline_coreml.get("full_pipeline_sec")

    if current_time is not None and baseline_time is not None and baseline_time > 0:
        delta_pct = (current_time - baseline_time) / baseline_time
        comparisons.append(("full_pipeline_sec", current_time, baseline_time, delta_pct))

        if delta_pct > THROUGHPUT_THRESHOLD:
            regressions.append(
                f"REGRESSION: full_pipeline_sec increased by {delta_pct * 100:.1f}% "
                f"(threshold: {THROUGHPUT_THRESHOLD * 100:.0f}%)"
            )

    # Memory peak
    current_memory = current_coreml.get("memory_peak_mb")
    baseline_memory = baseline_coreml.get("memory_peak_mb")

    if (
        current_memory is not None
        and baseline_memory is not None
        and baseline_memory > 0
    ):
        delta_pct = (current_memory - baseline_memory) / baseline_memory
        comparisons.append(("memory_peak_mb", current_memory, baseline_memory, delta_pct))

        if delta_pct > MEMORY_THRESHOLD:
            regressions.append(
                f"REGRESSION: memory_peak_mb increased by {delta_pct * 100:.1f}% "
                f"(threshold: {MEMORY_THRESHOLD * 100:.0f}%)"
            )

    return comparisons, regressions


def print_comparison_table(
    comparisons: list[tuple[str, float, float, float]], baseline_date: str
):
    """Print formatted comparison table."""
    print()
    print("=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"Baseline date: {baseline_date}")
    print()
    print(f"{'Metric':<25} {'Current':>12} {'Baseline':>12} {'Delta':>12}")
    print("-" * 70)

    for metric, current, baseline, delta_pct in comparisons:
        if "sec" in metric:
            current_str = f"{current:.2f}s"
            baseline_str = f"{baseline:.2f}s"
        elif "mb" in metric.lower():
            current_str = f"{current:.1f}MB"
            baseline_str = f"{baseline:.1f}MB"
        else:
            current_str = f"{current:.2f}"
            baseline_str = f"{baseline:.2f}"

        delta_str = f"{delta_pct * 100:+.1f}%"
        print(f"{metric:<25} {current_str:>12} {baseline_str:>12} {delta_str:>12}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results against baseline"
    )
    parser.add_argument(
        "--current",
        default="benchmarks/latest.json",
        help="Path to current benchmark results (default: benchmarks/latest.json)",
    )
    parser.add_argument(
        "--baseline",
        default="benchmarks/baseline.json",
        help="Path to baseline file (default: benchmarks/baseline.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if regressions are detected",
    )

    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    current_path = Path(args.current)
    if not current_path.is_absolute():
        current_path = repo_root / args.current

    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = repo_root / args.baseline

    # Load current results
    current = load_json(current_path)
    if current is None:
        print(f"Error: Current benchmark file not found: {current_path}")
        return 1

    # Load baseline
    baseline_data = load_json(baseline_path)
    if baseline_data is None:
        print(f"No baseline found at {baseline_path}. Skipping comparison.")
        return 0

    if not baseline_data:
        print("Baseline is empty. Skipping comparison.")
        return 0

    # Get most recent baseline entry
    baseline_entry = get_most_recent_baseline(baseline_data)
    if baseline_entry is None:
        print("No baseline entries found. Skipping comparison.")
        return 0

    # Compare metrics
    comparisons, regressions = compare_metrics(current, baseline_entry)

    if not comparisons:
        print("No comparable metrics found between current and baseline.")
        return 0

    # Print results
    print_comparison_table(comparisons, baseline_entry.get("date", "unknown"))

    # Print regression warnings
    if regressions:
        print()
        for warning in regressions:
            print(f"WARNING: {warning}")
        print()

        if args.strict:
            print("Exiting with code 1 due to --strict flag and detected regressions.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
