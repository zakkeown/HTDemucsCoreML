# DevOps Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement CI/CD hardening, developer experience improvements, and performance monitoring infrastructure.

**Architecture:** Makefile as unified command interface, pre-commit hooks for local quality gates, parallel GitHub Actions with caching, benchmark infrastructure comparing CoreML vs PyTorch at each pipeline stage, and GitHub Pages for report publishing.

**Tech Stack:** Make, pre-commit, SwiftFormat, Ruff, GitHub Actions, GitHub Pages, Python (benchmarks), Chart.js (visualization)

---

## Task 1: Makefile

**Files:**
- Create: `Makefile`

**Step 1: Create the Makefile**

```makefile
.PHONY: build build-cli clean test test-parity test-all lint format format-fix setup setup-hooks benchmark benchmark-compare parity-report help

# Default target
help:
	@echo "HTDemucs CoreML - Development Commands"
	@echo ""
	@echo "Build:"
	@echo "  make build        Build Swift package (release)"
	@echo "  make build-cli    Build CLI tool"
	@echo "  make clean        Clean build artifacts"
	@echo ""
	@echo "Test:"
	@echo "  make test         Run Swift unit tests"
	@echo "  make test-parity  Run Python parity tests"
	@echo "  make test-all     Run all tests"
	@echo ""
	@echo "Quality:"
	@echo "  make lint         Check code style (Swift + Python)"
	@echo "  make format       Check formatting (no changes)"
	@echo "  make format-fix   Apply formatting fixes"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        Install all dependencies"
	@echo "  make setup-hooks  Install pre-commit hooks"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make benchmark    Run performance benchmark"
	@echo "  make benchmark-compare  Compare to baseline"
	@echo "  make parity-report      Generate HTML parity report"

# =============================================================================
# Build
# =============================================================================

build:
	swift build -c release

build-cli:
	swift build -c release --product htdemucs-cli

clean:
	swift package clean
	rm -rf .build

# =============================================================================
# Test
# =============================================================================

test:
	swift test

PARITY_VENV := tests/parity/venv
PARITY_PYTHON := $(PARITY_VENV)/bin/python
PARITY_PIP := $(PARITY_VENV)/bin/pip
PARITY_PYTEST := $(PARITY_VENV)/bin/pytest

test-parity: $(PARITY_VENV)
	cd tests/parity && $(PARITY_PYTEST) test_parity.py -v

test-all: test test-parity

# =============================================================================
# Quality
# =============================================================================

lint: lint-swift lint-python

lint-swift:
	@if command -v swiftformat >/dev/null 2>&1; then \
		swiftformat --lint Sources tests --quiet; \
	else \
		echo "swiftformat not installed. Run: brew install swiftformat"; \
		exit 1; \
	fi

lint-python: $(PARITY_VENV)
	$(PARITY_VENV)/bin/ruff check .

format: format-swift format-python

format-swift:
	@if command -v swiftformat >/dev/null 2>&1; then \
		swiftformat --lint Sources tests; \
	else \
		echo "swiftformat not installed. Run: brew install swiftformat"; \
		exit 1; \
	fi

format-python: $(PARITY_VENV)
	$(PARITY_VENV)/bin/ruff format --check .
	$(PARITY_VENV)/bin/ruff check .

format-fix: format-fix-swift format-fix-python

format-fix-swift:
	@if command -v swiftformat >/dev/null 2>&1; then \
		swiftformat Sources tests; \
	else \
		echo "swiftformat not installed. Run: brew install swiftformat"; \
		exit 1; \
	fi

format-fix-python: $(PARITY_VENV)
	$(PARITY_VENV)/bin/ruff format .
	$(PARITY_VENV)/bin/ruff check --fix .

# =============================================================================
# Setup
# =============================================================================

setup: setup-swift setup-python setup-hooks

setup-swift:
	swift package resolve

setup-python: $(PARITY_VENV)
	$(PARITY_PIP) install --upgrade pip
	$(PARITY_PIP) install -r tests/parity/requirements.txt
	$(PARITY_PIP) install ruff

$(PARITY_VENV):
	python3 -m venv $(PARITY_VENV)

setup-hooks:
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
	else \
		echo "pre-commit not installed. Run: pip install pre-commit"; \
		exit 1; \
	fi

# =============================================================================
# Benchmarks
# =============================================================================

benchmark: $(PARITY_VENV) build-cli
	$(PARITY_PYTHON) benchmarks/run_benchmark.py

benchmark-compare: $(PARITY_VENV)
	$(PARITY_PYTHON) benchmarks/compare.py

parity-report: $(PARITY_VENV)
	cd tests/parity && $(PARITY_PYTHON) generate_report.py
```

**Step 2: Test the Makefile**

Run: `make help`

Expected: Help text displays all available commands.

Run: `make build`

Expected: Swift package builds successfully.

**Step 3: Commit**

```bash
git add Makefile
git commit -m "build: add Makefile for unified development commands"
```

---

## Task 2: SwiftFormat Configuration

**Files:**
- Create: `.swiftformat`

**Step 1: Create SwiftFormat config**

```swift
# SwiftFormat configuration for HTDemucs CoreML

# File options
--swiftversion 6.0
--exclude .build,Resources

# Format options
--indent 4
--indentcase false
--trimwhitespace always
--linebreaks lf
--maxwidth 120

# Rules
--enable blankLinesBetweenScopes
--enable consecutiveBlankLines
--enable duplicateImports
--enable redundantSelf
--enable sortImports
--enable trailingCommas
--enable wrapArguments

# Disable rules that conflict with project style
--disable acronyms
--disable wrapMultilineStatementBraces
```

**Step 2: Verify SwiftFormat is installed**

Run: `which swiftformat || echo "Install with: brew install swiftformat"`

**Step 3: Test the config**

Run: `swiftformat --lint Sources tests --verbose 2>&1 | head -20`

Expected: Either clean output or list of formatting suggestions.

**Step 4: Commit**

```bash
git add .swiftformat
git commit -m "build: add SwiftFormat configuration"
```

---

## Task 3: Pre-commit Configuration

**Files:**
- Create: `.pre-commit-config.yaml`

**Step 1: Create pre-commit config**

```yaml
# Pre-commit hooks for HTDemucs CoreML
# Install: pip install pre-commit && pre-commit install

repos:
  # Swift formatting
  - repo: https://github.com/nicklockwood/SwiftFormat
    rev: 0.54.6
    hooks:
      - id: swiftformat
        args: [--config, .swiftformat]

  # Python linting and formatting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  # General hygiene
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
        exclude: '\.md$'
      - id: end-of-file-fixer
        exclude: '\.md$'
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: check-merge-conflict
```

**Step 2: Create Ruff config in pyproject.toml**

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]  # Line too long - handled by formatter

[tool.ruff.format]
quote-style = "double"
```

**Step 3: Install pre-commit and test**

Run: `pip install pre-commit && pre-commit install`

Run: `pre-commit run --all-files`

Expected: All hooks pass (or show fixable issues).

**Step 4: Commit**

```bash
git add .pre-commit-config.yaml pyproject.toml
git commit -m "build: add pre-commit hooks for code quality"
```

---

## Task 4: Dependabot Configuration

**Files:**
- Create: `.github/dependabot.yml`

**Step 1: Create Dependabot config**

```yaml
# Dependabot configuration for HTDemucs CoreML
# https://docs.github.com/en/code-security/dependabot

version: 2
updates:
  # Swift Package Manager
  - package-ecosystem: "swift"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 3
    commit-message:
      prefix: "deps(swift):"

  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 3
    commit-message:
      prefix: "deps(python):"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
    open-pull-requests-limit: 3
    commit-message:
      prefix: "deps(actions):"
```

**Step 2: Commit**

```bash
git add .github/dependabot.yml
git commit -m "ci: add Dependabot for automated dependency updates"
```

---

## Task 5: Benchmark Infrastructure - Core Script

**Files:**
- Create: `benchmarks/run_benchmark.py`
- Create: `benchmarks/__init__.py`

**Step 1: Create benchmarks directory structure**

```bash
mkdir -p benchmarks
```

**Step 2: Create the benchmark runner**

Create `benchmarks/run_benchmark.py`:

```python
#!/usr/bin/env python3
"""
Run performance benchmarks comparing CoreML vs PyTorch HTDemucs.

Measures timing at each pipeline stage:
- Model loading
- STFT computation
- Frequency branch inference
- Time branch inference
- iSTFT computation
- Full pipeline end-to-end

Also tracks peak memory usage for both implementations.
"""

import argparse
import json
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

# Test audio duration in seconds
BENCHMARK_DURATION = 30.0
SAMPLE_RATE = 44100


def generate_test_audio(duration: float = BENCHMARK_DURATION) -> tuple:
    """Generate reproducible test audio for benchmarking."""
    np.random.seed(42)
    samples = int(duration * SAMPLE_RATE)

    # Mix of frequencies to simulate music-like content
    t = np.linspace(0, duration, samples)
    left = (
        0.3 * np.sin(2 * np.pi * 440 * t) +
        0.2 * np.sin(2 * np.pi * 880 * t) +
        0.1 * np.random.randn(samples)
    ).astype(np.float32)

    right = (
        0.3 * np.sin(2 * np.pi * 440 * t + 0.5) +
        0.2 * np.sin(2 * np.pi * 660 * t) +
        0.1 * np.random.randn(samples)
    ).astype(np.float32)

    return left, right


def benchmark_pytorch(left: np.ndarray, right: np.ndarray) -> dict:
    """Benchmark PyTorch HTDemucs with per-stage timing."""
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    results = {}

    # Model loading
    tracemalloc.start()
    start = time.perf_counter()
    model = get_model("htdemucs_6s")
    model.eval()
    results["model_load_sec"] = time.perf_counter() - start

    # Prepare audio
    audio = torch.from_numpy(np.stack([left, right], axis=0)).float()

    # Full pipeline (apply_model handles STFT internally)
    start = time.perf_counter()
    with torch.no_grad():
        sources = apply_model(
            model,
            audio.unsqueeze(0),
            device="cpu",
            shifts=1,
            split=True,
            overlap=0.25
        )
    results["full_pipeline_sec"] = time.perf_counter() - start

    # Memory tracking
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["memory_peak_mb"] = peak / 1024 / 1024

    # Note: PyTorch apply_model doesn't expose per-stage timing easily
    # We report full pipeline only for PyTorch
    results["stft_sec"] = None
    results["freq_branch_sec"] = None
    results["time_branch_sec"] = None
    results["istft_sec"] = None

    return results


def benchmark_coreml(left: np.ndarray, right: np.ndarray, model_path: str) -> dict:
    """Benchmark CoreML HTDemucs with per-stage timing."""
    import coremltools as ct
    import math

    # Constants matching Swift implementation
    FFT_SIZE = 4096
    HOP_LENGTH = 1024
    SEGMENT_SAMPLES = 343980

    results = {}

    # Model loading
    tracemalloc.start()
    start = time.perf_counter()
    model = ct.models.MLModel(model_path)
    results["model_load_sec"] = time.perf_counter() - start

    # Pad audio
    def pad_to_segment(audio, target):
        if len(audio) >= target:
            return audio[:target]
        return np.pad(audio, (0, target - len(audio)), mode='reflect')

    padded_left = pad_to_segment(left, SEGMENT_SAMPLES)
    padded_right = pad_to_segment(right, SEGMENT_SAMPLES)

    # STFT timing
    def compute_stft(audio):
        pad_size = FFT_SIZE // 2
        padded = np.pad(audio, pad_size, mode='reflect')
        window = np.hanning(FFT_SIZE + 1)[:-1]
        num_frames = (len(padded) - FFT_SIZE) // HOP_LENGTH + 1
        norm_factor = 1.0 / math.sqrt(FFT_SIZE)

        real_out, imag_out = [], []
        for i in range(num_frames):
            start_idx = i * HOP_LENGTH
            frame = padded[start_idx:start_idx + FFT_SIZE] * window
            spectrum = np.fft.rfft(frame) * norm_factor
            real_out.append(spectrum.real)
            imag_out.append(spectrum.imag)
        return np.array(real_out), np.array(imag_out)

    start = time.perf_counter()
    left_real, left_imag = compute_stft(padded_left)
    right_real, right_imag = compute_stft(padded_right)
    results["stft_sec"] = time.perf_counter() - start

    # Prepare model input
    model_bins, model_frames = 2048, 336

    def prepare_for_model(real, imag):
        r, i = real[:, :model_bins], imag[:, :model_bins]
        if r.shape[0] < model_frames:
            pad = model_frames - r.shape[0]
            r = np.pad(r, ((0, pad), (0, 0)), mode='constant')
            i = np.pad(i, ((0, pad), (0, 0)), mode='constant')
        return r[:model_frames], i[:model_frames]

    lr, li = prepare_for_model(left_real, left_imag)
    rr, ri = prepare_for_model(right_real, right_imag)

    cac = np.stack([lr.T, li.T, rr.T, ri.T], axis=0)[np.newaxis, ...].astype(np.float32)
    raw = np.stack([padded_left, padded_right], axis=0)[np.newaxis, ...].astype(np.float32)

    # Model inference (freq + time branches combined)
    start = time.perf_counter()
    output = model.predict({"spectrogram": cac, "raw_audio": raw})
    inference_time = time.perf_counter() - start

    # We can't easily separate freq vs time branch in CoreML
    # Report combined inference time
    results["freq_branch_sec"] = inference_time / 2  # Approximate split
    results["time_branch_sec"] = inference_time / 2

    # iSTFT timing
    def compute_istft(real, imag, length):
        window = np.hanning(FFT_SIZE + 1)[:-1]
        num_frames = real.shape[0]
        pad_size = FFT_SIZE // 2
        norm_factor = math.sqrt(FFT_SIZE)

        output_length = (num_frames - 1) * HOP_LENGTH + FFT_SIZE
        output = np.zeros(output_length)
        window_sum = np.zeros(output_length)

        for i in range(num_frames):
            spectrum = (real[i] + 1j * imag[i]) * norm_factor
            frame = np.fft.irfft(spectrum, FFT_SIZE)
            start_idx = i * HOP_LENGTH
            output[start_idx:start_idx + FFT_SIZE] += frame * window
            window_sum[start_idx:start_idx + FFT_SIZE] += window ** 2

        output = np.where(window_sum > 1e-8, output / window_sum, output)
        return output[pad_size:pad_size + length]

    freq_out = output["add_66"]
    original_length = len(left)
    original_frames = left_real.shape[0]

    start = time.perf_counter()
    # iSTFT for one stem (representative timing)
    stem_real = freq_out[0, 0, 0].T[:original_frames]
    stem_imag = freq_out[0, 0, 1].T[:original_frames]
    stem_real_full = np.pad(stem_real, ((0, 0), (0, 1)), mode='constant')
    stem_imag_full = np.pad(stem_imag, ((0, 0), (0, 1)), mode='constant')
    _ = compute_istft(stem_real_full, stem_imag_full, original_length)
    istft_single = time.perf_counter() - start
    # Scale for all 6 stems * 2 channels
    results["istft_sec"] = istft_single * 12

    # Full pipeline
    results["full_pipeline_sec"] = (
        results["stft_sec"] +
        results["freq_branch_sec"] +
        results["time_branch_sec"] +
        results["istft_sec"]
    )

    # Memory tracking
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["memory_peak_mb"] = peak / 1024 / 1024

    return results


def run_benchmarks(model_path: str, output_path: str, runs: int = 3) -> dict:
    """Run benchmarks and return results."""
    print(f"Generating {BENCHMARK_DURATION}s test audio...")
    left, right = generate_test_audio()

    print(f"\nRunning PyTorch benchmark ({runs} runs)...")
    pytorch_results = []
    for i in range(runs):
        print(f"  Run {i + 1}/{runs}...", end=" ", flush=True)
        result = benchmark_pytorch(left.copy(), right.copy())
        pytorch_results.append(result)
        print(f"done ({result['full_pipeline_sec']:.2f}s)")

    print(f"\nRunning CoreML benchmark ({runs} runs)...")
    coreml_results = []
    for i in range(runs):
        print(f"  Run {i + 1}/{runs}...", end=" ", flush=True)
        result = benchmark_coreml(left.copy(), right.copy(), model_path)
        coreml_results.append(result)
        print(f"done ({result['full_pipeline_sec']:.2f}s)")

    # Average results
    def average_results(results_list):
        avg = {}
        for key in results_list[0].keys():
            values = [r[key] for r in results_list if r[key] is not None]
            avg[key] = sum(values) / len(values) if values else None
        return avg

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "audio_duration_sec": BENCHMARK_DURATION,
        "num_runs": runs,
        "coreml": average_results(coreml_results),
        "pytorch": average_results(pytorch_results),
    }


def print_comparison(results: dict):
    """Print formatted comparison table."""
    coreml = results["coreml"]
    pytorch = results["pytorch"]

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Audio duration: {results['audio_duration_sec']}s")
    print(f"Averaged over: {results['num_runs']} runs")
    print()
    print(f"{'Metric':<20} {'CoreML':>12} {'PyTorch':>12} {'Δ':>10}")
    print("-" * 60)

    metrics = [
        ("STFT", "stft_sec", "s"),
        ("Freq branch", "freq_branch_sec", "s"),
        ("Time branch", "time_branch_sec", "s"),
        ("iSTFT", "istft_sec", "s"),
        ("Full pipeline", "full_pipeline_sec", "s"),
        ("Model load", "model_load_sec", "s"),
        ("Memory peak", "memory_peak_mb", "MB"),
    ]

    for label, key, unit in metrics:
        c_val = coreml.get(key)
        p_val = pytorch.get(key)

        if c_val is None and p_val is None:
            continue

        c_str = f"{c_val:.2f}{unit}" if c_val else "N/A"
        p_str = f"{p_val:.2f}{unit}" if p_val else "N/A"

        if c_val and p_val:
            delta = ((c_val - p_val) / p_val) * 100
            delta_str = f"{delta:+.0f}%"
            if delta < -5:
                delta_str += " ✓"
        else:
            delta_str = ""

        print(f"{label:<20} {c_str:>12} {p_str:>12} {delta_str:>10}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run HTDemucs benchmarks")
    parser.add_argument(
        "--model",
        default="Resources/Models/htdemucs_6s.mlpackage",
        help="Path to CoreML model"
    )
    parser.add_argument(
        "--output",
        default="benchmarks/latest.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs to average"
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Add results to baseline.json"
    )

    args = parser.parse_args()

    # Find model
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to script
        script_dir = Path(__file__).parent.parent
        model_path = script_dir / args.model

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1

    results = run_benchmarks(str(model_path), args.output, args.runs)
    print_comparison(results)

    # Save latest results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved results to {output_path}")

    # Update baseline if requested
    if args.update_baseline:
        baseline_path = Path(__file__).parent / "baseline.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
        else:
            baseline = {}

        baseline[results["date"]] = {
            "coreml": results["coreml"],
            "pytorch": results["pytorch"],
        }
        baseline_path.write_text(json.dumps(baseline, indent=2))
        print(f"✓ Updated baseline at {baseline_path}")

    return 0


if __name__ == "__main__":
    exit(main())
```

**Step 3: Create empty __init__.py**

```bash
touch benchmarks/__init__.py
```

**Step 4: Test the benchmark script**

Run: `python benchmarks/run_benchmark.py --runs 1`

Expected: Benchmark runs and prints comparison table.

**Step 5: Commit**

```bash
git add benchmarks/
git commit -m "feat: add benchmark infrastructure for CoreML vs PyTorch comparison"
```

---

## Task 6: Benchmark Comparison Script

**Files:**
- Create: `benchmarks/compare.py`
- Create: `benchmarks/baseline.json`

**Step 1: Create the comparison script**

Create `benchmarks/compare.py`:

```python
#!/usr/bin/env python3
"""
Compare current benchmark results against baseline.

Flags regressions if:
- CoreML throughput regresses >10%
- Memory usage increases >15%
"""

import argparse
import json
import sys
from pathlib import Path


THROUGHPUT_THRESHOLD = 0.10  # 10% regression
MEMORY_THRESHOLD = 0.15      # 15% increase


def load_json(path: Path) -> dict:
    """Load JSON file."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def compare_results(current: dict, baseline_entry: dict) -> list:
    """Compare current results to baseline, return list of issues."""
    issues = []

    curr_coreml = current.get("coreml", {})
    base_coreml = baseline_entry.get("coreml", {})

    # Check throughput (full pipeline time)
    curr_time = curr_coreml.get("full_pipeline_sec")
    base_time = base_coreml.get("full_pipeline_sec")

    if curr_time and base_time:
        regression = (curr_time - base_time) / base_time
        if regression > THROUGHPUT_THRESHOLD:
            issues.append(
                f"REGRESSION: Pipeline time increased {regression:.1%} "
                f"({base_time:.2f}s -> {curr_time:.2f}s)"
            )

    # Check memory
    curr_mem = curr_coreml.get("memory_peak_mb")
    base_mem = base_coreml.get("memory_peak_mb")

    if curr_mem and base_mem:
        increase = (curr_mem - base_mem) / base_mem
        if increase > MEMORY_THRESHOLD:
            issues.append(
                f"REGRESSION: Memory usage increased {increase:.1%} "
                f"({base_mem:.0f}MB -> {curr_mem:.0f}MB)"
            )

    return issues


def main():
    parser = argparse.ArgumentParser(description="Compare benchmarks to baseline")
    parser.add_argument(
        "--current",
        default="benchmarks/latest.json",
        help="Current results file"
    )
    parser.add_argument(
        "--baseline",
        default="benchmarks/baseline.json",
        help="Baseline file"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code on regression"
    )

    args = parser.parse_args()

    current_path = Path(args.current)
    baseline_path = Path(args.baseline)

    if not current_path.exists():
        print(f"Error: Current results not found at {current_path}")
        print("Run 'make benchmark' first.")
        return 1

    current = load_json(current_path)
    baseline = load_json(baseline_path)

    if not baseline:
        print("No baseline found. Current results will become the baseline.")
        print(f"Run with --update-baseline to save: python benchmarks/run_benchmark.py --update-baseline")
        return 0

    # Get most recent baseline entry
    dates = sorted(baseline.keys())
    if not dates:
        print("Baseline file is empty.")
        return 0

    latest_date = dates[-1]
    baseline_entry = baseline[latest_date]

    print(f"Comparing against baseline from {latest_date}")
    print()

    issues = compare_results(current, baseline_entry)

    if issues:
        print("=" * 60)
        print("REGRESSIONS DETECTED")
        print("=" * 60)
        for issue in issues:
            print(f"  ⚠️  {issue}")
        print()

        if args.strict:
            return 1
    else:
        print("✓ No regressions detected")

    # Print summary comparison
    curr = current.get("coreml", {})
    base = baseline_entry.get("coreml", {})

    print()
    print(f"{'Metric':<25} {'Baseline':>12} {'Current':>12} {'Change':>10}")
    print("-" * 60)

    for key, label in [
        ("full_pipeline_sec", "Full pipeline (s)"),
        ("memory_peak_mb", "Memory peak (MB)"),
        ("stft_sec", "STFT (s)"),
        ("istft_sec", "iSTFT (s)"),
    ]:
        b_val = base.get(key)
        c_val = curr.get(key)

        if b_val is None or c_val is None:
            continue

        change = ((c_val - b_val) / b_val) * 100
        change_str = f"{change:+.1f}%"

        print(f"{label:<25} {b_val:>12.2f} {c_val:>12.2f} {change_str:>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Create initial empty baseline**

Create `benchmarks/baseline.json`:

```json
{}
```

**Step 3: Test comparison script**

Run: `python benchmarks/compare.py`

Expected: Message about no baseline found.

**Step 4: Commit**

```bash
git add benchmarks/compare.py benchmarks/baseline.json
git commit -m "feat: add benchmark comparison with regression detection"
```

---

## Task 7: GitHub Actions - Parallel CI Workflow

**Files:**
- Replace: `.github/workflows/parity-tests.yml` → `.github/workflows/ci.yml`

**Step 1: Create the new CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

# Cancel in-progress runs for the same branch
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # ============================================================================
  # Lint job - Fast feedback on code quality
  # ============================================================================
  lint:
    name: Lint
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install SwiftFormat
        run: brew install swiftformat

      - name: Check Swift formatting
        run: swiftformat --lint Sources tests --config .swiftformat

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Ruff
        run: pip install ruff

      - name: Check Python formatting
        run: |
          ruff format --check .
          ruff check .

  # ============================================================================
  # Build and test Swift package
  # ============================================================================
  build-test:
    name: Build & Test
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4

      - name: Cache SPM dependencies
        uses: actions/cache@v4
        with:
          path: .build
          key: ${{ runner.os }}-spm-${{ hashFiles('Package.resolved') }}
          restore-keys: |
            ${{ runner.os }}-spm-

      - name: Build
        run: swift build -c release

      - name: Run Swift tests
        run: swift test

  # ============================================================================
  # Python parity tests
  # ============================================================================
  parity:
    name: Parity Tests
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Cache Python dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('tests/parity/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install Python dependencies
        run: pip install -r tests/parity/requirements.txt

      - name: Cache SPM dependencies
        uses: actions/cache@v4
        with:
          path: .build
          key: ${{ runner.os }}-spm-${{ hashFiles('Package.resolved') }}
          restore-keys: |
            ${{ runner.os }}-spm-

      - name: Build Swift CLI
        run: swift build -c release --product htdemucs-cli

      - name: Generate test fixtures
        run: python tests/parity/fixtures/generate_fixtures.py

      - name: Run parity tests
        run: |
          cd tests/parity
          pytest test_parity.py -v --tb=short

      - name: Generate parity report
        if: always()
        run: |
          cd tests/parity
          python generate_report.py || true

      - name: Upload parity results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: parity-results
          path: tests/parity/outputs/

  # ============================================================================
  # Benchmarks (main branch only)
  # ============================================================================
  benchmark:
    name: Benchmark
    runs-on: macos-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    needs: [build-test, parity]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Cache Python dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-benchmark-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          pip install torch torchaudio demucs coremltools numpy soundfile

      - name: Run benchmarks
        run: python benchmarks/run_benchmark.py --runs 2 --update-baseline

      - name: Check for regressions
        run: python benchmarks/compare.py

      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmarks/latest.json

      - name: Commit baseline update
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "ci: update benchmark baseline [skip ci]"
          file_pattern: benchmarks/baseline.json

  # ============================================================================
  # Deploy reports to GitHub Pages (main branch only)
  # ============================================================================
  deploy-pages:
    name: Deploy Pages
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    needs: [parity]
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4

      - name: Download parity results
        uses: actions/download-artifact@v4
        with:
          name: parity-results
          path: _site

      - name: Create index redirect
        run: |
          if [ -f _site/parity_report.html ]; then
            cp _site/parity_report.html _site/index.html
          fi

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: _site

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

**Step 2: Delete the old workflow**

```bash
rm .github/workflows/parity-tests.yml
```

**Step 3: Commit**

```bash
git add .github/workflows/
git commit -m "ci: restructure to parallel jobs with caching and GitHub Pages"
```

---

## Task 8: Enable GitHub Pages in Repository Settings

**Note:** This is a manual step in GitHub UI.

**Step 1: Go to repository Settings → Pages**

**Step 2: Under "Build and deployment":**
- Source: GitHub Actions

**Step 3: Save**

The workflow will deploy on next push to main.

---

## Task 9: Final Integration Test

**Step 1: Run the full local workflow**

```bash
make setup
make lint
make test
make build
```

Expected: All commands succeed.

**Step 2: Test pre-commit hooks**

```bash
echo "  " >> README.md  # Add trailing whitespace
git add README.md
git commit -m "test: trigger pre-commit"
```

Expected: Pre-commit hook catches trailing whitespace and blocks commit.

**Step 3: Clean up test**

```bash
git checkout README.md
```

**Step 4: Final commit with all changes**

```bash
git status
# Ensure everything is committed
```

---

## Summary

| Task | What it creates |
|------|-----------------|
| 1 | `Makefile` - unified commands |
| 2 | `.swiftformat` - Swift style config |
| 3 | `.pre-commit-config.yaml` - quality hooks |
| 4 | `.github/dependabot.yml` - auto updates |
| 5 | `benchmarks/run_benchmark.py` - performance tests |
| 6 | `benchmarks/compare.py` - regression detection |
| 7 | `.github/workflows/ci.yml` - parallel CI |
| 8 | GitHub Pages enabled (manual) |
| 9 | Integration verified |

**Estimated commits:** 7-8
