# DevOps & Architecture Foundation Design

**Date:** 2026-02-03
**Status:** Approved
**Goal:** Establish low-maintenance DevOps infrastructure for CI/CD hardening, developer experience, and performance monitoring.

## Context

HTDemucsCoreML is a completed conversion project - the core work is done. The DevOps focus is:

- Low maintenance automation (set and forget)
- Regression detection (catch if something breaks)
- Easy onboarding for occasional contributors
- Not over-engineered for a "done" codebase

## Design Overview

| Component | Purpose |
|-----------|---------|
| Makefile | Single entry point for all commands |
| Pre-commit hooks | Automated quality checks on every commit |
| GitHub Actions | Parallel CI with caching |
| Benchmarks | CoreML vs PyTorch performance tracking |
| Dependabot | Automated dependency updates |
| GitHub Pages | Published parity reports and benchmark history |

## Component Details

### 1. Makefile

A single `Makefile` at repo root serves as the command hub.

**Targets:**

```makefile
# Build targets
build           # swift build -c release
build-cli       # swift build -c release --product htdemucs-cli
clean           # swift package clean + remove .build

# Test targets
test            # swift test (unit tests)
test-parity     # run Python parity tests
test-all        # both of the above

# Quality targets
lint            # swiftlint + ruff check
format          # swiftformat --lint + ruff format --check
format-fix      # swiftformat + ruff format (apply fixes)

# Setup targets
setup           # install dependencies, setup Python venv
setup-hooks     # install pre-commit hooks

# Benchmark targets
benchmark       # run performance benchmark, output results
benchmark-compare  # compare current run to baseline

# Report targets
parity-report   # generate HTML parity report
```

**Design decisions:**

- Makefile over alternatives (Just, Taskfile) because `make` is on every Mac with zero dependencies
- Python venv activation handled transparently within targets
- Tab-completion works out of the box

### 2. Pre-commit Hooks

Using the [pre-commit](https://pre-commit.com) framework.

**`.pre-commit-config.yaml`:**

```yaml
repos:
  # Swift formatting
  - repo: https://github.com/nicklockwood/SwiftFormat
    rev: 0.54.0
    hooks:
      - id: swiftformat

  # Python quality
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
      - id: ruff-format

  # General hygiene
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
```

**Tool choices:**

- **SwiftFormat**: Most popular Swift formatter, configurable
- **Ruff**: Replaces flake8+black+isort, extremely fast
- **check-added-large-files**: Prevents accidental model/binary commits

### 3. GitHub Actions

Restructure from sequential to parallel jobs with caching.

**Workflow structure:**

```
┌─────────────────────────────────────────────────────────┐
│                    On PR / Push to main                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│   │  Lint    │   │  Build   │   │  Parity Tests    │   │
│   │  (30s)   │   │  + Test  │   │  (Python, slow)  │   │
│   │          │   │  (2-3m)  │   │  (3-5m)          │   │
│   └──────────┘   └──────────┘   └──────────────────┘   │
│        │              │                   │             │
│        └──────────────┴───────────────────┘             │
│                       ▼                                 │
│              ┌───────────────┐                          │
│              │ Upload Report │                          │
│              │ to GH Pages   │                          │
│              └───────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

**Key features:**

1. **Parallel jobs** - Lint, Swift build/test, and Python parity run simultaneously
2. **Caching** - SPM dependencies and Python venv cached
3. **Pre-commit in CI** - Same hooks as local
4. **GitHub Pages deploy** - Parity report auto-published on main
5. **PR status checks** - Each job reports separately

**Estimated improvement:** ~8-10 min sequential → ~5 min parallel

### 4. Performance Benchmarks

Track CoreML vs PyTorch performance at each pipeline stage.

**Metrics:**

| Metric | What it measures |
|--------|------------------|
| STFT | vDSP vs `torch.stft()` |
| Frequency branch | CoreML vs PyTorch encoder/decoder |
| Time branch | CoreML vs PyTorch temporal processing |
| iSTFT | vDSP vs `torch.istft()` |
| Full pipeline | End-to-end separation time |
| Model load | CoreML compile vs PyTorch model load |
| Memory peak | Both implementations |

**Output format:**

```
                    CoreML     PyTorch    Δ
─────────────────────────────────────────────
STFT (30s audio)    0.42s      0.38s    +11%
Freq branch         3.21s      4.87s    -34%  ✓
Time branch         2.89s      3.92s    -26%  ✓
iSTFT               0.31s      0.29s    +7%
─────────────────────────────────────────────
Full pipeline       8.12s     11.20s    -28%  ✓
Memory peak        1.84GB     2.91GB    -37%  ✓
```

**File structure:**

```
benchmarks/
├── run_benchmark.py      # Runs benchmark, outputs JSON
├── compare.py            # Compares to baseline, flags regressions
└── baseline.json         # Historical results (committed to repo)
```

**`baseline.json` schema:**

```json
{
  "2026-02-03": {
    "coreml": {
      "stft_sec": 0.42,
      "freq_branch_sec": 3.21,
      "time_branch_sec": 2.89,
      "istft_sec": 0.31,
      "full_pipeline_sec": 8.12,
      "model_load_sec": 2.1,
      "memory_peak_mb": 1840
    },
    "pytorch": {
      "stft_sec": 0.38,
      "freq_branch_sec": 4.87,
      "time_branch_sec": 3.92,
      "istft_sec": 0.29,
      "full_pipeline_sec": 11.20,
      "model_load_sec": 3.2,
      "memory_peak_mb": 2910
    }
  }
}
```

**CI integration:**

- Runs on push to `main` only (not PRs - too slow)
- Compares to last baseline entry
- Warns if CoreML throughput regresses >10% or memory increases >15%
- Auto-commits new entry to `baseline.json`

### 5. Dependabot

**`.github/dependabot.yml`:**

```yaml
version: 2
updates:
  # Swift Package Manager
  - package-ecosystem: "swift"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 3

  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 3

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
```

**Settings rationale:**

- Weekly for code deps catches security updates without daily noise
- Monthly for Actions since they change less often
- Limit of 3 PRs prevents flood if ignored

### 6. GitHub Pages

Publish parity reports and benchmark history automatically.

**Site structure:**

```
https://zakkeown.github.io/HTDemucsCoreML/
├── index.html              # Latest parity report
├── benchmarks/
│   ├── latest.html         # Latest benchmark comparison
│   └── history.html        # Graph of metrics over time
└── archive/
    └── YYYY-MM-DD.html     # Historical reports
```

**Deployment:**

```yaml
- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v4
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./tests/parity/report
```

## Files to Create/Modify

```
.
├── Makefile                          # NEW
├── .pre-commit-config.yaml           # NEW
├── .swiftformat                      # NEW (config)
├── .github/
│   ├── dependabot.yml                # NEW
│   └── workflows/
│       └── ci.yml                    # REPLACE parity-tests.yml
└── benchmarks/
    ├── run_benchmark.py              # NEW
    ├── compare.py                    # NEW
    └── baseline.json                 # NEW (auto-updated by CI)
```

## Implementation Order

1. **Makefile** - Immediate developer experience improvement
2. **Pre-commit hooks** - Quality gates before CI
3. **GitHub Actions restructure** - Parallel jobs and caching
4. **Dependabot** - Low effort, immediate value
5. **Benchmarks** - Requires instrumentation in parity tests
6. **GitHub Pages** - Final polish, depends on reports existing

## Success Criteria

- [ ] `make test` runs full test suite
- [ ] `make benchmark` produces comparison table
- [ ] Pre-commit hooks block commits with formatting issues
- [ ] CI completes in <5 minutes
- [ ] Dependabot opens weekly PRs
- [ ] Parity reports visible at GitHub Pages URL
- [ ] Benchmark history shows trend over 3+ data points
