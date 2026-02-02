# Task 9: Code Reference

## Quick Reference for NumPy Bridge Implementation

### 1. Python Script: Load NPZ → JSON

**File:** `/Users/zakkeown/Code/HTDemucsCoreML/.worktrees/phase2b-coreml-integration/scripts/npz_to_json.py`

```python
#!/usr/bin/env python3
import sys
import json
import numpy as np

def main():
    npz_path = sys.argv[1]
    data = np.load(npz_path)

    result = {
        'audio': data['audio'].tolist(),
        'stft_real': data['stft_real'].tolist(),
        'stft_imag': data['stft_imag'].tolist()
    }

    json.dump(result, sys.stdout, separators=(',', ':'))

if __name__ == '__main__':
    main()
```

### 2. Swift Function: Load Fixtures in Tests

**File:** `/Users/zakkeown/Code/HTDemucsCoreML/.worktrees/phase2b-coreml-integration/Tests/HTDemucsKitTests/TestSignals.swift`

```swift
static func loadNPZFixture(name: String) throws -> (audio: [Float], real: [[Float]], imag: [[Float]]) {
    // 1. Resolve paths
    let projectRoot = URL(fileURLWithPath: #file)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    let npzPath = projectRoot
        .appendingPathComponent("Resources/GoldenOutputs/\(name).npz")
        .path

    let scriptPath = projectRoot
        .appendingPathComponent("scripts/npz_to_json.py")
        .path

    // 2. Verify files exist
    guard FileManager.default.fileExists(atPath: npzPath) else {
        throw NSError(domain: "TestSignals", code: 1,
                     userInfo: [NSLocalizedDescriptionKey: "Fixture not found: \(npzPath)"])
    }

    // 3. Run Python script with async pipe handlers (prevents deadlock)
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["python3", scriptPath, npzPath]

    let outputPipe = Pipe()
    let errorPipe = Pipe()
    process.standardOutput = outputPipe
    process.standardError = errorPipe

    var outputData = Data()
    var errorData = Data()

    outputPipe.fileHandleForReading.readabilityHandler = { handle in
        outputData.append(handle.availableData)
    }

    errorPipe.fileHandleForReading.readabilityHandler = { handle in
        errorData.append(handle.availableData)
    }

    try process.run()
    process.waitUntilExit()

    // Close handlers
    outputPipe.fileHandleForReading.readabilityHandler = nil
    errorPipe.fileHandleForReading.readabilityHandler = nil

    guard process.terminationStatus == 0 else {
        let errorMessage = String(data: errorData, encoding: .utf8) ?? "Unknown error"
        throw NSError(domain: "TestSignals", code: 3,
                     userInfo: [NSLocalizedDescriptionKey: "Python script failed: \(errorMessage)"])
    }

    // 4. Parse JSON
    struct NPZData: Codable {
        let audio: [Float]
        let stft_real: [[Float]]
        let stft_imag: [[Float]]
    }

    let decoder = JSONDecoder()
    let data = try decoder.decode(NPZData.self, from: outputData)

    return (audio: data.audio, real: data.stft_real, imag: data.stft_imag)
}
```

### 3. Using in Tests

**File:** `/Users/zakkeown/Code/HTDemucsCoreML/.worktrees/phase2b-coreml-integration/Tests/HTDemucsKitTests/PyTorchParityTests.swift`

```swift
func testSilenceMatchesPyTorch() throws {
    // Load golden reference from PyTorch
    let (audio, pytorchReal, pytorchImag) = try TestSignals.loadNPZFixture(name: "silence")

    // Compute Swift STFT
    let (swiftReal, swiftImag) = try fft.stft(audio)

    // Compare with tolerance
    let rtol: Float = 1e-5
    let atol: Float = 1e-6

    for (frameIdx, (sr, pr)) in zip(swiftReal, pytorchReal).enumerated() {
        for (binIdx, (sv, pv)) in zip(sr, pr).enumerated() {
            let tolerance = atol + rtol * abs(pv)
            let error = abs(sv - pv)
            XCTAssertLessThanOrEqual(error, tolerance)
        }
    }
}
```

## Available Fixtures

```
Resources/GoldenOutputs/
├── silence.npz      - 2 seconds of zeros
├── sine_440hz.npz   - 440Hz sine wave
└── white_noise.npz  - Random noise
```

Each fixture contains:
- `audio`: (88200,) samples at 44.1kHz
- `stft_real`: (83, 2049) - Real part of STFT
- `stft_imag`: (83, 2049) - Imaginary part of STFT

## Usage Example

```swift
// In any test
let (audio, real, imag) = try TestSignals.loadNPZFixture(name: "sine_440hz")
print("Loaded \(audio.count) audio samples")
print("STFT shape: \(real.count) × \(real[0].count)")
```

## Testing the Bridge

```bash
# Test Python script directly
python3 scripts/npz_to_json.py Resources/GoldenOutputs/silence.npz | head -20

# Run verification script
./verify_task9.sh

# Run parity tests (once XCTest is fixed)
swift test --filter PyTorchParityTests
```

## Key Implementation Details

### Why Async Handlers?

The JSON output is ~2MB per fixture. Using synchronous `readDataToEndOfFile()` after `waitUntilExit()` causes a deadlock because:

1. Process writes to stdout pipe
2. Pipe buffer fills up (~64KB)
3. Process blocks waiting for buffer to drain
4. We block waiting for process to exit
5. **DEADLOCK**

Solution: Read pipe asynchronously while process runs:
```swift
outputPipe.fileHandleForReading.readabilityHandler = { handle in
    outputData.append(handle.availableData)
}
```

### Why Subprocess vs FFI?

Considered options:
1. ✗ C FFI bridge - Complex, requires NumPy C API
2. ✗ Swift NumPy wrapper - Doesn't exist
3. ✓ **Subprocess + JSON** - Simple, portable, testable

Trade-off: ~200ms overhead per test acceptable for test environment.

## Dependencies

- Python 3 with NumPy
- Swift Foundation framework
- XCTest framework

## Troubleshooting

**Problem:** Process hangs
- **Solution:** Use async readability handlers

**Problem:** "Fixture not found"
- **Solution:** Check `#file` path resolution is correct

**Problem:** "Python script failed"
- **Solution:** Check Python 3 and NumPy are installed: `python3 -c "import numpy; print(numpy.__version__)"`

## Performance

| Operation | Time |
|-----------|------|
| Python script execution | ~150ms |
| JSON parsing in Swift | ~50ms |
| **Total per fixture** | **~200ms** |

Acceptable for test environment (3 fixtures × 200ms = 600ms overhead).
