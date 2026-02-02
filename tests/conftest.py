"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def fixture_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent.parent / "test_fixtures"

@pytest.fixture(scope="session")
def test_metadata(fixture_dir):
    """Load test fixture metadata."""
    return np.load(fixture_dir / "metadata.npy", allow_pickle=True).item()

@pytest.fixture(scope="session", params=['silence', 'sine_440hz', 'white_noise'])
def test_case_name(request):
    """Parametrize tests across all test cases."""
    return request.param

@pytest.fixture(scope="session")
def test_audio(fixture_dir, test_case_name):
    """Load test audio input."""
    audio_np = np.load(fixture_dir / f"{test_case_name}_input.npy")
    return torch.from_numpy(audio_np)

@pytest.fixture(scope="session")
def expected_output(fixture_dir, test_case_name):
    """Load expected PyTorch Demucs output."""
    output_np = np.load(fixture_dir / f"{test_case_name}_output.npy")
    return torch.from_numpy(output_np)

@pytest.fixture(scope="session")
def htdemucs_model():
    """Load htdemucs_6s model (cached for session)."""
    from demucs.pretrained import get_model
    model = get_model('htdemucs_6s')
    model.eval()
    return model

@pytest.fixture
def assert_close():
    """Helper for tensor comparison with clear error messages."""
    def _assert_close(actual, expected, rtol=1e-5, atol=1e-7, name="tensors"):
        """Assert tensors are close with detailed error reporting."""
        if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(actual - expected)).item()
            mean_diff = torch.mean(torch.abs(actual - expected)).item()
            rel_error = mean_diff / (torch.mean(torch.abs(expected)).item() + 1e-8)

            msg = (
                f"\n{name} not close enough:\n"
                f"  Max absolute diff: {max_diff:.2e}\n"
                f"  Mean absolute diff: {mean_diff:.2e}\n"
                f"  Mean relative error: {rel_error:.2%}\n"
                f"  Tolerance: rtol={rtol}, atol={atol}"
            )
            raise AssertionError(msg)

    return _assert_close
