import pytest
import torch
import numpy as np
from src.utils.security_validator import SecurityValidator


def test_security_validator_shape_enforcement():
    """
    Ensures that the validator correctly identifies and blocks incorrect shapes.
    """
    # Valid shape (3, 32, 32)
    valid_data = np.random.rand(3, 32, 32).astype(np.float32)
    sanitized = SecurityValidator.sanitize_input(valid_data)
    assert sanitized.shape == torch.Size([3, 32, 32])

    # Invalid shape (3, 64, 64)
    invalid_data = np.random.rand(3, 64, 64).astype(np.float32)
    with pytest.raises(ValueError, match="Input shape mismatch"):
        SecurityValidator.sanitize_input(invalid_data)


def test_security_validator_domain_boundaries():
    """
    Ensures that adversarial pixel values are clamped.
    """
    outlier_data = (
        np.array([2.5, -1.2, 0.5])
        .reshape(3, 1, 1)
        .repeat(32, axis=1)
        .repeat(32, axis=2)
    )
    sanitized = SecurityValidator.sanitize_input(outlier_data)

    assert torch.max(sanitized) <= 3.0
    assert torch.min(sanitized) >= -3.0


def test_malicious_payload_detection():
    """
    Ensures NaN and Inf injection attacks are blocked.
    """
    malicious_data = (
        np.array([np.nan, np.inf, 0.5])
        .reshape(3, 1, 1)
        .repeat(32, axis=1)
        .repeat(32, axis=2)
    )
    with pytest.raises(ValueError, match="Malicious payload detected"):
        SecurityValidator.sanitize_input(malicious_data)
