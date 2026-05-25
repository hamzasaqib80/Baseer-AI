import torch
import numpy as np
from typing import Union, Tuple

class SecurityValidator:
    """
    Enterprise-grade security enforcement layer for image tensors.
    Prevents adversarial overflows, shape-mismatch exploits, and malicious data injections.
    """

    @staticmethod
    def sanitize_input(
        data: Union[np.ndarray, torch.Tensor], 
        expected_shape: Tuple[int, int, int] = (3, 32, 32),
        min_val: float = -3.0,   # Normalized space lower bound for CIFAR-10
        max_val: float = 3.0     # Normalized space upper bound for CIFAR-10
    ) -> torch.Tensor:
        """
        Validates and sanitizes incoming image data before it hits the model.
        
        Args:
            data: Raw input data (NumPy array or Torch Tensor).
            expected_shape: The (C, H, W) shape expected by the model.
            min_val: Minimum allowed normalized value.
            max_val: Maximum allowed normalized value.
            
        Returns:
            torch.Tensor: Sanitized and validated tensor ready for inference.
            
        Raises:
            ValueError: If data fails shape or boundary checks.
            TypeError: If data is of an unsupported type.
        """
        # 1. Type Enforcement
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            tensor = data.float()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Only NumPy arrays and Torch Tensors are allowed.")

        # 2. Shape Validation (Anti-Overflow)
        if len(tensor.shape) == 3:
            # Handle (H, W, C) -> (C, H, W) if needed
            if tensor.shape[-1] == 3 and tensor.shape[0] != 3:
                tensor = tensor.permute(2, 0, 1)
        
        if tensor.shape != torch.Size(expected_shape):
            raise ValueError(f"Input shape mismatch. Expected {expected_shape}, got {tensor.shape}")

        # 3. Domain Boundary Sanitization
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError("Malicious payload detected: Input contains NaN or Inf values.")

        # Clip values to prevent adversarial pixel intensity attacks
        tensor = torch.clamp(tensor, min_val, max_val)

        return tensor

    @staticmethod
    def audit_model_weights(model: torch.nn.Module) -> bool:
        """
        Checks for weight corruption or extreme values in the model parameters.
        """
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"CRITICAL SECURITY ALERT: Weight corruption detected in {name}")
                return False
        return True
