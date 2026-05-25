import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from src.config_handler import load_config
from src.models.custom_resnet import get_model
from src.utils.security_validator import SecurityValidator

class InferenceEngine:
    """
    Production-ready asynchronous inference engine.
    Stateless, secure, and optimized for million-user request streams.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model in Inference Mode
        self.model = get_model(num_classes=self.config.dataset.num_classes).to(self.device)
        self._load_best_checkpoint()
        self.model.eval()
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _load_best_checkpoint(self):
        # Graceful fallback for initialized models
        path = self.config.infrastructure.model_save_path
        if torch.cuda.is_available() is False:
             # Load to CPU if local
             loc = 'cpu'
        else:
             loc = 'cuda'

        try:
            checkpoint = torch.load(path, map_location=loc)
            self.model.load_state_dict(checkpoint['model'])
            print(f"[INFO] Weight initialization complete. Path: {path}")
        except FileNotFoundError:
            print("[WARN] Checkpoint not found. Initializing with random weights.")

    @torch.inference_mode()
    def predict(self, raw_data: np.ndarray) -> dict:
        """
        Predicts class from raw image array with integrated security validation.
        Designed for high-concurrency IO.
        """
        # 1. Security & Sanitization Layer
        sanitized_tensor = SecurityValidator.sanitize_input(
            raw_data, 
            expected_shape=tuple(self.config.dataset.input_shape)
        ).unsqueeze(0).to(self.device)
        
        # 2. Forward Pass
        logits = self.model(sanitized_tensor)
        probs = F.softmax(logits, dim=1)
        
        # 3. Formatted Response
        conf, pred = torch.max(probs, 1)
        
        return {
            "prediction": self.classes[pred.item()],
            "confidence": float(conf.item()),
            "class_index": int(pred.item()),
            "status": "v1.0"
        }

if __name__ == "__main__":
    # Integration Example
    engine = InferenceEngine()
    sample_img = np.random.rand(3, 32, 32).astype(np.float32)
    result = engine.predict(sample_img)
    print(f"INF_RESULT: {result}")
