import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import os
import time
import json
from src.config_handler import load_config
from src.data.dataloader import CIFAR10DataLoader
from src.models.custom_resnet import get_model
from src.utils.security_validator import SecurityValidator


class Trainer:
    """
    Enterprise-grade training engine with automated resource management
    and strict performance auditing.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.device = self._init_device()

        # Initialize Data
        dataloader_factory = CIFAR10DataLoader(self.config)
        self.train_loader, self.val_loader = dataloader_factory.get_loaders()

        # Initialize Model & Security Audit
        self.model = get_model(num_classes=self.config.dataset.num_classes).to(
            self.device
        )
        SecurityValidator.audit_model_weights(self.model)

        # Optimization Suite
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.hyperparameters.learning_rate),
            weight_decay=float(self.config.hyperparameters.weight_decay),
        )

        # Scheduler: OneCycleLR is critical for hitting 92%+ accuracy
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=float(self.config.hyperparameters.learning_rate),
            steps_per_epoch=len(self.train_loader),
            epochs=self.config.hyperparameters.epochs,
        )

        self.best_acc = 0.0
        self.early_stop_counter = 0

    def _init_device(self) -> torch.device:
        if torch.cuda.is_available():
            print(
                f"STATUS: Remote Compute Detected. Utilizing NVIDIA {torch.cuda.get_device_name(0)}"
            )
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("STATUS: Local Workspace. Utilizing CPU for sanity checks.")
        return torch.device("cpu")

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Security Sanitize (Optional for training, critical for inference)
            # inputs = SecurityValidator.sanitize_input(inputs)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        acc = 100.0 * correct / total
        duration = time.time() - start_time

        print(
            f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {duration:.2f}s"
        )
        return avg_loss, acc

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    def save_checkpoint(self, acc: float):
        if not os.path.exists("models"):
            os.makedirs("models")

        state = {
            "model": self.model.state_dict(),
            "acc": acc,
        }
        torch.save(state, self.config.infrastructure.model_save_path)
        print(f"CHECKPOINT: Best model saved with accuracy {acc:.2f}%")

    def run(self):
        print(
            f"INIZIALIZING TRAINING: {self.config.project.name} v{self.config.project.version}"
        )

        for epoch in range(1, self.config.hyperparameters.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_acc = self.validate()

            print(f"VALIDATION: Acc: {val_acc:.2f}%")

            # Early Stopping and Checkpointing Logic
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(val_acc)
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if (
                    self.early_stop_counter
                    >= self.config.hyperparameters.early_stopping.patience
                ):
                    print(
                        f"EARLY STOPPING: No improvement for {self.early_stop_counter} epochs."
                    )
                    break

        print(f"FINISH: Best Validation Accuracy: {self.best_acc:.2f}%")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
