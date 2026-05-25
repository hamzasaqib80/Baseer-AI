import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.config_handler import load_config
from src.data.dataloader import CIFAR10DataLoader
from src.models.custom_resnet import get_model
import os


class Evaluator:
    """
    Post-training forensic evaluation suite.
    Generates high-fidelity metrics and visualizes model performance.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Data
        _, self.test_loader = CIFAR10DataLoader(self.config).get_loaders()

        # Load Model
        self.model = get_model(num_classes=self.config.dataset.num_classes).to(
            self.device
        )
        self._load_best_checkpoint()

        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    def _load_best_checkpoint(self):
        if os.path.exists(self.config.infrastructure.model_save_path):
            checkpoint = torch.load(
                self.config.infrastructure.model_save_path, map_location=self.device
            )  # nosec B301
            self.model.load_state_dict(checkpoint["model"])
            print(f"LOADED: Best model with Accuracy: {checkpoint['acc']:.2f}%")
        else:
            print("WARNING: No checkpoint found. Evaluating initialized model.")

    def run_evaluation(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        print("RUNNING EVALUATION...")
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 1. Classification Report
        report = classification_report(
            all_targets, all_preds, target_names=self.classes
        )
        print("\nCLASSIFICATION REPORT:")
        print(report)

        with open("monitoring/classification_report.txt", "w") as f:
            f.write(report)

        # 2. Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        self._plot_confusion_matrix(cm)

        print("EVALUATION COMPLETE. Artifacts saved to 'monitoring/' directory.")

    def _plot_confusion_matrix(self, cm: np.ndarray):
        if not os.path.exists("monitoring"):
            os.makedirs("monitoring")

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.classes,
            yticklabels=self.classes,
        )
        plt.title("CIFAR-10 Confusion Matrix - Enterprise ResNet")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig("monitoring/confusion_matrix.png")
        plt.close()


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_evaluation()
