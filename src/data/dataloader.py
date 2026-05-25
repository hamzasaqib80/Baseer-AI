import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Optional
import os


class CIFAR10DataLoader:
    """
    Highly optimized CIFAR-10 Data Management System.
    Implements production-grade augmentation and performance hooks.
    """

    def __init__(self, config: any):
        self.config = config
        self.mean = config.dataset.mean
        self.std = config.dataset.std

        # Training Augmentations: Optimized for Residual Learning
        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(
                    32, padding=self.config.dataset.augmentation.crop_padding
                ),
                transforms.RandomHorizontalFlip(
                    p=self.config.dataset.augmentation.horizontal_flip_prob
                ),
                transforms.RandomRotation(
                    self.config.dataset.augmentation.rotation_degrees
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        # Test/Validation: Deterministic Normalization only
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Initializes and returns primary training and validation DataLoaders.
        """
        # Production check: Only attempt download if data folder is missing
        data_exists = os.path.exists(os.path.join("./data", "cifar-10-batches-py"))

        try:
            train_set = torchvision.datasets.CIFAR10(
                root="./data",
                train=True,
                download=not data_exists,
                transform=self.train_transform,
            )

            test_set = torchvision.datasets.CIFAR10(
                root="./data",
                train=False,
                download=not data_exists,
                transform=self.test_transform,
            )
        except Exception as e:
            print(
                f"CRITICAL: Failed to load dataset. If your local internet is timing out,"
            )
            print(
                f"please copy the 'data/' folder from your Google Colab workspace into this directory."
            )
            raise e

        # CPU/GPU Performance Optimization
        train_loader = DataLoader(
            train_set,
            batch_size=self.config.hyperparameters.batch_size,
            shuffle=True,
            num_workers=self.config.infrastructure.num_workers,
            pin_memory=self.config.infrastructure.pin_memory,
            drop_last=True,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=self.config.hyperparameters.batch_size,
            shuffle=False,
            num_workers=self.config.infrastructure.num_workers,
            pin_memory=self.config.infrastructure.pin_memory,
        )

        return train_loader, test_loader

    def get_tiny_loader(self, samples: int = 100) -> DataLoader:
        """
        Creates a lightweight loader for local CPU sanity testing.
        """
        full_train = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.test_transform
        )
        subset_indices = list(range(samples))
        subset = Subset(full_train, subset_indices)

        return DataLoader(
            subset, batch_size=10, shuffle=False, num_workers=0, pin_memory=False
        )
