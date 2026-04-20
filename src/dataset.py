"""Dataset loading, splitting, and augmentation for ISIC 2019."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.utils import get_logger

logger = get_logger(__name__)

# ImageNet stats — used because ResNet was pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, is_train: bool) -> transforms.Compose:
    """
    Build augmentation pipeline for training or evaluation.

    Training applies random augmentations to improve generalisation.
    Evaluation only resizes and normalises — no randomness.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ISICDataset(Dataset):
    """PyTorch Dataset for the ISIC 2019 skin lesion classification task."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        transform: transforms.Compose,
    ) -> None:
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        row        = self.df.iloc[idx]
        image_path = self.image_dir / f"{row['image']}.jpg"
        image      = Image.open(image_path).convert("RGB")
        label      = int(row["label"])
        return self.transform(image), label


def _build_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to handle class imbalance.

    Rare classes get sampled more frequently so the model sees
    a balanced distribution during training.
    """
    class_counts   = np.bincount(train_df["label"].values)
    sample_weights = 1.0 / class_counts[train_df["label"].values]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders from config.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    root      = Path(cfg["data"]["root_dir"])
    df        = pd.read_csv(root / "metadata.csv")
    image_dir = root / "images"

    # stratified split — preserves class distribution in each split
    train_df, test_df = train_test_split(
        df,
        test_size=cfg["data"]["test_split"],
        stratify=df["label"],
        random_state=cfg["seed"],
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg["data"]["val_split"],
        stratify=train_df["label"],
        random_state=cfg["seed"],
    )

    logger.info(
        f"Dataset splits — "
        f"train: {len(train_df)} | "
        f"val: {len(val_df)} | "
        f"test: {len(test_df)}"
    )

    image_size    = cfg["data"]["image_size"]
    train_dataset = ISICDataset(train_df, image_dir, build_transforms(image_size, is_train=True))
    val_dataset   = ISICDataset(val_df,   image_dir, build_transforms(image_size, is_train=False))
    test_dataset  = ISICDataset(test_df,  image_dir, build_transforms(image_size, is_train=False))

    # pin_memory only works with CUDA — not supported on MPS
    pin_memory    = torch.cuda.is_available()
    loader_kwargs = {
        "num_workers": cfg["data"]["num_workers"],
        "pin_memory" : pin_memory,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        sampler=_build_weighted_sampler(train_df),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader