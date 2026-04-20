"""Tests for src/dataset.py"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from PIL import Image

from src.dataset import build_transforms, ISICDataset


def _make_dummy_dataset(tmp_path: Path, num_samples: int = 10) -> tuple:
    """
    Helper that creates a tiny fake ISIC dataset on disk.
    Returns (df, image_dir) ready to pass into ISICDataset.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    records = []
    for i in range(num_samples):
        image_name = f"ISIC_000{i:04d}"
        # create a random 300x300 RGB image
        img = Image.fromarray(np.uint8(np.random.rand(300, 300, 3) * 255))
        img.save(image_dir / f"{image_name}.jpg")
        records.append({"image": image_name, "label": i % 8})

    df = pd.DataFrame(records)
    return df, image_dir


def test_build_transforms_train_output_shape():
    """Training transform should resize any image to (3, 224, 224)."""
    transform = build_transforms(image_size=224, is_train=True)
    dummy     = Image.fromarray(np.uint8(np.random.rand(300, 300, 3) * 255))
    tensor    = transform(dummy)
    assert tensor.shape == (3, 224, 224)


def test_build_transforms_eval_output_shape():
    """Eval transform should resize any image to (3, 224, 224)."""
    transform = build_transforms(image_size=224, is_train=False)
    dummy     = Image.fromarray(np.uint8(np.random.rand(128, 256, 3) * 255))
    tensor    = transform(dummy)
    assert tensor.shape == (3, 224, 224)


def test_dataset_length(tmp_path):
    """ISICDataset __len__ should match the dataframe length."""
    df, image_dir = _make_dummy_dataset(tmp_path, num_samples=10)
    transform     = build_transforms(image_size=224, is_train=False)
    dataset       = ISICDataset(df, image_dir, transform)
    assert len(dataset) == 10


def test_dataset_item_shapes(tmp_path):
    """ISICDataset __getitem__ should return (tensor, int) with correct shapes."""
    df, image_dir = _make_dummy_dataset(tmp_path, num_samples=5)
    transform     = build_transforms(image_size=224, is_train=False)
    dataset       = ISICDataset(df, image_dir, transform)

    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)


def test_dataset_label_range(tmp_path):
    """All labels should be within the expected class range 0-7."""
    df, image_dir = _make_dummy_dataset(tmp_path, num_samples=10)
    transform     = build_transforms(image_size=224, is_train=False)
    dataset       = ISICDataset(df, image_dir, transform)

    for i in range(len(dataset)):
        _, label = dataset[i]
        assert 0 <= label <= 7, f"Unexpected label {label} at index {i}"