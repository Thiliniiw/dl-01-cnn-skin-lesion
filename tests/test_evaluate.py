"""Tests for src/evaluate.py"""

import numpy as np
import pytest
import torch
from pathlib import Path

from src.evaluate import (
    _build_inverse_transform,
    _save_confusion_matrix,
    _save_classification_report,
    CLASS_NAMES,
)


def test_class_names_length():
    """CLASS_NAMES should have exactly 8 entries matching ISIC 2019."""
    assert len(CLASS_NAMES) == 8, \
        f"Expected 8 class names but got {len(CLASS_NAMES)}"


def test_inverse_transform_reverses_normalisation():
    """Inverse transform should map normalised tensors back to 0-1 range."""
    import torchvision.transforms as T

    # Simulate a normalised image tensor
    normalize   = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    raw_image   = torch.rand(3, 224, 224)          # random image 0-1
    normalised  = normalize(raw_image)              # push to ~-1 to 1
    inv         = _build_inverse_transform()
    recovered   = inv(normalised)                   # should return to ~0-1

    assert recovered.min() >= -0.05, "Recovered image too dark"
    assert recovered.max() <=  1.05, "Recovered image too bright"


def test_save_confusion_matrix_creates_file(tmp_path):
    """_save_confusion_matrix should save a PNG file."""
    labels = [0, 1, 2, 3, 4, 5, 6, 7] * 4
    preds  = [0, 1, 2, 3, 4, 5, 6, 7] * 4

    _save_confusion_matrix(labels, preds, tmp_path)

    saved = tmp_path / "confusion_matrix.png"
    assert saved.exists(), "confusion_matrix.png was not created"
    assert saved.stat().st_size > 0, "confusion_matrix.png is empty"


def test_save_classification_report_runs_without_error():
    """_save_classification_report should not raise any exceptions."""
    labels = [0, 1, 2, 3, 4, 5, 6, 7] * 4
    preds  = [0, 1, 2, 3, 4, 5, 6, 7] * 4
    # Should run without raising
    _save_classification_report(labels, preds)


def test_classification_report_handles_wrong_predictions():
    """Report should still work when all predictions are wrong."""
    labels = [0, 1, 2, 3, 4, 5, 6, 7]   # all 8 classes present
    preds  = [1, 2, 3, 4, 5, 6, 7, 0]   # all wrong — shifted by one
    _save_classification_report(labels, preds)