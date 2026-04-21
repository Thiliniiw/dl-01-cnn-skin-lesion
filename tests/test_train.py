"""Tests for src/train.py"""

import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.train import run_epoch


def _make_dummy_loader(
    num_samples: int = 32,
    num_classes: int = 8,
    batch_size: int = 8,
) -> DataLoader:
    """Create a DataLoader with random tensors — no real images needed."""
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
    )


def test_run_epoch_train_returns_loss_and_accuracy():
    """run_epoch in train mode should return (loss, accuracy) floats."""
    model     = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 8),
    )
    loader    = _make_dummy_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    device    = torch.device("cpu")

    loss, acc = run_epoch(model, loader, criterion, optimizer, device, is_train=True)

    assert isinstance(loss, float), "Loss should be a float"
    assert isinstance(acc,  float), "Accuracy should be a float"
    assert loss > 0,        "Loss should be positive"
    assert 0.0 <= acc <= 1.0, "Accuracy should be between 0 and 1"


def test_run_epoch_val_does_not_update_weights():
    """run_epoch in val mode should not change model weights."""
    model     = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 8),
    )
    # Snapshot weights before validation pass
    weights_before = [p.clone() for p in model.parameters()]

    loader    = _make_dummy_loader()
    criterion = nn.CrossEntropyLoss()
    device    = torch.device("cpu")

    run_epoch(model, loader, criterion, None, device, is_train=False)

    # Confirm weights are identical after validation pass
    for before, after in zip(weights_before, model.parameters()):
        assert torch.allclose(before, after), \
            "Weights changed during validation — they should not"


def test_run_epoch_accuracy_range():
    """Accuracy should always be between 0.0 and 1.0."""
    model     = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 8),
    )
    loader    = _make_dummy_loader(num_samples=16, batch_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    device    = torch.device("cpu")

    _, acc = run_epoch(
        model, loader, criterion, optimizer, device, is_train=True
    )
    assert 0.0 <= acc <= 1.0


def test_run_epoch_loss_decreases_with_training():
    """Loss should decrease after several training steps on a small dataset."""
    torch.manual_seed(42)
    model     = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 8),
    )
    loader    = _make_dummy_loader(num_samples=16, batch_size=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    device    = torch.device("cpu")

    first_loss, _  = run_epoch(
        model, loader, criterion, optimizer, device, is_train=True
    )
    # Run several more epochs
    for _ in range(10):
        last_loss, _ = run_epoch(
            model, loader, criterion, optimizer, device, is_train=True
        )

    assert last_loss < first_loss, \
        f"Loss did not decrease: first={first_loss:.4f} last={last_loss:.4f}"