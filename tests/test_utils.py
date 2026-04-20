"""Tests for src/utils.py"""

import torch
import pytest
from pathlib import Path

from src.utils import load_config, set_seed, get_device, save_checkpoint, load_checkpoint


def test_load_config_returns_dict(tmp_path):
    """load_config should parse a valid YAML file into a dict."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("seed: 42\ndata:\n  batch_size: 32\n")
    cfg = load_config(str(config_file))
    assert isinstance(cfg, dict)
    assert cfg["seed"] == 42
    assert cfg["data"]["batch_size"] == 32


def test_load_config_raises_on_missing_file():
    """load_config should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent/config.yaml")


def test_set_seed_reproducibility():
    """Same seed should produce identical random tensors."""
    set_seed(42)
    tensor_a = torch.randn(3, 3)
    set_seed(42)
    tensor_b = torch.randn(3, 3)
    assert torch.allclose(tensor_a, tensor_b)


def test_get_device_returns_torch_device():
    """get_device should always return a valid torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")


def test_save_and_load_checkpoint(tmp_path):
    """Checkpoint saved by save_checkpoint should be loadable by load_checkpoint."""
    model = torch.nn.Linear(4, 2)
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    save_checkpoint(model, checkpoint_path, epoch=1, metric=0.95)

    # Load into a fresh model
    new_model = torch.nn.Linear(4, 2)
    device = torch.device("cpu")
    metadata = load_checkpoint(new_model, checkpoint_path, device)

    assert metadata["epoch"] == 1
    assert metadata["metric"] == 0.95
    # Weights should match
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)