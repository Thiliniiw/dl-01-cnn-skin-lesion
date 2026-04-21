"""Utility functions shared across the project."""

import random
import logging
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a stdout logger with a consistent format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Return the best available device.
    - CUDA  → NVIDIA GPU (Linux / Windows)
    - MPS   → Apple Silicon (M1 / M2 / M3)
    - CPU   → fallback
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    epoch: int,
    metric: float,
) -> None:
    """Save model weights and metadata to a checkpoint file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metric": metric,
        },
        path,
    )
    logger = get_logger(__name__)
    logger.info(f"Checkpoint saved → {path} (epoch {epoch}, metric {metric:.4f})")


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
) -> dict:
    """Load a checkpoint into a model and return the metadata dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def log_memory_usage(device: torch.device) -> None:
    """Log current GPU memory usage for CUDA and MPS devices."""
    logger = get_logger(__name__)
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved  = torch.cuda.memory_reserved(device) / 1e9
        logger.info(f"CUDA memory — allocated: {allocated:.2f}GB | reserved: {reserved:.2f}GB")
    elif device.type == "mps":
        allocated = torch.mps.current_allocated_memory() / 1e9
        logger.info(f"MPS memory  — allocated: {allocated:.2f}GB")
    else:
        logger.info("Running on CPU — no GPU memory to report")