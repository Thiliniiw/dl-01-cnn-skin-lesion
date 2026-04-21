"""Training loop for ISIC 2019 skin lesion classification.

This file has four jobs:
    Job 1 - SETUP    load everything needed before training starts
    Job 2 - EPOCH    run one full pass through a dataloader
    Job 3 - LOOP     repeat epochs, track best model, early stopping
    Job 4 - ENTRY    parse arguments and start training
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.model import SkinLesionClassifier
from src.utils import (
    get_device,
    get_logger,
    load_config,
    log_memory_usage,
    save_checkpoint,
    set_seed,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# JOB 2 — EPOCH
#
# Runs one complete pass through a dataloader.
# Used for both training and validation — the is_train flag controls
# whether gradients are computed and weights are updated.
#
# Why one function for both train and val?
#   The logic is almost identical — forward pass, loss, accuracy.
#   The only difference is whether we update weights.
#   One function keeps the code DRY (Don't Repeat Yourself).
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_train: bool,
) -> tuple[float, float]:
    """Run one epoch of training or validation.

    Args:
        model:     the SkinLesionClassifier
        loader:    DataLoader for train or val split
        criterion: loss function (CrossEntropyLoss)
        optimizer: AdamW optimiser (None during validation)
        device:    cuda, mps, or cpu
        is_train:  True for training, False for validation

    Returns:
        Tuple of (average_loss, accuracy) for this epoch
    """
    # model.train() enables dropout and batch normalisation training mode
    # model.eval()  disables them for consistent evaluation
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    # torch.set_grad_enabled(is_train):
    #   During training  → gradients are tracked (needed for backprop)
    #   During validation → gradients are NOT tracked (saves memory + speed)
    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(
            loader,
            desc="train" if is_train else "val  ",
            leave=False,
        ):
            # Move data to the same device as the model
            # If model is on MPS, data must also be on MPS
            images = images.to(device)
            labels = labels.to(device)

            # ── FORWARD PASS ──────────────────────────────────────────────
            # Pass images through the model to get predictions (logits)
            # logits shape: (batch_size, num_classes) = (32, 8)
            logits = model(images)

            # ── LOSS ─────────────────────────────────────────────────────
            # CrossEntropyLoss measures how wrong the predictions are
            # It expects: logits (batch, classes) and labels (batch,)
            # It internally applies softmax before computing the loss
            loss = criterion(logits, labels)

            # ── BACKWARD PASS + UPDATE (training only) ────────────────────
            if is_train:
                # Zero gradients from previous batch
                # Without this, gradients accumulate across batches
                optimizer.zero_grad()

                # Compute gradients via backpropagation
                # Walks backwards through every operation in the forward pass
                # and computes: d(loss)/d(weight) for every weight
                loss.backward()

                # Update weights using computed gradients
                # AdamW: weight = weight - learning_rate * gradient
                # (with momentum and weight decay adjustments)
                optimizer.step()

            # ── TRACK METRICS ─────────────────────────────────────────────
            # loss.item() extracts the scalar value from the tensor
            # multiply by len(labels) to get total loss for this batch
            # we divide by total samples at the end to get average
            total_loss += loss.item() * len(labels)

            # logits.argmax(dim=1): for each image pick the class
            # with the highest score — that is the prediction
            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += len(labels)

    average_loss = total_loss / total
    accuracy     = correct / total
    return average_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# JOB 3 — LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(config_path: str) -> None:
    """Full training loop with validation, checkpointing, and early stopping.

    Args:
        config_path: path to configs/config.yaml
    """
    # ── JOB 1: SETUP ─────────────────────────────────────────────────────
    cfg    = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    logger.info(f"Training on device: {device}")

    # Initialise Weights and Biases for experiment tracking
    # All metrics logged with wandb.log() appear on wandb.ai dashboard
    wandb.init(project="skin-lesion-cnn", config=cfg)

    # Build dataloaders and model from config
    train_loader, val_loader, _ = build_dataloaders(cfg)

    model = SkinLesionClassifier(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    # .to(device) moves ALL model parameters to the target device (MPS/CUDA/CPU)

    # ── LOSS FUNCTION ─────────────────────────────────────────────────────
    # CrossEntropyLoss = softmax + negative log likelihood in one step
    # Expects raw logits — do NOT apply softmax before passing to this
    criterion = nn.CrossEntropyLoss()

    # ── OPTIMISER ─────────────────────────────────────────────────────────
    # AdamW: Adam optimiser with decoupled weight decay
    # Weight decay adds a penalty for large weights — prevents overfitting
    # learning_rate controls how big each update step is
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ── LEARNING RATE SCHEDULER ───────────────────────────────────────────
    # CosineAnnealingLR gradually reduces learning rate following a cosine curve
    # Starts high (fast learning), ends low (fine adjustments)
    # Why? Large LR early on explores broadly, small LR later fine-tunes
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )

    # ── TRAINING STATE ────────────────────────────────────────────────────
    best_val_acc      = 0.0
    patience_counter  = 0
    checkpoint_path   = (
        Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt"
    )

    # ── JOB 3: LOOP ───────────────────────────────────────────────────────
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        logger.info(f"Epoch {epoch}/{cfg['training']['epochs']}")

        # Training pass
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )

        # Validation pass
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )

        # Step the scheduler after each epoch
        # This updates the learning rate for the next epoch
        scheduler.step()

        # Log all metrics to Weights and Biases
        wandb.log({
            "epoch"     : epoch,
            "train_loss": train_loss,
            "train_acc" : train_acc,
            "val_loss"  : val_loss,
            "val_acc"   : val_acc,
            "lr"        : scheduler.get_last_lr()[0],
        })

        logger.info(
            f"  train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
        )

        # Log memory on first epoch to confirm GPU is being used
        if epoch == 1:
            log_memory_usage(device)

        # ── CHECKPOINTING ─────────────────────────────────────────────────
        # Save model only when validation accuracy improves
        # This ensures best_model.pt always holds the best weights seen
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            save_checkpoint(model, checkpoint_path, epoch, val_acc)
            logger.info(f"  New best val_acc: {val_acc:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            logger.info(
                f"  No improvement. "
                f"Patience: {patience_counter}/"
                f"{cfg['training']['early_stopping_patience']}"
            )

        # ── EARLY STOPPING ────────────────────────────────────────────────
        # If val_acc has not improved for N epochs, stop training
        # Why? Further training is likely just overfitting
        if patience_counter >= cfg["training"]["early_stopping_patience"]:
            logger.info(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best val_acc: {best_val_acc:.4f}"
            )
            break

    wandb.finish()
    logger.info(f"Training complete. Best val_acc: {best_val_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# JOB 4 — ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train skin lesion classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    train(args.config)