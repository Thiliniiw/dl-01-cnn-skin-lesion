"""Evaluation, metrics, and Grad-CAM visualisation for ISIC 2019.

This file has three jobs:
    Job 1 - EVALUATE   run model on test set, collect all predictions
    Job 2 - METRICS    classification report and confusion matrix
    Job 3 - GRAD-CAM   visualise what the model attends to per prediction
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm

from src.dataset import build_dataloaders, build_transforms
from src.model import SkinLesionClassifier
from src.utils import (
    get_device,
    get_logger,
    load_checkpoint,
    load_config,
    set_seed,
)

logger = get_logger(__name__)

# Human-readable class names for ISIC 2019
# Index matches the integer label in metadata.csv
CLASS_NAMES = [
    "MEL",   # 0 — Melanoma
    "NV",    # 1 — Melanocytic nevus
    "BCC",   # 2 — Basal cell carcinoma
    "AK",    # 3 — Actinic keratosis
    "BKL",   # 4 — Benign keratosis
    "DF",    # 5 — Dermatofibroma
    "VASC",  # 6 — Vascular lesion
    "SCC",   # 7 — Squamous cell carcinoma
]


# ─────────────────────────────────────────────────────────────────────────────
# JOB 1 — EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(config_path: str) -> None:
    """Load best checkpoint and evaluate on the held-out test set.

    Args:
        config_path: path to configs/config.yaml
    """
    cfg    = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    logger.info(f"Evaluating on device: {device}")

    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── LOAD DATA ─────────────────────────────────────────────────────────
    # We only need the test loader here
    # The _ discards train and val loaders we do not need
    _, _, test_loader = build_dataloaders(cfg)

    # ── LOAD MODEL FROM CHECKPOINT ────────────────────────────────────────
    # pretrained=False because we are loading OUR weights not ImageNet
    # The checkpoint contains the weights we saved during training
    model = SkinLesionClassifier(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    checkpoint_path = Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt"
    metadata        = load_checkpoint(model, checkpoint_path, device)
    logger.info(
        f"Loaded checkpoint from epoch {metadata['epoch']} "
        f"with val_acc {metadata['metric']:.4f}"
    )

    # ── COLLECT PREDICTIONS ───────────────────────────────────────────────
    model.eval()
    all_preds  = []
    all_labels = []

    # torch.no_grad() — no gradients needed for evaluation
    # saves memory and speeds up inference
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="evaluating"):
            logits = model(images.to(device))
            preds  = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # ── JOB 2: METRICS ───────────────────────────────────────────────────
    _save_classification_report(all_labels, all_preds)
    _save_confusion_matrix(all_labels, all_preds, figures_dir)

    # ── JOB 3: GRAD-CAM ──────────────────────────────────────────────────
    _save_gradcam(model, test_loader, device, cfg, figures_dir)

    logger.info(f"All outputs saved to {figures_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# JOB 2 — METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _save_classification_report(
    labels: list,
    preds: list,
) -> None:
    """Print per-class precision, recall, F1, and overall accuracy.

    Why these metrics and not just accuracy?
        Overall accuracy is misleading for imbalanced datasets.
        A model that always predicts NV (the dominant class) gets
        ~50% accuracy while being completely useless clinically.

        Precision, Recall, and F1 per class reveal this:
            Precision: of all times I predicted MEL, how often was I right?
            Recall:    of all actual MEL cases, how many did I catch?
            F1:        harmonic mean of precision and recall

        For medical diagnosis, Recall (sensitivity) is most important —
        missing a melanoma is far worse than a false alarm.

    Args:
        labels: true class indices
        preds:  predicted class indices
    """
    report = classification_report(
        labels,
        preds,
        target_names=CLASS_NAMES,
        digits=4,
    )
    logger.info(f"\nClassification Report:\n{report}")


def _save_confusion_matrix(
    labels: list,
    preds: list,
    figures_dir: Path,
) -> None:
    """Save confusion matrix as a PNG figure.

    What is a confusion matrix?
        A grid where rows = true classes, columns = predicted classes.
        The diagonal shows correct predictions.
        Off-diagonal shows where the model confuses one class for another.

        For ISIC this reveals clinically important patterns:
        e.g. Is the model confusing MEL (melanoma) with NV (benign nevus)?
        That specific confusion is dangerous and worth investigating.

    Args:
        labels:      true class indices
        preds:       predicted class indices
        figures_dir: where to save the PNG
    """
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues")

    ax.set_title("Confusion Matrix — ISIC 2019 Test Set", fontsize=14, pad=16)
    plt.tight_layout()

    save_path = figures_dir / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# JOB 3 — GRAD-CAM
#
# Grad-CAM (Gradient-weighted Class Activation Mapping) visualises WHICH
# PARTS of an image the model focused on when making a prediction.
#
# Why does this matter?
#   Without Grad-CAM you have a black box — you know what the model predicted
#   but not why. Grad-CAM opens the box:
#
#   Good model → highlights the actual lesion
#   Bad model  → highlights irrelevant areas (ruler, skin background)
#                This tells you the model learned spurious correlations
#                and will fail in the real world
#
# How it works:
#   1. Run a forward pass to get the prediction
#   2. Run a backward pass from that prediction to the last conv layer
#   3. Average the gradients across spatial dimensions
#   4. Use those averaged gradients to weight each feature map channel
#   5. The result is a heatmap showing which spatial regions mattered most
# ─────────────────────────────────────────────────────────────────────────────

def _save_gradcam(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: dict,
    figures_dir: Path,
    num_images: int = 6,
) -> None:
    """Generate and save Grad-CAM visualisations for a sample of test images.

    Shows two rows per image:
        Top row    → original image with true label
        Bottom row → same image with Grad-CAM heatmap overlay

    Args:
        model:       trained SkinLesionClassifier
        loader:      test DataLoader
        device:      cuda, mps, or cpu
        cfg:         config dict
        figures_dir: where to save the PNG
        num_images:  how many images to visualise
    """
    # ── TARGET LAYER ──────────────────────────────────────────────────────
    # Grad-CAM needs a convolutional layer to compute gradients against.
    # The last conv layer (layer4[-1]) produces the richest, most
    # semantically meaningful feature maps — it sees high-level concepts
    # like "lesion shape" rather than low-level "edges"
    target_layer = [model.backbone.layer4[-1]]

    # ── GET A BATCH OF TEST IMAGES ────────────────────────────────────────
    images, labels = next(iter(loader))
    images = images[:num_images]
    labels = labels[:num_images]

    # ── COMPUTE GRAD-CAM ──────────────────────────────────────────────────
    # targets=None means "explain the predicted class"
    # GradCAM returns a (num_images, H, W) array of heatmap values 0-1
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cams = cam(
            input_tensor=images.to(device),
            targets=None,
        )

    # ── BUILD THE FIGURE ──────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, num_images,
        figsize=(num_images * 3, 7),
    )
    fig.suptitle(
        "Grad-CAM Visualisations — Top: Original | Bottom: Model Attention",
        fontsize=12, y=1.01,
    )

    inv_transform = _build_inverse_transform()

    with torch.no_grad():
        preds = model(images.to(device)).argmax(dim=1).cpu()

    for i in range(num_images):
        # Reconstruct the original image from the normalised tensor
        raw_img = inv_transform(images[i]).permute(1, 2, 0).numpy()
        raw_img = np.clip(raw_img, 0, 1)

        # Overlay the Grad-CAM heatmap on the original image
        cam_img = show_cam_on_image(
            raw_img,
            grayscale_cams[i],
            use_rgb=True,
        )

        true_name = CLASS_NAMES[labels[i].item()]
        pred_name = CLASS_NAMES[preds[i].item()]
        correct   = true_name == pred_name

        # Top row — original image
        axes[0, i].imshow(raw_img)
        axes[0, i].set_title(f"True: {true_name}", fontsize=9)
        axes[0, i].axis("off")

        # Bottom row — Grad-CAM overlay
        # Green border = correct prediction, Red border = wrong prediction
        axes[1, i].imshow(cam_img)
        axes[1, i].set_title(
            f"Pred: {pred_name}",
            fontsize=9,
            color="green" if correct else "red",
        )
        axes[1, i].axis("off")

        # Add coloured border to make correct/wrong immediately obvious
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor("green" if correct else "red")
            spine.set_linewidth(2)

    plt.tight_layout()
    save_path = figures_dir / "gradcam.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Grad-CAM saved → {save_path}")


def _build_inverse_transform() -> T.Normalize:
    """Build a transform that reverses ImageNet normalisation.

    Why do we need this?
        Images stored as tensors are normalised to ~-1 to 1.
        matplotlib expects pixel values between 0 and 1.
        We must undo the normalisation before displaying.

    The inverse of Normalize(mean, std) is:
        Normalize(mean=[-mean/std], std=[1/std])
    """
    return T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229,       1 / 0.224,       1 / 0.225],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate skin lesion classifier on test set"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    evaluate(args.config)