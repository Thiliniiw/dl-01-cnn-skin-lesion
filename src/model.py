"""ResNet-50 model wrapper for ISIC 2019 skin lesion classification.

This file has three jobs:
    Job 1 - REGISTRY   BACKBONE_REGISTRY   lookup table of supported backbones
    Job 2 - WRAPPER    SkinLesionClassifier clean class that builds the model
    Job 3 - SURGERY    _replace_classifier  swap the final layer for our task
"""

from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.utils import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# JOB 1 — REGISTRY
#
# A dictionary mapping backbone name strings to their torchvision constructor,
# pretrained weights object, and the name of the final classification layer.
#
# Why a registry instead of hardcoding?
#   Swapping backbones becomes a one-line change in config.yaml.
#   Add a new backbone here once and it is available everywhere.
#   Keeps model.py open for extension but closed for modification —
#   a core software engineering principle (Open/Closed Principle).
#
# Structure of each entry:
#   "name" : (constructor_function, pretrained_weights, classifier_layer_name)
# ─────────────────────────────────────────────────────────────────────────────
BACKBONE_REGISTRY = {
    "resnet50": (
        models.resnet50,              # function that builds the model
        ResNet50_Weights.IMAGENET1K_V2,  # best available pretrained weights
        "fc",                         # name of the final layer inside ResNet
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# JOB 2 — WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class SkinLesionClassifier(nn.Module):
    """ResNet-50 fine-tuned for 8-class skin lesion classification.

    Inherits from nn.Module — the base class for ALL PyTorch models.
    Every model you ever build in PyTorch inherits from nn.Module.

    Two methods are required by nn.Module:
        __init__   → build the model architecture
        forward    → define how data flows through the model
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
    ) -> None:
        """Build the model by loading a pretrained backbone and replacing
        its final classification layer.

        Args:
            backbone:    name of the backbone e.g. 'resnet50'
            num_classes: number of output classes (8 for ISIC 2019)
            pretrained:  True to load ImageNet weights, False for random init
            dropout:     dropout probability before the final linear layer
        """
        # Always call super().__init__() first in any nn.Module subclass.
        # This initialises PyTorch's internal machinery for tracking
        # parameters, layers, and gradients.
        super().__init__()

        # ── STEP 1: Validate the backbone name ───────────────────────────
        # Fail early with a clear message rather than a cryptic error later.
        if backbone not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from: {list(BACKBONE_REGISTRY.keys())}"
            )

        # ── STEP 2: Look up the backbone in the registry ─────────────────
        model_fn, weights, classifier_attr = BACKBONE_REGISTRY[backbone]

        # ── STEP 3: Load the backbone with or without pretrained weights ──
        # weights=None means random initialisation — used during testing
        # so tests run fast without downloading 100MB of weights
        self.backbone = model_fn(weights=weights if pretrained else None)
        logger.info(
            f"Loaded {backbone} "
            f"({'pretrained ImageNet' if pretrained else 'random init'})"
        )

        # ── STEP 4: Replace the final classification layer ────────────────
        # This is the core of transfer learning — see Job 3 below
        self.backbone = _replace_classifier(
            model=self.backbone,
            classifier_attr=classifier_attr,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        """Define the forward pass — how data flows through the model.

        Args:
            x: input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            logits tensor of shape (batch_size, num_classes)
            Raw scores — NOT probabilities. CrossEntropyLoss expects logits.
        """
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# JOB 3 — SURGERY
#
# Replace the final classification layer of the pretrained backbone
# with a new head suited to our task.
#
# Why do we need to replace it?
#   ResNet-50 was trained on ImageNet which has 1000 classes.
#   Its final layer outputs 1000 scores — one per ImageNet class.
#   We have 8 classes (ISIC skin lesion types).
#   We must replace the final layer to output 8 scores instead.
#
# What stays the same?
#   All layers BEFORE the final one — these contain the learned
#   feature detectors (edges, textures, shapes, gradients).
#   These transfer directly from ImageNet to skin lesions.
#
# What changes?
#   Only the final layer — replaced with Dropout + Linear(in, 8)
# ─────────────────────────────────────────────────────────────────────────────

def _replace_classifier(
    model: nn.Module,
    classifier_attr: str,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    """Replace the final classification layer with a new task-specific head.

    Args:
        model:           the pretrained backbone (e.g. ResNet-50)
        classifier_attr: name of the layer to replace (e.g. 'fc' for ResNet)
        num_classes:     number of output classes for the new task
        dropout:         dropout probability before the linear layer

    Returns:
        model with the final layer replaced
    """
    # ── STEP 1: Find out how many features feed into the final layer ──────
    # ResNet-50's final layer takes in 2048 features.
    # We need to know this to build the replacement layer correctly.
    original_layer = getattr(model, classifier_attr)
    in_features    = original_layer.in_features
    logger.info(
        f"Replacing '{classifier_attr}' layer: "
        f"in_features={in_features} → out_features={num_classes}"
    )

    # ── STEP 2: Build the new classification head ─────────────────────────
    # Dropout → randomly zeroes some features during training
    #           prevents the model from relying too heavily on any
    #           single feature — reduces overfitting
    # Linear  → the actual classification layer
    #           maps 2048 features → num_classes scores
    new_head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    # ── STEP 3: Swap the layer in the model ──────────────────────────────
    # setattr(model, 'fc', new_head) is equivalent to: model.fc = new_head
    # We use setattr because the layer name comes from the registry string
    # and cannot be hardcoded.
    setattr(model, classifier_attr, new_head)

    return model