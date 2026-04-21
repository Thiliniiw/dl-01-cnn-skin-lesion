"""Tests for src/model.py"""

import pytest
import torch

from src.model import SkinLesionClassifier, BACKBONE_REGISTRY


def test_forward_pass_output_shape():
    """Model output should be (batch_size, num_classes)."""
    model  = SkinLesionClassifier(
        backbone="resnet50",
        num_classes=8,
        pretrained=False,   # False so test runs without downloading weights
        dropout=0.4,
    )
    dummy  = torch.randn(4, 3, 224, 224)  # batch of 4 images
    output = model(dummy)
    assert output.shape == (4, 8), \
        f"Expected (4, 8) but got {output.shape}"


def test_output_is_logits_not_probabilities():
    """Output should be raw logits — values outside 0-1 range expected."""
    model  = SkinLesionClassifier(
        backbone="resnet50", num_classes=8,
        pretrained=False, dropout=0.0,
    )
    dummy  = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    # Logits are not constrained to 0-1
    # If all values were 0-1 the model might be outputting probabilities
    # which would mean softmax was accidentally applied inside forward()
    assert not (output >= 0).all() or not (output <= 1).all(), \
        "Output looks like probabilities — forward() should return raw logits"


def test_invalid_backbone_raises_error():
    """Passing an unsupported backbone name should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        SkinLesionClassifier(
            backbone="vgg16",   # not in registry
            num_classes=8,
            pretrained=False,
            dropout=0.4,
        )


def test_classifier_head_replaced():
    """Final layer should output num_classes features not 1000."""
    model = SkinLesionClassifier(
        backbone="resnet50", num_classes=8,
        pretrained=False, dropout=0.4,
    )
    # Access the replaced final layer
    final_layer = model.backbone.fc
    # It should be our Sequential(Dropout, Linear) not the original Linear
    assert isinstance(final_layer, torch.nn.Sequential), \
        "Final layer should be nn.Sequential (Dropout + Linear)"
    # The Linear inside should output 8 classes
    linear = final_layer[1]
    assert linear.out_features == 8, \
        f"Expected 8 output features but got {linear.out_features}"


def test_dropout_is_applied():
    """Dropout should be the first element of the classification head."""
    model = SkinLesionClassifier(
        backbone="resnet50", num_classes=8,
        pretrained=False, dropout=0.4,
    )
    final_layer = model.backbone.fc
    dropout_layer = final_layer[0]
    assert isinstance(dropout_layer, torch.nn.Dropout), \
        "First element of head should be Dropout"
    assert dropout_layer.p == 0.4, \
        f"Expected dropout p=0.4 but got {dropout_layer.p}"


def test_backbone_registry_has_resnet50():
    """BACKBONE_REGISTRY must contain resnet50."""
    assert "resnet50" in BACKBONE_REGISTRY, \
        "resnet50 must be in BACKBONE_REGISTRY"