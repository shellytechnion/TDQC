import torch
import torch.nn as nn
import torchvision.models as tvm

_EMBED_DIM = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


def build_frozen_cnn_encoder(encoder_name: str, device):
    """Build a frozen pretrained CNN encoder stripped of its final FC layer.

    Returns (encoder, embed_dim) where encoder outputs (B, embed_dim, 1, 1).
    """
    model = getattr(tvm, encoder_name)(weights="IMAGENET1K_V1")
    encoder = nn.Sequential(*list(model.children())[:-1])  # strip FC, keep avgpool
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval().to(device)
    return encoder, _EMBED_DIM[encoder_name]
