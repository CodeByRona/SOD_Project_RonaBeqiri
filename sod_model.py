"""
sod_model.py - CNN Encoder-Decoder architecture for Salient Object Detection.

Architecture:
  Encoder: 3 convolutional blocks (3->32->64->128 channels) with BatchNorm,
           ReLU, and MaxPooling. Dropout2d(0.3) at the deepest block.
  Decoder: 3 transposed convolution blocks (128->64->32->16 channels) with
           BatchNorm and ReLU.
  Output:  1x128x128 saliency probability map via Sigmoid activation.
"""

import torch
import torch.nn as nn


class SODModel(nn.Module):
    """
    Convolutional Encoder-Decoder for Salient Object Detection.

    Takes a 3x128x128 RGB image and outputs a 1x128x128 saliency mask
    where each pixel value in [0, 1] represents the probability of
    belonging to the salient (foreground) object.
    """

    def __init__(self):
        super(SODModel, self).__init__()

        # ── Encoder (Downsampling) ─────────────────────────────────────────────
        # Each block: Conv2D -> BatchNorm -> ReLU -> MaxPool(2)
        # Spatial size halves at each block; channel depth doubles.
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)             # 128x128 -> 64x64
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)             # 64x64 -> 32x32
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),          # regularization at the bottleneck
            nn.MaxPool2d(2)             # 32x32 -> 16x16
        )

        # ── Decoder (Upsampling) ───────────────────────────────────────────────
        # Each block: ConvTranspose2D -> BatchNorm -> ReLU
        # Spatial size doubles at each block; channel depth halves.
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()                   # 16x16 -> 32x32
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()                   # 32x32 -> 64x64
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()                   # 64x64 -> 128x128
        )

        # ── Output layer ──────────────────────────────────────────────────────
        # 1x1 conv collapses channels to 1; Sigmoid maps to [0, 1]
        self.final_layer = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through encoder then decoder.

        Args:
            x (Tensor): Input batch of shape (N, 3, 128, 128).

        Returns:
            Tensor: Saliency map of shape (N, 1, 128, 128) with values in [0, 1].
        """
        # Encoder
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        return torch.sigmoid(self.final_layer(x))