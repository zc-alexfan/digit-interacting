""" Full assembly of the parts to form the complete network """
import torch.nn as nn
import src.nets.unet_parts as parts


class MiniUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = parts.DoubleConv(n_channels, 64)
        self.down1 = parts.Down(64, 128)
        self.down2 = parts.Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = parts.Down(256, 512 // factor)

        self.up1 = parts.Up(512, 256 // factor, bilinear)
        self.up2 = parts.Up(256, 128 // factor, bilinear)
        self.up3 = parts.Up(128, 64, bilinear)
        self.outc = parts.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
