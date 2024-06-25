import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Projector(nn.Module):
    def __init__(self, unet_scale, bottleneck_ch_in, bottleneck_ch_out, projection_dim):
        super(Projector, self).__init__()
        self.Bottleneck = conv_block(
            ch_in=bottleneck_ch_in//unet_scale, ch_out=bottleneck_ch_out//unet_scale)
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.Head = nn.Sequential(
            nn.Linear(bottleneck_ch_out//unet_scale, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        x = self.Bottleneck(x)
        x = self.GlobalAvgPool(x)
        x = torch.squeeze(x, dim=(-1, -2))
        x = self.Head(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, img_ch, scale):
        super(UNetEncoder, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64//scale)
        self.Conv2 = conv_block(ch_in=64//scale, ch_out=128//scale)
        self.Conv3 = conv_block(ch_in=128//scale, ch_out=256//scale)
        self.Conv4 = conv_block(ch_in=256//scale, ch_out=512//scale)
        self.Conv5 = conv_block(ch_in=512//scale, ch_out=1024//scale)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        return x5
