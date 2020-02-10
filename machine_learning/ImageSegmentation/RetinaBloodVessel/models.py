import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from collections import OrderedDict

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels=in_channels, features=features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features *2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features *2, features *4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features *4, features *8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNet._block(features *8, features *16, name="bottleneck")
        # Decoder networks do convolutions and then upsample the image through the transpose
        # convolutions which learn how to upsample images with fractional strides( zero padded )
        self.upconv4 = nn.ConvTranspose2d(features *16, features *8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block(features *16, features *8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features *8, features *4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block(features *8, features *4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features *4, features *2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block(features *4, features *2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features *2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features *2, features, name="dec1")
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        # print("Bottleneck: ", bottleneck.shape)
        dec4 = self.upconv4(bottleneck)
        # print("DEC4: ", dec4.shape, " ENC4: ", enc4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1) # This combines encoder4 and decoder 4
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1) ) *255
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential( OrderedDict( [
            (name +"conv1",
             nn.Conv2d(in_channels=in_channels,
                       out_channels=features,
                       kernel_size=3,
                       padding=1,
                       bias=False),),
            (name +"norm1", nn.BatchNorm2d(num_features=features)),
            (name +"relu1", nn.ReLU(inplace=True)),
            (name +"conv2",
             nn.Conv2d(in_channels=features,
                       out_channels=features,
                       kernel_size=3,
                       padding=1,
                       bias=False),),
            (name +"norm2", nn.BatchNorm2d(num_features=features)),
            (name +"relu2", nn.ReLU(inplace=True)),
        ]))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.relu(out)
        return out


class RUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(RUNet, self).__init__()
        features = init_features
        self.encoder1 = ResidualBlock(in_channels=in_channels, out_channels=features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(in_channels=features, out_channels=features*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(in_channels=features*2, out_channels=features*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ResidualBlock(in_channels=features*4, out_channels=features*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ResidualBlock(in_channels=features*8, out_channels=features*16)
        # Decoder networks do convolutions and then upsample the image through the transpose
        # convolutions which learn how to upsample images with fractional strides( zero padded )
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(in_channels=features*16, out_channels=features*8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(in_channels=features*8, out_channels=features*4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(in_channels=features*4, out_channels=features*2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(in_channels=features*2, out_channels=features)
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        # print("Bottleneck: ", bottleneck.shape)
        dec4 = self.upconv4(bottleneck)
        # print("DEC4: ", dec4.shape, " ENC4: ", enc4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1) # This combines encoder4 and decoder 4
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1) ) *255