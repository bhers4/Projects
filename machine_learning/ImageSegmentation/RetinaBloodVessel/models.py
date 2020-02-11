import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from collections import OrderedDict

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        # Encoder block is pair of 2 conv3x3 blocks with normalization and relu
        self.encoder1 = UNet._block(in_channels=in_channels, features=features, name="enc1")
        self.encoder2 = UNet._block(features, features *2, name="enc2")
        self.encoder3 = UNet._block(features *2, features *4, name="enc3")
        self.encoder4 = UNet._block(features *4, features *8, name="enc4")
        # Max Pooling reduces the dimensionality of the image
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck is just the same as the encoder block
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
        # Encode
        enc1 = self.encoder1(x)
        # Pool and Encode
        enc2 = self.encoder2(self.pool1(enc1))
        # Pool and Encode 
        enc3 = self.encoder3(self.pool2(enc2))
        # Pool and Encode
        enc4 = self.encoder4(self.pool3(enc3))
        # Pool and Encode
        bottleneck = self.bottleneck(self.pool4(enc4))
        # Upsample through ConvTranspose
        dec4 = self.upconv4(bottleneck)
        # Add in result from encoder section to give larger context
        dec4 = torch.cat((dec4, enc4), dim=1)
        # Decode block
        dec4 = self.decoder4(dec4)
        # Upsample through ConvTranspose
        dec3 = self.upconv3(dec4)
        # Add in result from encoder section to give larger context
        dec3 = torch.cat((dec3, enc3), dim=1)
        # Decode block
        dec3 = self.decoder3(dec3)
        # Upsample through ConvTranspose
        dec2 = self.upconv2(dec3)
        # Add in result from encoder section to give larger context
        dec2 = torch.cat((dec2, enc2), dim=1)
        # Decode block
        dec2 = self.decoder2(dec2)
        # Upsample through ConvTranspose
        dec1 = self.upconv1(dec2)
        # Add in result form encoder section to give larger context
        dec1 = torch.cat((dec1, enc1), dim=1)
        # Final decode
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

# Does residual block that skips the 2 conv 3x3 blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResidualBlock, self).__init__()
        # 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,padding=1, bias=False)
        # Batch Normalization
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)
        # Activation Function
        self.relu = nn.ReLU(inplace=True)
        # Need a kernel of 1x1 to get the same number of features out
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Do 1x1 kernel to skip get dimensions to match
        residual = self.conv1x1(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # Add in skip connection before normalization
        out += residual
        out = self.norm2(out)
        # Perform activation function
        out = self.relu(out)
        return out


class RUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(RUNet, self).__init__()
        # See UNet/ResidualBlock classes for further explanation
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
        # TODO Try only doing ResNet blocks on encoder part
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
        # See UNet class for explanation
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
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