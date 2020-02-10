import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import os, sys
import imageio
import random
from PIL import Image

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels=in_channels, features=features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features*2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features*2, features*4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features*4, features*8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNet._block(features*8, features*16, name="bottleneck")
		# Decoder networks do convolutions and then upsample the image through the transpose
		# convolutions which learn how to upsample images with fractional strides( zero padded )
        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block(features*16, features*8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block(features*8, features*4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block(features*4, features*2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features*2, features, name="dec1")
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
        return torch.sigmoid(self.conv(dec1))*255
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential( OrderedDict( [
            (name+"conv1",
             nn.Conv2d(in_channels=in_channels,
                       out_channels=features,
                       kernel_size=3,
                       padding=1,
                       bias=False),),
            (name+"norm1", nn.BatchNorm2d(num_features=features)),
            (name+"relu1", nn.ReLU(inplace=True)),
            (name+"conv2",
             nn.Conv2d(in_channels=features,
                       out_channels=features,
                       kernel_size=3,
                       padding=1,
                       bias=False),),
            (name+"norm2", nn.BatchNorm2d(num_features=features)),
            (name+"relu2", nn.ReLU(inplace=True)),
        ]))

class RetinaDataset(Dataset):
    def __init__(self, imageData, imageTarget, transform=None,randomSampling=True):
        self.to_tensor = transforms.ToTensor()
        self.images = []
        self.masks = []
        # Make this not fixed later
        self.randCrop = transforms.RandomCrop((512, 512))
        self.resize = transforms.Resize((512, 512))
        # Load original images
        for item in glob.glob(os.path.join(imageData, "*.png")):
            img = imageio.imread(item)
            name = item.split("\\")[1]
            self.images.append(img)
        # Load segmented images
        for item in glob.glob(os.path.join(imageTarget, "*.png")):
            img = imageio.imread(item)
            name = item.split("\\")[1]
            self.masks.append(img)
        assert(len(self.images)==len(self.masks), "Don't have the same amount of images and masks")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.fromarray(np.asarray(self.images[index]))
        mask = Image.fromarray(np.asarray(self.masks[index]))
        img = torch.from_numpy(np.array(self.resize(img), dtype=np.float32).transpose(2, 0, 1))
        mask = torch.from_numpy(np.array(self.resize(mask), dtype=np.float32))
        return img, mask
    
def trainTestSplit(imageData, imageTarget, trainImage, trainTarget, testImage, testTarget):
    images = []
    masks = []
    # Load original images
    for item in glob.glob(os.path.join(imageData, "*.png")):
        img = imageio.imread(item)
        name = item.split("\\")[1]
        images.append(img)
    # Load segmented images
    for item in glob.glob(os.path.join(imageTarget, "*.png")):
        img = imageio.imread(item)
        name = item.split("\\")[1]
        masks.append(img)
    trainImages = random.sample(images, int(len(images)*0.75))
    for item in trainImages:
        img = imageio.imread(imageData+item)
        imgTarget = imageio.imread(imageTarget+item)
        imageio.imwrite(trainImage+item, img)
        imageio.imwrite(trainTarget+item, imgTarget)
    # Find the rest
    for item in images:
        if item not in trainImages:
            print("Test: ", item)
            img = imageio.imread(imageData+item)
            imgTarget = imageio.imread(imageTarget+item)
            imageio.imwrite(testTarget+item, imgTarget)
            imageio.imwrite(testImage+item, img)

    

if __name__ == "__main__":
    print("Started Here")
    imageData = "Images/"
    imageTarget = "AnnotatedImages/"
    trainImages = "trainImages/"
    trainTargets = "trainTargets/"
    testImages = "testImages/"
    testTargets = "testTargets/"
    # Split data into training and test
    # trainTestSplit(imageData, imageTarget, trainImages, trainTargets, testImages, testTargets)
    # Make datasets with specific methods
    trainData = RetinaDataset(trainImages, trainTargets)
    testData = RetinaDataset(testImages, testTargets)
    # Convert datasets into PyTorch datasets
    batchSize = 3
    loaderTrain = DataLoader(trainData, batch_size=batchSize,
                             shuffle=True, num_workers=1)
    loaderTest = DataLoader(testData, batch_size=batchSize,
                             shuffle=True, num_workers=1)
    model = UNet(in_channels=3, out_channels=1, init_features=32)
    loss = nn.MSELoss()
    learningRate = 0.002
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    epochs = 100
    loaders = {"train": loaderTrain, "valid": loaderTest}
    losses = []
    epochLosses = []
    for epoch in range(epochs):
        model.train()
        eLosses = []
        for i, data in enumerate(loaders['train']):
            img, mask = data
            optimizer.zero_grad()
            print("Img shape: ", img.shape, " Mask shape: ", mask.shape)

            with torch.set_grad_enabled(True):
                maskPred = model(img)
                maskLoss = loss(maskPred, mask)
                losses.append(maskLoss)
                maskLoss.backward()
                optimizer.step()
                eLosses.append(maskLoss)
        epochLosses.append(eLosses)
        print("Epoch: %d, Loss: %f" %(epoch, np.mean(eLosses)))
    torch.save(model, 'unet.pt')

    for i, data in enumerate(testData):
        img, mask = data
        maskPred = model(img)
        maskLoss = loss(maskPred, mask)
        print("Loss: ", maskLoss)
        print(type(maskPred))
        imageio.imshow(maskPred)
            