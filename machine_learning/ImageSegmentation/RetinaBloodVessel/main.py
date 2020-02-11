"""
    Author: Ben Hers
    Title: Retina Blood Vessel Segmentation using UNet and Resnet
    Acknowledgements: The Stare Project from UCSD for the dataset
    UNet inspiration: https://www.kaggle.com/mateuszbuda/brain-segmentation-pytorch
"""
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import os
import imageio
import random
from PIL import Image
import argparse
from sklearn.metrics import f1_score, precision_score,recall_score
from models import UNet, RUNet
import sys

# Argparser Name and Help Message
name = 'Retina Blood Vessel Segmentation Tool'
help = 'Train or Evaluate Retina Blood Vessel Segmentation Tool'


def add_arguments(parser):
    # Describe Model
    parser.add_argument('--describe', action='store_true', help='Just prints the model')
    # Select Device
    parser.add_argument('-d', '--device', default=0, type=int, help='Selects device(default=0), Set to -1 to force CPU')
    # Number of Threads for training and dataloaders
    parser.add_argument('--num-workers', default=0, type=int, help='Number of processes/workers for training and dataloaders')
    # UNet or RUnet
    parser.add_argument('--model', default="UNet", help="Select model to train either RUNet or UNet (Default=UNet)")
    # Train or Evaluate
    parser.add_argument('--action', default="Train", help="Choose to train model or evaluate model from .pt file and evaluate(Default=train) Enter eval for evaluating")


class RetinaDataset(Dataset):
    """
        Custom Dataset that can be passed to torchvision dataloader, loads images from directory and resizes them
    """
    def __init__(self, imageData, imageTarget, transform=None, randomSampling=True):
        self.to_tensor = transforms.ToTensor()
        self.images = []
        self.masks = []
        # Make this not fixed later
        self.randCrop = transforms.RandomCrop((512, 512))
        self.resize = transforms.Resize((512, 512))
        # Load original images
        for item in glob.glob(os.path.join(imageData, "*.png")):
            img = imageio.imread(item)
            self.images.append(img)
        # Load segmented images
        for item in glob.glob(os.path.join(imageTarget, "*.png")):
            img = imageio.imread(item)
            self.masks.append(img)
    
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        # To resize the image we need to load it into PIL format for PyTorch
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
    # In this next section we write 75% of the data into folders for training and rest in other folders for testing
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


def evalModel(evalArgs, loaderTrain, loaderTest, model, device, loss):
    f1scores = []
    precisions = []
    recalls = []
    mseloss = []
    loaders = {"train": loaderTrain, "test": loaderTest}
    # Important to switch model into eval mode
    model.eval()
    for i, data in enumerate(loaders['test']):
        img, mask = data
        # Send data to gpu if we have one
        img, mask = img.to(device), mask.to(device)
        # Predict a masl
        maskPred = model(img)
        # Calculate the loss between generated mask and actual
        maskLoss = loss(maskPred, mask)
        # Get loss as float
        lossFloat = maskLoss.float().data.item()
        # Have to send predicted mask to cpu and detach it
        npMaskPred = maskPred.cpu().detach().numpy()
        npMaskPred = np.reshape(npMaskPred, (512, 512, 1))
        npMaskPred = np.reshape(npMaskPred, (512, 512))
        print(np.max(npMaskPred))
        print(np.min(npMaskPred))
        img = img.cpu().detach().numpy()
        img = np.reshape(img, (3, 512, 512))
        img = img.transpose(1, 2, 0)
        imgActual = Image.fromarray(img.astype('uint8'))
        imgActual.save("outputImages/Org" + str(i) + ".png")
        npMaskPred = (npMaskPred > (np.max(npMaskPred) / 2)) * 255
        npMaskPred = Image.fromarray(npMaskPred.astype(dtype="uint8"))
        savestr = "Pred" + str(i)
        savestr1 = "Actual" + str(i)
        npMaskPred.save("outputImages/" + savestr + ".png")
        maskTarget = mask.cpu().numpy()
        maskTarget = np.reshape(maskTarget, (512, 512, 1))
        maskTarget = np.reshape(maskTarget, (512, 512))
        maskTarget = Image.fromarray(maskTarget.astype('uint8'))
        maskTarget.save("outputImages/" + savestr1 + ".png")
        npMaskPred = (npMaskPred > (np.max(npMaskPred) / 2))
        maskTarget = (maskTarget > (np.max(maskTarget) / 2))
        f1score = f1_score(maskTarget, npMaskPred, average="micro")
        precisionScore = precision_score(maskTarget, npMaskPred, average="micro")
        recallScore = recall_score(maskTarget, npMaskPred, average="micro")
        mseloss.append(lossFloat)
        f1scores.append(f1score)
        precisions.append(precisionScore)
        recalls.append(recallScore)
    print("F1 score: ", np.mean(f1scores), " Precision: ", np.mean(precisions), " Recall: ", np.mean(recalls))
    print("Loss: ", np.mean(mseloss))


def trainModel(trainArgs, loaderTrain, loaderTest, model, device):
    loss = nn.MSELoss()
    learningRate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    epochs = 300
    loaders = {"train": loaderTrain, "test": loaderTest}
    losses = []
    epochLosses = []
    for epoch in range(epochs):
        model.train()
        eLosses = []
        for i, data in enumerate(loaders['train']):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                maskPred = model(img)
                maskLoss = loss(maskPred, mask)
                losses.append(maskLoss)
                maskLoss.backward()
                optimizer.step()
                lossFloat = maskLoss.float().data.item()
                eLosses.append(lossFloat)
        epochLosses.append(np.mean(eLosses))
        print("Epoch: %d, Loss: %f" % (epoch, np.mean(eLosses)))
    if trainArgs.model == "unet":
        torch.save(model.state_dict(), 'unet.pt')
    else:
        torch.save(model.state_dict(), 'runet.pt')
    print("Finished Training -- Moving to Testing")
    plt.plot(epochLosses)
    plt.title("Loss per Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('MSELoss')
    plt.show()
    evalModel(trainArgs, loaderTrain, loaderTest, model, device, loss)


def main(argsMain):
    # Get arguments from arg
    numWorkers = argsMain.num_workers
    model = argsMain.model
    if model.lower() == "runet":
        print("Model: RUNet")
        model = RUNet(in_channels=3, out_channels=1, init_features=32)
    else:
        print("Model: UNet")
        model = UNet(in_channels=3, out_channels=1, init_features=32)
    if argsMain.describe:
        print(model)
        sys.exit()
    useCuda = torch.cuda.is_available()
    if useCuda is False:
        device = torch.device("cpu")
    else:
        if argsMain.device >= 0:
            # set to device number, still do else "cpu" just in case somehow we get here with only a cpu
            device = torch.device("cuda:"+str(argsMain.device) if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
            useCuda = False
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
                             shuffle=True, num_workers=numWorkers)
    loaderTest = DataLoader(testData, batch_size=1,
                             shuffle=True, num_workers=numWorkers)
    print("Cuda: ", useCuda)
    print("Device: ", device)
    if useCuda:
      model.to(device)
    action = argsMain.action
    if action.lower() == "eval":
        print("Evaluating")
        # Means we are evaluating
        if argsMain.model.lower() == "runet":
            model.load_state_dict(torch.load('runet.pt'))
            print(type(model))
        else:
            model.load_state_dict(torch.load('unet.pt'))
        evalModel(argsMain, loaderTrain, loaderTest, model, device, nn.MSELoss())
    else:
        print("Training......")
        trainModel(argsMain, loaderTrain, loaderTest, model, device)

if __name__=="__main__":
    parser = argparse.ArgumentParser(help)
    add_arguments(parser)
    args = parser.parse_args()
    main(args)