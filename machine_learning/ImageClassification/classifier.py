import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models


# Going to set up template class for tensorboard
class TensorBoard:
    def __init__(self, runName):
        self.runName = runName
        self.setUpTensorBoard()

    def setUpTensorBoard(self):
        from torch.utils.tensorboard import SummaryWriter
        import datetime
        today = datetime.datetime.now()
        print(today)
        writerDate = "m" + str(today.month) + "d" + str(today.day) + "h" + str(today.hour) + "m" + str(today.minute)
        self.writer = SummaryWriter('runs/'+self.runName+writerDate)
        #self.writer = SummaryWriter('runs/'+self.runName)

    def add_image(self, imageTitle, imageIn):
        imgGrid = torchvision.utils.make_grid(imageIn)
        self.writer.add_image(imageTitle, imgGrid)

    def add_model(self, model, inputToModel):
        self.writer.add_graph(model, inputToModel)

    def add_training_loss(self, trainingLoss, xVal):
        self.writer.add_scalar('Loss/train', trainingLoss, xVal)
        self.writer.flush()

    def add_scalar(self, scalarCategory, val, xVal):
        self.writer.add_scalar(scalarCategory, val, xVal)

    def add_hparam(self, paramDict):
        self.writer.add_hparams(paramDict, {})


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LargeCNN(nn.Module):
    def __init__(self, inputChannels=3, outputChannels=10):
        super(LargeCNN, self).__init__()
        # Specify Number of Channels in different parts of the network
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.pool = nn.MaxPool2d(2, 2)
        self.ReLU = nn.ReLU(inplace=True)
        self.numCNNLayers = 5
        self.createNetwork()

    def createNetwork(self):
        self.conv1 = nn.Conv2d(in_channels=self.inputChannels, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(in_features=16*7*7, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3= nn.Linear(in_features=100, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=100)
        self.fcOut = nn.Linear(in_features=100, out_features=self.outputChannels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.ReLU(self.conv2(x))
        x = self.ReLU(self.conv3(x))
        x = self.ReLU(self.conv4(x))
        x = self.ReLU(self.conv5(x))
        x = x.view(-1, 16*7*7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        out = self.fcOut(x)
        return out


def calcCIFAR10Accuracy(testLoader, model, modelName, training, cuda, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if training:
                if cuda:
                    images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(modelName + ': Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    acc = correct/total
    return acc

def calcClassAccuracies(testloader, model, training, cuda, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if training:
                if cuda:
                    images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    return class_correct, class_total

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# Training Data
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True,num_workers=0)
# Validation Data
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
# Class Names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# List of Models to Evaluate
evalModels = ["CNN", "LargeCNN", "Resnet18", "Resnet50"]
#evalModels = ["CNN"]
results = {}
# Epochs
epochs = 10
# Get Random Training Images
dataiter = iter(trainloader)
images, labels = dataiter.__next__()
print(images.shape)
# Train Models
for item in evalModels:
    # Make the models
    if item == "CNN":
        model = CNN()
    elif item == "LargeCNN":
        model = LargeCNN()
    elif item == "Resnet18":
        model = models.resnet18()
    elif item == "Resnet50":
        model = models.resnet50()
    else:
        import sys
        import time
        print("Error in Choosing Model")
        time.sleep(1)
        sys.exit()
    # Make the TensorBoard
    board = TensorBoard(item)
    board.add_hparam({'Model':item, 'Optim': "Adam", "Epochs": epochs})
    board.add_model(model, images)
    board.add_image(item+"_images", images)
    # Choose Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print("Starting Training: ", item, " With Adam and CE Loss")
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Cuda")
        model.to(device)
        # Train Network
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # Get inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # Zero optimizer
                optimizer.zero_grad()
                # Training step
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    board.add_scalar('Loss/Train', running_loss/2000, epoch*len(trainloader)+i)
                    acc = calcCIFAR10Accuracy(testloader, model, item, True, torch.cuda.is_available(), device)
                    board.add_scalar('Performance/Accuracy', acc, epoch*len(trainloader)+i)
                    running_loss = 0.0
    else:
        # Train Network
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # Get inputs
                inputs, labels = data
                # Zero optimizer
                optimizer.zero_grad()
                # Training step
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    board.add_scalar('Loss/Train', running_loss / 2000, epoch * len(trainloader) + i)
                    acc = calcCIFAR10Accuracy(testloader, model, item)
                    board.add_scalar('Performance/Accuracy', acc, epoch * len(trainloader) + i)
                    running_loss = 0.0

    acc = calcCIFAR10Accuracy(testloader, model, item, True, torch.cuda.is_available(), device)
    classCorrect, classTotal = calcClassAccuracies(testloader, model, True, torch.cuda.is_available(), device)
    assert len(classCorrect) == len(classTotal)
    for i in range(len(classCorrect)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * classCorrect[i] / classTotal[i]))
