import torch
from torchvision import models
from models.resnet import Resnet18, Resnet34
from models.mobilenet import MobileNetV2

supported_models = ["Resnet18", "Resnet34", "Mobilenet", "Squeezenet", "Inception"]

def load_models(model_info, dataset):
    print("Loading Model...")
    model_name = model_info['name']
    model_pretrained = model_info['pretrained']
    in_channels = dataset.channels
    if model_name in supported_models:
        if model_name == "Resnet18":
            # TODO set task
            model = Resnet18(num_classes=model_info['num_classes'], in_channels=in_channels)
            # TODO add in functionality that loads all pretrained parameters except last layer
        elif model_name == "Resnet34":
            model = Resnet34(num_classes=model_info['num_classes'], in_channels=in_channels)
        elif model_name == "Mobilenet":
            model = MobileNetV2(num_classes=model_info['num_classes'], in_channels=in_channels)
        elif model_name == "Squeezenet":
            model = models.squeezenet1_1(pretrained=model_pretrained)
        elif model_name == "Inception":
            model = models.inception_v3(pretrained=model_pretrained)
        else:
            print("Unknown model... defaulting to resnet18")
            model = models.resnet18(pretrained=model_pretrained)
    else:
        print("Unknown model... defaulting to resnet18")
        model = models.resnet18(pretrained=model_pretrained)
    return model

# TODO add in class versions of this because based on task need different outputs