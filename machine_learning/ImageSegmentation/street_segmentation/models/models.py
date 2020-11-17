import torch
from torchvision import models
from models.resnet import Resnet18

supported_models = ["Resnet18", "Resnet34", "Mobilenet", "Squeezenet", "Inception"]

def load_models(model_info):
    print("Loading Model...")
    model_name = model_info['name']
    model_pretrained = model_info['pretrained']
    if model_name in supported_models:
        if model_name == "Resnet18":
            # model = models.resnet18(pretrained=model_pretrained)
            model = Resnet18(num_classes=model_info['num_classes'])
        elif model_name == "Resnet50":
            model = models.resnet34(pretrained=model_pretrained)
        elif model_name == "Mobilenet":
            model = models.mobilenet_v2(pretrained=model_pretrained)
        elif model_name == "Squeezenet":
            model = models.squeezenet1_1(pretrained=model_pretrained)
        elif model_name == "Inception":
            model = models.inception_v3(pretrained=model_pretrained)
        else:
            print("Unknown model... defaulting to resnet18")
            model = models.resnet18(pretrained=model_pretrained)
    return

# TODO add in class versions of this because based on task need different outputs