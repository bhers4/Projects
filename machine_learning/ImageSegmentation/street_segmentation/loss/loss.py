import torch


supported_losses = ['BCELoss', 'CrossEntropyLoss']

def load_loss(loss_info, custom_loss=None):
    loss_name = loss_info['name']
    if loss_name in supported_losses:
        print("Loading supported loss func")
        if loss_name == "BCELoss":
            loss = torch.nn.BCELoss()
        elif loss_name == "CrossEntropyLoss":
            loss = torch.nn.CrossEntropyLoss()
        else:
            print("Unknown Loss... Defaulting to CrossEntropy")
            loss = torch.nn.CrossEntropyLoss()
    else:
        print("Custom Loss")
        loss = custom_loss
    return loss