import torch
import numpy as np
import enum


# Main Trainer
class Trainer(object):

    def __init__(self):
        '''
            Set Default Parameters
            - num_epochs
        '''
        # Number of Epochs to Train
        self.num_epoch = 0

        # Datasets
        self.dataset_trainer = None
        self.train_dataset = None
        self.test_dataset = None

        # Flag to say good to train
        self.valid_args = False
        # Model
        self.model = None
        # Loss Function for training
        self.loss = None
        # Optimizer
        self.optim = None
        # Epoch losses
        self.epoch_losses = []
        self.epoch_test_losses = []
        self.test_accs = []
        # Iter losses
        self.iter_losses = []
        # Active flag
        self.active = False

        return

    def set_epochs(self, num_epochs):
        self.num_epoch = num_epochs
        return

    def get_epochs(self):
        return self.num_epoch

    def set_trainer_dataset(self, dataset):
        self.dataset_trainer = dataset
        return

    # Model Getters/Setters
    def set_model(self, model):
        self.model = model
        return

    def get_model(self):
        return self.model
    # Loss Function Getter/Setter
    def set_loss(self, loss):
        self.loss = loss
        return

    def get_loss(self):
        return self.loss
    # Optimizer Getter/Setter
    def set_optim(self, optim):
        self.optim = optim
        return

    def get_optim(self):
        return self.optim

    def check_parameters(self):
        # First check, number of epochs is greater than 0
        if self.num_epoch > 0:
            self.valid_args = True
        return

    def test_network(self, test_iter):
        if not self.valid_args:
            print("Invalid args")
            return
            # Enumerate over dataset
        self.test_losses = []
        self.test_correct = 0
        self.test_total = 0
        for i, data in enumerate(test_iter, 0):
            self.optim.zero_grad()
            inputs = data[0].to(self.device)
            targets = data[1].to(self.device)
            output = self.model(inputs)
            prediction = torch.max(output, dim=1)[1]
            correct = (prediction == targets).sum()
            self.test_correct += correct.item()
            self.test_total += prediction.shape[0]
            local_loss = self.loss(output, targets)
            local_loss.backward()
            self.optim.step()
            self.test_losses.append(local_loss.item())
        return np.mean(self.test_losses)

    def train_network(self):
        self.check_parameters()
        if not self.valid_args:
            print("Havent set everything in JSON")
            return
        # Set Active Flag
        self.active = True
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print("Starting training...")
        for epoch in range(self.num_epoch):
            # Get iterators
            self.train_dataset = self.dataset_trainer.get_train_iter()
            self.test_dataset = self.dataset_trainer.get_test_iter()
            self.iter_losses = []

            # Enumerate over dataset
            for i, data in enumerate(self.train_dataset, 0):
                self.optim.zero_grad()
                inputs = data[0]
                targets = data[1]
                # Get output
                output = self.model(inputs.to(self.device))
                local_loss = self.loss(output, targets.to(self.device))
                local_loss.backward()
                self.optim.step()
                self.iter_losses.append(local_loss.item())
            self.epoch_losses.append(np.mean(self.iter_losses))
            test_loss = self.test_network(self.test_dataset)
            self.epoch_test_losses.append(test_loss)
            test_acc = self.test_correct / self.test_total
            self.test_accs.append(test_acc)
            print("Epoch: %d, Loss: %.2f, Test Loss: %.2f, Test Acc: %.1f" % (epoch, self.epoch_losses[-1], test_loss,
                                                                              test_acc*100))

        self.active = False
        return