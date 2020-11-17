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

    def check_parameters(self):
        # First check, number of epochs is greater than 0
        if self.num_epoch > 0:
            self.valid_args = True
        return

    def train_network(self):
        self.check_parameters()
        if not self.valid_args:
            print("Havent set everything in JSON")
            return
        print("Starting training...")
        for i in range(self.num_epoch):
            # Get iterators
            self.train_dataset = self.dataset_trainer.get_train_iter()
            self.test_dataset = self.dataset_trainer.get_test_iter()
            # Enumerate over dataset
            # for i, data in enumerate(self.train_dataset, 0):
            #     inputs = data[0]
            #     targets = data[1]
            #     print("inputs: ", inputs.shape)
            #     print("targets: ", targets.shape)

        return