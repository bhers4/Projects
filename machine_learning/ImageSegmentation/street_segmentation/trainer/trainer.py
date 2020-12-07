import torch
import numpy as np
import enum
import datetime
import os
import json


# Main Trainer
class Trainer(object):

    def __init__(self):
        '''
            Set Default Parameters
        '''
        # Number of Epochs to Train
        self.num_epoch = 0
        self.curr_epoch = 0
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
        self.train_accs = []
        # Iter losses
        self.iter_losses = []
        # Active flag
        self.active = False
        # Save Config
        self.config = None
        # Run id
        self.run_id = None
        # Saved For some +1 and +5 epochs logic
        self.saved = False
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

    def train_network(self, train_info):
        self.check_parameters()
        if not self.valid_args:
            print("Havent set everything in JSON")
            return
        # Do something about potentially changed parameters here from UI
        project_name, dataset_name, batch_size, shuffle, optim, lr = train_info
        if self.config['name'] != project_name:
            print("Changed Project Name to ", project_name)
        if self.dataset_trainer.dataset_name != dataset_name:
            print("Dataset Name changed: ", dataset_name)
        if self.dataset_trainer.batch_size != batch_size:
            print("Batch size changed: ", batch_size)
        # Optim
        if self.config['optim'] != lr:
            print("Learning rate changed: ", lr)
        # Run ID
        self.run_id = "".join(self.config['name'].split(" "))
        self.project_name = str(self.run_id)
        curr_date = datetime.datetime.now()
        self.run_id += "_"+str(curr_date.year)+str(curr_date.month)+str(curr_date.day)+"_"+str(curr_date.hour)+\
                       "_"+str(curr_date.minute)+"_"+str(curr_date.second)
        print("Self run id: ", self.run_id)
        # Set Active Flag
        self.active = True
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print("Starting training...")
        for epoch in range(self.num_epoch):
            # Get iterators
            self.curr_epoch = epoch
            self.train_dataset = self.dataset_trainer.get_train_iter()
            self.test_dataset = self.dataset_trainer.get_test_iter()
            self.iter_losses = []
            self.train_correct = 0
            self.train_total = 0

            # Enumerate over dataset
            for i, data in enumerate(self.train_dataset, 0):
                self.optim.zero_grad()
                inputs = data[0].to(self.device)
                targets = data[1].to(self.device)
                # Get output
                output = self.model(inputs)
                prediction = torch.max(output, dim=1)[1]
                correct = (prediction == targets).sum()
                self.train_correct += correct.item()
                self.train_total += prediction.shape[0]
                local_loss = self.loss(output, targets)
                local_loss.backward()
                self.optim.step()
                self.iter_losses.append(local_loss.item())
            self.epoch_losses.append(np.mean(self.iter_losses))
            test_loss = self.test_network(self.test_dataset)
            self.epoch_test_losses.append(test_loss)
            test_acc = self.test_correct / self.test_total
            train_acc = self.train_correct / self.train_total
            self.test_accs.append(test_acc)
            print("Epoch: %d, Loss: %.2f, Test Loss: %.2f, Train Acc: %.1f Test Acc: %.1f" % (epoch,
                                                                                              self.epoch_losses[-1],
                                                                                              test_loss,
                                                                                              train_acc*100,
                                                                                              test_acc*100))
            self.train_accs.append(self.train_correct/self.train_total)
        self.save()
        self.active = False
        return

    def save(self):
        # Make sure history dir exists
        history_dir = os.path.join(os.getcwd(), "history")
        if not os.path.isdir(history_dir):
            os.mkdir(history_dir)
        run_dir = os.path.join(history_dir, self.run_id)
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)
        json_file = os.path.join(run_dir, 'run.json')
        with open(json_file, "w+") as json_data:
            run_data = {}
            run_data['model_info'] = self.config['models']
            run_data['train_loss'] = self.epoch_losses
            run_data['test_loss'] = self.epoch_test_losses
            run_data['train_accs'] = self.train_accs
            run_data['test_accs'] = self.test_accs
            json.dump(run_data, json_data)
        # Save model
        model_file = os.path.join(run_dir, 'run.pt')
        torch.save(self.model.state_dict(), model_file)
        return