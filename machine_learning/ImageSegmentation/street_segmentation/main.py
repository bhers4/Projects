import argparse
import json
from datasets.datasets import parse_dataset
from trainer.trainer import Trainer
from models.models import load_models
from loss.loss import load_loss
from optim.optim import load_optim
from matplotlib import pyplot as plt

help='deep learning framework'

def add_arguments(parser):
    # Path to JSON file
    parser.add_argument('--configs', help='Path to json file')
    # Number of workers
    parser.add_argument('--j', type=int, default=4, help='Number of Threads/workers')
    # Once basic UI is done do "start server"
    return

if __name__ == "__main__":
    # Arg parser
    parser = argparse.ArgumentParser(description=help)
    add_arguments(parser)
    args = parser.parse_args()
    configs_file = args.configs
    configs_file = json.load(open(configs_file, "r"))
    threads = args.j
    print("Threads: ", threads)
    print("Configs: ", configs_file)
    # Get Dataset Parameters
    dataset = parse_dataset(configs_file['dataset'], threads)
    run_args = configs_file['run']
    # Model
    model = load_models(configs_file['models'])
    # Trainer
    nn_trainer = Trainer()
    # Set number of epochs
    nn_trainer.set_epochs(run_args['num_epochs'])
    # Set Datasets
    nn_trainer.set_trainer_dataset(dataset)
    # Set Model
    nn_trainer.set_model(model)
    # Set Loss
    loss = load_loss(configs_file["loss"])
    nn_trainer.set_loss(loss)
    # Optim
    optim = load_optim(configs_file['optim'], model)
    nn_trainer.set_optim(optim)
    # Train!
    nn_trainer.train_network()
    plt.plot(nn_trainer.epoch_losses, c='b', label='train')
    plt.plot(nn_trainer.epoch_test_losses, c='r', label='test')
    plt.plot(nn_trainer.test_accs, c='m', label='Test Acc')
    plt.title("Epoch losses")
    plt.legend()
    plt.show()
'''
    IDEA: calculate approximate data size, along with RAM target
'''

