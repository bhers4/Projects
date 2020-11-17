import argparse
import json
from datasets.datasets import parse_dataset
from trainer.trainer import Trainer
from models.models import load_models

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
    model = load_models(configs_file['model'])
    # Trainer
    nn_trainer = Trainer()
    nn_trainer.set_epochs(run_args['num_epochs'])
    nn_trainer.set_trainer_dataset(dataset)
    nn_trainer.train_network()
'''
    IDEA: calculate approximate data size, along with RAM target
'''

