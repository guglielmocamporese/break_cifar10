##################################################
# Imports
##################################################

import argparse
import json


def get_args(stdin, verbose=False):
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(stdin)

    # Global params
    parser.add_argument('--seed', type=int, default=35771, help='The random seed.')
    parser.add_argument('--logger', type=str, default='wandb', help='The logger to use for the experiments.')
    parser.add_argument('--mode', type=str, default='train', help='The mode of the program, can "train" or "validate"')
    parser.add_argument('--num_gpus', type=int, default=1, help='The number of GPUs.')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs for the training.')

    # Datasets params
    parser.add_argument('--dataset', type=str, default='voc', help='Name of the dataset.')
    parser.add_argument('--data_base_path', type=str, default='./datasets/data', help='The data base path.')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size.')
    parser.add_argument('--num_workers', type=int, default=8, help='The number of workers for the dataloader.')
    parser.add_argument('--validation_ratio', type=float, default=0.2, help='The validation ratio.')

    # Optimizer params
    parser.add_argument('--optimizer', type=str, default='sgd', help='The kind of optimizer.')
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate.')
    parser.add_argument('--metric_monitor', type=str, default='val_acc', help='The metric used for early stopping.')

    # Model params
    parser.add_argument('--model_checkpoint', type=str, default='', help='The model checkpoint path (*.ckpt).')
    parser.add_argument('--backbone', type=str, default='vit', help='The model bacbkone.')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing.')

    # Parse args
    args = parser.parse_args()

    if verbose:
        print('Input Args: ' + json.dumps(vars(args), indent=4))
    return args
