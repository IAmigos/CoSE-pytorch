import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.loaders import *
from models import *
import json
from utils import set_seed


def train(config_file, use_wandb=True):
    set_seed(0)
    cose = CoSEModel(config_file, use_wandb)
    cose.fit()

if __name__=='__main__':
    args = parse_arguments()
    use_wandb = args.wandb

    train('config.json', use_wandb)
