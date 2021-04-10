import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.loaders import *
from models import *
import json
import os
from utils import set_seed, configure_model, parse_arguments
import wandb

PROJECT_WANDB = "CoSE_Pytorch"

def train(config_file, use_wandb, run_name, run_notes):
    config = configure_model(config_file, use_wandb)

    if use_wandb:
        wandb.init(project=PROJECT_WANDB, config=config, name=run_name, notes=run_notes)
        config = wandb.config
        wandb.watch_called = False

    cose = CoSEModel(config, use_wandb)
    cose.fit()

if __name__=='__main__':
    args = parse_arguments()
    use_wandb = args.wandb
    run_name = args.run_name
    run_notes = args.run_notes

    train('config.json', use_wandb, run_name, run_notes)