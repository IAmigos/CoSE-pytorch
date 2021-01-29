import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.loaders import *
from models import *
import json
import os
from utils import set_seed


def train(config_file, use_wandb=True):
    set_seed(0)
    cose = CoSEModel(config_file, use_wandb)
    path_to_model = f"/home/{user}/CoSE-pytorch/pesos_descargados"
    cose.encoder.load_state_dict(torch.load(os.path.join(os.getcwd(), path_to_model,"encoder.pth"), map_location=device))
    cose.decoder.load_state_dict(torch.load(os.path.join(os.getcwd(),path_to_model ,"decoder.pth"), map_location=device))
    cose.position_predictive_model.load_state_dict(torch.load(os.path.join(os.getcwd(),path_to_model,"pos_pred.pth"), map_location=device))
    cose.embedding_predictive_model.load_state_dict(torch.load(os.path.join(os.getcwd(),path_to_model,"emb_pred.pth"), map_location=device))
    cose.fit()

if __name__=='__main__':
    args = parse_arguments()
    use_wandb = args.wandb

    train('config.json', use_wandb)
