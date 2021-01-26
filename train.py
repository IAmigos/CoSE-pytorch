import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.loaders import *
from models import *
import json
from models import *

batchdata = BatchCoSELoader(path = "D:\Projects\RC-PWC2020\data",
                          filenames={"inputs_file" : "inputs_list_based.pkl",
                                     "targets_file": "target_list_based.pkl"}
                        )

train_loader = DataLoader(dataset =batchdata,
                batch_size = 1, #data is already in batch mode, batch_size = 1 means iterating every .get_next() returns a new batch
                )

config_ = json.load(open("config.json", 'rb'))

cose_model  = CoSEModel(**config_)

#TODO
#Crear dummies con los shapes que entran a cada modelo
#forward del embedding (Joel)
#procesamiento previo al relacional (agregar un config) (Alex)
#forward del relacional (Stev)
#forward del decoder (Alex)
#transform2image (Daniel)
config = {"input_type": "hybrid",
                 "num_predictive_inputs": 32,
                 "replace_padding": True,
                 "end_positions": False,
                 "stop_predictive_grad": False}

def train(config_file, use_wandb=True):

    cose = CoSEModel(config_file, use_wandb)
    cose.fit()

if __name__=='__main__':
    args = parse_arguments()
    use_wandb = args.wandb

    train('config.json', use_wandb=True)
