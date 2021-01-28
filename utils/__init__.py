from .relational import *
from .encoder import *
from .visualization import *
import argparse
import json
from data.loaders import *
from torch.utils.data import DataLoader
import wandb
import torch
import numpy as np
import random
import os

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w','--wandb', default=False, action='store_true',
    help="use weights and biases")
    ap.add_argument('-nw  ','--no-wandb', dest='wandb', action='store_false',
    help="not use weights and biases")

    args = ap.parse_args()

    return args


def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file,'r') as json_file:
            return json.load(json_file)
    else:
        return config_file


def configure_model(config_file, use_wandb=False):

    config_file = parse_configuration(config_file)

    if use_wandb:
        config = wandb.config                                    
    else:
        config = type("configuration", (object,), {})

    #general config
    config.use_gpu = config_file["general_config"]["use_gpu"]   
    config.root_path = config_file["general_config"]["root_path"]
    config.save_path = config_file["general_config"]["save_path"]
    config.dataset_path = config_file["general_config"]["dataset_path"]
    config.num_backups = config_file["general_config"]["num_backups"]
    config.model_path = config_file["general_config"]["model_path"]
    config.save_weights = config_file["general_config"]["save_weights"]

    config.ae_model_type = config_file["ae_model_type"]

    if config_file["ae_model_type"] == "transformer":
        #encoder config
        config.enc_d_model = config_file['enc_hparams']["transformer"]["enc_d_model"]
        config.enc_nhead = config_file['enc_hparams']["transformer"]["enc_nhead"]
        config.enc_dff = config_file['enc_hparams']["transformer"]["enc_dff"]
        config.enc_n_layers = config_file['enc_hparams']["transformer"]["enc_n_layers"]
        config.enc_dropout = config_file['enc_hparams']["transformer"]["enc_dropout"]
        #decoder config
        config.dec_gmm_num_components = config_file["dec_hparams"]["transformer"]["dec_gmm_num_components"]
        config.dec_layer_features = config_file["dec_hparams"]["transformer"]["dec_layer_features"]
    elif config_file["ae_model_type"] == "rnn":
        #encoder config
        config.enc_hsize = config_file["enc_hparams"]["rnn"]["enc_hsize"],
        config.enc_n_layers = config_file["enc_hparams"]["rnn"]["enc_n_layers"],
        config.enc_dropout = config_file["enc_hparams"]["rnn"]["enc_dropout"]
        #decoder config
        config.dec_hsize = config_file["dec_hparams"]["rnn"]["dec_hsize"]
        config.dec_n_layers = config_file["dec_hparams"]["rnn"]["dec_n_layers"]
        config.dec_dim_layer = config_file["dec_hparams"]["rnn"]["dec_dim_layer"]
        config.dec_dropout = config_file["dec_hparams"]["rnn"]["dec_dropout"]
        config.dec_gmm_num_components = config_file["dec_hparams"]["rnn"]["dec_gmm_num_components"]

    else:
        raise ValueError("specified ae_model_type does not exist, please change config in config.json 'ae_model_type'")
    #relational config
    config.rel_d_model = config_file["rel_hparams"]["rel_d_model"]
    config.rel_nhead = config_file["rel_hparams"]["rel_nhead"]
    config.rel_dff = config_file["rel_hparams"]["rel_dff"]
    config.rel_n_layers = config_file["rel_hparams"]["rel_n_layers"]
    config.rel_dropout = config_file["rel_hparams"]["rel_dropout"]
    config.rel_gmm_num_components = config_file["rel_hparams"]["rel_gmm_num_components"]

    #cose model config
    config.size_embedding = config_file["cose_model_params"]["size_embedding"]
    config.num_predictive_inputs = config_file["cose_model_params"]["num_predictive_inputs"]
    config.end_positions = config_file["cose_model_params"]["end_positions"]
    #training params config
    config.input_type = config_file["training_params"]["input_type"]
    config.replace_padding = config_file["training_params"]["replace_padding"]
    config.stop_predictive_grad = config_file["training_params"]["stop_predictive_grad"]
    config.num_epochs = config_file["training_params"]["num_epochs"]
    config.lr_ae = config_file["training_params"]["lr_ae"]
    config.lr_pos_pred = config_file["training_params"]["lr_pos_pred"]
    config.lr_emb_pred = config_file["training_params"]["lr_emb_pred"]
    
    ##TODO completar configuracion
    

    return config


def parse_inputs(batch_input, device):
    enc_inputs = batch_input['encoder_inputs'].squeeze(dim=0).to(device)
    t_inputs = batch_input['t_input'].squeeze(dim=0).to(device)
    stroke_len_inputs = batch_input['seq_len'].squeeze(dim=0).to(device)
    inputs_start_coord = batch_input['start_coord'].squeeze(dim = 0).to(device)
    inputs_end_coord = batch_input['end_coord'].squeeze(dim = 0).to(device)
    num_strokes_x_diagram_tensor = batch_input['num_strokes'].squeeze(dim = 0).to(device)
    return enc_inputs, t_inputs, stroke_len_inputs, inputs_start_coord, inputs_end_coord, num_strokes_x_diagram_tensor

def parse_targets(batch_target, device):
    t_target_ink = batch_target['t_target_ink'].squeeze(dim=0)[:,:2].to(device)
    return t_target_ink

def get_batch_iterator(path):
    batchdata = BatchCoSELoader(path = path,
                        filenames={"inputs_file" : "inputs_list_based_x16.pkl",
                                    "targets_file": "target_list_based_x16.pkl"}
                    )

    train_loader = DataLoader(dataset =batchdata,
                    batch_size = 1, #data is already in batch mode, batch_size = 1 means iterating every .get_next() returns a new batch
                    )
    return train_loader

def get_stats(stats_path='/data/jcabrera/didi_wo_text/'):
    stats_json = 'didi_wo_text-stats-origin_abs_pos.json'
    with open(os.path.join(stats_path, stats_json)) as json_file:
        stats = json.load(json_file)
    return stats

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
