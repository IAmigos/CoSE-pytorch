from .relational import *
from .utils import *
from .encoder import *
from .visualization import *
from .eval import *
import argparse
import json
from data.loaders import *
from torch.utils.data import DataLoader
import wandb
import torch
import numpy as np
import random
import os
from torch.autograd import Variable


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--wandb', default=False, action='store_true',
                    help="use weights and biases")
    ap.add_argument('-nw  ', '--no-wandb', dest='wandb', action='store_false',
                    help="not use weights and biases")
    ap.add_argument('-n', '--run_name', required=False, type=str, default=None,
                    help="name of the execution to save in wandb")
    ap.add_argument('-nt', '--run_notes', required=False, type=str, default=None,
                    help="notes of the execution to save in wandb")

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

    config = dict(
        use_gpu=config_file["general_config"]["use_gpu"],
        root_path=config_file["general_config"]["root_path"],
        save_path=config_file["general_config"]["save_path"],
        train_dataset_path=config_file["general_config"]["train_dataset_path"],
        validation_dataset_path=config_file["general_config"]["validation_dataset_path"],
        test_dataset_path=config_file["general_config"]["test_dataset_path"],
        num_backups=config_file["general_config"]["num_backups"],
        model_path=config_file["general_config"]["model_path"],
        save_weights=config_file["general_config"]["save_weights"],
        stats_path=config_file["general_config"]["stats_path"],
        diagrams_img_path=config_file["general_config"]["diagrams_img_path"],

        #autoencoder model type
        ae_model_type=config_file["ae_model_type"],

        #relational config
        rel_d_model=config_file["rel_hparams"]["rel_d_model"],
        rel_nhead=config_file["rel_hparams"]["rel_nhead"],
        rel_dff=config_file["rel_hparams"]["rel_dff"],
        rel_n_layers=config_file["rel_hparams"]["rel_n_layers"],
        rel_dropout=config_file["rel_hparams"]["rel_dropout"],
        rel_gmm_num_components=config_file["rel_hparams"]["rel_gmm_num_components"],

        #cose model config
        size_embedding=config_file["cose_model_params"]["size_embedding"],
        num_predictive_inputs=config_file["cose_model_params"]["num_predictive_inputs"],
        end_positions=config_file["cose_model_params"]["end_positions"],
        
        #training params config
        input_type=config_file["training_params"]["input_type"],
        replace_padding=config_file["training_params"]["replace_padding"],
        stop_predictive_grad=config_file["training_params"]["stop_predictive_grad"],
        num_epochs=config_file["training_params"]["num_epochs"],
        lr_ae=config_file["training_params"]["lr_ae"],
        lr_pos_pred=config_file["training_params"]["lr_pos_pred"],
        lr_emb_pred=config_file["training_params"]["lr_emb_pred"]
    )

    #transformer
    if config_file["ae_model_type"] == "transformer":
        #encoder config
        config["enc_d_model"] = config_file['enc_hparams']["transformer"]["enc_d_model"]
        config["enc_nhead"] = config_file['enc_hparams']["transformer"]["enc_nhead"]
        config["enc_dff"] = config_file['enc_hparams']["transformer"]["enc_dff"]
        config["enc_n_layers"] = config_file['enc_hparams']["transformer"]["enc_n_layers"]
        config["enc_dropout"] = config_file['enc_hparams']["transformer"]["enc_dropout"]
        #decoder config
        config["dec_gmm_num_components"] = config_file["dec_hparams"]["transformer"]["dec_gmm_num_components"]
        config["dec_layer_features"] = config_file["dec_hparams"]["transformer"]["dec_layer_features"]
    
    #rnn    
    elif config_file["ae_model_type"] == "rnn":
        #encoder config
        config["enc_hsize"] = config_file["enc_hparams"]["rnn"]["enc_hsize"]
        config["enc_n_layers"] = config_file["enc_hparams"]["rnn"]["enc_n_layers"]
        config["enc_dropout"] = config_file["enc_hparams"]["rnn"]["enc_dropout"]
        #decoder config
        config["dec_hsize"] = config_file["dec_hparams"]["rnn"]["dec_hsize"]
        config["dec_n_layers"] = config_file["dec_hparams"]["rnn"]["dec_n_layers"]
        config["dec_dim_layer"] = config_file["dec_hparams"]["rnn"]["dec_dim_layer"]
        config["dec_dropout"] = config_file["dec_hparams"]["rnn"]["dec_dropout"]
        config["dec_gmm_num_components"] = config_file["dec_hparams"]["rnn"]["dec_gmm_num_components"]
    
    else:
        raise ValueError("specified ae_model_type does not exist, please change config in config.json 'ae_model_type'")


    if not use_wandb:
        config = type("configuration", (object,), config)

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

def get_batch_iterator(path, test = False, batch_size = 64):

    if not test:
        batchdata = BatchCoSELoader(path = path,
                            filenames={"inputs_file" : f"inputs_list_based_x{batch_size}.pkl",
                                        "targets_file": f"target_list_based_x{batch_size}.pkl"}
                        )
    else:
        batchdata = BatchCoSELoader(path = path,
                            filenames={"inputs_file" : "inputs_list_based.pkl",
                                        "targets_file": "target_list_based.pkl"}
                        )

    loader = DataLoader(dataset =batchdata,
                    batch_size = 1, #data is already in batch mode, batch_size = 1 means iterating every .get_next() returns a new batch
                    )
    return loader

def get_stats(stats_path='/data/jcabrera/didi_wo_text/didi_wo_text-stats-origin_abs_pos.json'):
    
    with open(stats_path) as json_file:
        stats = json.load(json_file)

    mean_channel = stats['mean_channel'][:2]
    std_channel = np.sqrt(stats['var_channel'][:2])

    return mean_channel, std_channel

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

    
def mask_target(inputs, device, n_heads=4):
    """
    inputs: shape (batch_size, sequence_length, num_features)
    result_final: shape (batch_size, sequence_length, sequence_length)
    1: appears
    0: doesnt appear
    """
    mask = inputs[:,:,0].detach().clone()
    mask[mask!=0] = 1
    mask[mask!=1] = 0
    mask = (mask==1)
    mask = mask.unsqueeze(dim=1).repeat(1,mask.shape[1],1).reshape(mask.shape[0], mask.shape[-1],-1)
    
    nopeak_mask = np.triu(np.ones((inputs.shape[0], inputs.shape[1], inputs.shape[1])),k=1).astype('uint8')
    nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)

    nopeak_mask = nopeak_mask.to(device)
    
    result_final = (nopeak_mask & mask).gt(0).to(torch.float32)

    result_final = result_final.masked_fill(result_final == 0, float('-inf'))
    result_final = result_final.masked_fill(result_final == 1, float(0))

    return result_final.repeat_interleave(n_heads, dim = 0)