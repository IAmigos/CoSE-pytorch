from .relational import *
from .encoder import *


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

    ##completar configuracion   

def parse_inputs(batch_input):
    enc_inputs = batch_input['encoder_inputs'].squeeze(dim=0)
    t_inputs = batch_input['t_input'].squeeze(dim=0)
    stroke_len_inputs = batch_input['seq_len'].squeeze(dim=0)
    inputs_start_coord = batch_input['start_coord'].squeeze(dim = 0)
    inputs_end_coord = batch_input['end_coord'].squeeze(dim = 0)
    num_strokes_x_diagram_tensor = batch_input['num_strokes'].squeeze(dim = 0)
    return enc_inputs, t_inputs, stroke_len_inputs, inputs_start_coord, inputs_end_coord, num_strokes_x_diagram_tensor

def get_batch_iterator(path):
    batchdata = BatchCoSELoader(path = path,
                        filenames={"inputs_file" : "inputs_list_based.pkl",
                                    "targets_file": "target_list_based.pkl"}
                    )

    train_loader = DataLoader(dataset =batchdata,
                    batch_size = 1, #data is already in batch mode, batch_size = 1 means iterating every .get_next() returns a new batch
                    )
    return train_loader