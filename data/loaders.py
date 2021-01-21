from torch.utils.data import Dataset
import torch
import pickle
import os

class BatchCoSELoader(Dataset):
    '''
    Returns Batches of data, every new item is a new Batch of data
    Args:
    path: str, directory data path
    filenames: dict, dictionary {'inputs_file': str, 'targets_file': str}
    '''
    def __init__(self, path: str, filenames: dict):
        self.inputs = pickle.load(open(os.path.join(path, filenames["inputs_file"]), 'rb'))
        self.targets = pickle.load(open(os.path.join(path, filenames["targets_file"]), 'rb'))
        self.len = len(self.inputs)
        assert (len(self.inputs) == len(self.targets))
    
    def __getitem__(self, index):
        for input_key, input_value in self.inputs[index].items():
            self.inputs[index][input_key] = torch.from_numpy(input_value)
        for target_key, target_value in self.targets[index].items():
            self.targets[index][target_key] = torch.from_numpy(target_value)
        return self.inputs[index], self.targets[index]
    
    def __len__(self):
        return self.len
