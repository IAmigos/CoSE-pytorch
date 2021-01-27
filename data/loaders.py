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

        for input_index in len(self.inputs):
            for input_key, input_value in self.inputs[input_index].items():
                self.inputs[input_index][input_key] = torch.from_numpy(input_value)

        for target_index in len(self.targets):
            for target_key, target_value in self.targets[target_index].items():
                self.targets[target_index][target_key] = torch.from_numpy(target_value)

        self.len = len(self.inputs)
        assert (len(self.inputs) == len(self.targets))
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    
    def __len__(self):
        return self.len
