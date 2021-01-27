import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from math import pi
from .gmm import *
from utils import *
import os


class Decoder(nn.Module):
    def __init__(self, size_embedding, num_components, out_units, layer_features: list):
        super(Decoder, self).__init__()
        self.dense_layers= nn.ModuleList(
                [nn.Linear(
                    in_features=size_embedding + 1, #+1 por los t_inputs 
                    out_features=layer_features[0])] +\
                [nn.Linear(
                    in_features=layer_features[i], 
                    out_features= layer_features[i+1]) for i in range(len(layer_features)- 1)]
        )
        self.gmm = OutputModelGMMDense(input_size= layer_features[-1], out_units= 2, num_components= num_components) # 2 units for (x,y)

    def forward(self, x):
        for layer in self.dense_layers:
            x = F.relu(layer(x))
        out_mu, out_sigma, out_pi = self.gmm(x)
        strokes_out = self.gmm.draw_sample(out_mu, out_sigma, out_pi, greedy=True)
        return strokes_out, out_mu, out_sigma, out_pi