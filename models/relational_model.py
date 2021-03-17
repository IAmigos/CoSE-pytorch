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


class TransformerGMM(nn.Module):
    def __init__(self,d_model, nhead, dff, nlayers, input_size, num_components, out_units, dropout = 0, mid_concat = False):
        super(TransformerGMM, self).__init__()
        from torch.nn import TransformerDecoderLayer, TransformerDecoder
        self.dense1 = nn.Linear(input_size, d_model)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dff, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        if mid_concat:
            self.dense2 = nn.Linear(d_model + 2, dff*2) #[64 256]
        else:
            self.dense2 = nn.Linear(d_model, dff*2)
        self.dense3 = nn.Linear(dff*2, dff) #[256 - 512 - 256]
        self.relu = nn.ReLU()
        self.gmm = OutputModelGMMDense(input_size=dff, out_units=out_units, num_components=num_components)
        
        nn.init.kaiming_normal_(self.dense1.weight)
        #nn.init.kaiming_normal_(self.transformer_decoder.weight)
        nn.init.kaiming_normal_(self.dense2.weight)
        nn.init.kaiming_normal_(self.dense3.weight)

    def get_last_stroke(self, tensor, num_strokes):
        
        embeddingd_lt = []
        
        for pos_embedding in range(tensor.shape[0]):
            embedding = tensor[pos_embedding, num_strokes[pos_embedding]-1,:]
            embeddingd_lt.append(embedding)
        
        embeddingd_lt = torch.vstack(embeddingd_lt) 
        
        return embeddingd_lt

    def forward(self, src, num_strokes, tgt_cond = None, src_mask =  None):
        output = self.dense1(src.permute(1,0,2))
        output = self.transformer_decoder(output, output, tgt_mask  = src_mask, memory_mask = src_mask).permute(1,0,2)
        output = self.get_last_stroke(output, num_strokes)
        if tgt_cond is not None:
            output = torch.cat([output, tgt_cond], dim = 1)
        output = self.dense2(output)
        output = self.dense3(output)
        output = self.relu(output)
        out_mu, out_sigma, out_pi = self.gmm(output)
        
        return out_mu, out_sigma, out_pi

    def draw_sample(self, out_mu, out_sigma, out_pi, greedy = True ):
        return self.gmm.draw_sample(out_mu, out_sigma, out_pi, greedy=greedy)