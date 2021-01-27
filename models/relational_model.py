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
    def __init__(self,d_model, nhead, dff, nlayers, input_size, num_components, out_units, dropout = 0):
        super(TransformerGMM, self).__init__()
        from torch.nn import TransformerDecoderLayer, TransformerDecoder
        self.dense1 = nn.Linear(input_size, d_model)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dff, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.dense2 = nn.Linear(d_model, dff*2) #[256 256]
        self.dense3 = nn.Linear(dff*2, dff) #[256 - 512 - 256]
        self.gmm = OutputModelGMMDense(input_size=dff, out_units=out_units, num_components=num_components)
        
    def get_last_stroke(self, tensor, num_strokes):
        
        embeddingd_lt = []
        
        for pos_embedding in range(tensor.shape[0]):
            embedding = tensor[pos_embedding, num_strokes[pos_embedding]-1,:]
            embeddingd_lt.append(embedding)
        
        embeddingd_lt = torch.vstack(embeddingd_lt) 
        
        return embeddingd_lt

    def forward(self, src, num_strokes, src_mask):
        #src = self.pos_encoder(src)
        output = self.dense1(src)
        #print(output[:,None,:].shape)
        output = self.transformer_decoder(output, output)
        output = self.dense2(output) #512,256 [256, feedforward: (512, 256)]
        output = self.dense3(output)
        output = self.get_last_stroke(output, num_strokes)
        
        out_mu, out_sigma, out_pi = self.gmm(output)

        #output = self.gmm.draw_sample(out_mu, out_sigma, out_pi, greedy=True)
        
        return out_mu, out_sigma, out_pi

    def draw_sample(self, out_mu, out_sigma, out_pi):
        return self.gmm.draw_sample(out_mu, out_sigma, out_pi, greedy=True)