import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tfrecord.torch.dataset import TFRecordDataset
import tensorflow as tf
import math
import numpy as np
from math import pi
from scipy.special import logsumexp
from .gmm import *


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    
    
class Encoder(nn.Module):
    def __init__(self,d_model, nhead, dff, nlayers, size_embedding, dropout = 0):
        super(Encoder, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.dense1 = nn.Linear(3, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dff, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.dense2 = nn.Linear(d_model, size_embedding)
        #self.init_weights()
    
    
    def get_last_time_step(self, tensor, stroke_lengths):
        
        embeddingd_lt = []
        
        for pos_embedding in range(tensor.shape[0]):
            embedding = tensor[pos_embedding, stroke_lengths[pos_embedding]-1,:]
            embeddingd_lt.append(embedding)
        
        embeddingd_lt = torch.vstack(embeddingd_lt) 
        
        return embeddingd_lt
    
    
    def forward(self, src, stroke_lengths, src_mask):
        #src = self.pos_encoder(src)
        output = self.dense1(src)
        output = self.pos_encoder(output)
        output = self.transformer_encoder(output, src_mask)
     
        output = self.get_last_time_step(output, stroke_lengths)
        
        output = self.dense2(output)
        
        return output
    
    
    
class Decoder(nn.Module):
    def __init__(self, size_embedding):
        self.dense1 = nn.Linear(in_features=size_embedding, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=512)
        self.dense4 = nn.Linear(in_features=512, out_features=512)
    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        
        return x