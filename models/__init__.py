import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tfrecord.torch.dataset import TFRecordDataset
import tensorflow as tf
import math
import numpy as np


class Encoder(nn.Module):
    def __init__(self,d_model, nhead, dff, nlayers, size_embedding, dropout = 0):
        super(Encoder, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.dense1 = nn.Linear(3, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dff, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.dense2 = nn.Linear(d_model, size_embedding)
    
    
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
        output = self.transformer_encoder(output, src_mask)
     
        output = self.get_last_time_step(output, stroke_lengths)
        
        output = self.dense2(output)
        
        return output
