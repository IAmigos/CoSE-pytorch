import json
import torch
import torch.nn as nn
import torch.nn.functional as F
#from tfrecord.torch.dataset import TFRecordDataset
#import tensorflow as tf
import math
import numpy as np
from math import pi
#from scipy.special import logsumexp
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
    def __init__(self, size_embedding, num_components, out_units, layer_features: list):
        super(Decoder, self).__init__()
        self.dense_layers= nn.ModuleList(
                [nn.Linear(
                    in_features=size_embedding, 
                    out_features=layer_features[0])] +\
                [nn.Linear(
                    in_features=layer_features[i], 
                    out_features= layer_features[i+1]) for i in range(len(layer_features)- 1)]
        )
        self.gmm = OutputModelGMMDense(input_size= layer_features[-1], out_units= 2, num_components= num_components) # 2 units for (x,y)

    def forward(self, x):

        for layer in self.dense_layers:
            x = F.relu(layer(x))
        output_gmm = self.gmm(x)
        strokes_out = self.gmm.draw_sample(outputs=output_gmm, greedy=True)
        return x
    
    
    
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

    def forward(self, src, num_strokes, target_cond, src_mask):
        #src = self.pos_encoder(src)
        output = self.dense1(src)
        #print(output[:,None,:].shape)
        output = self.transformer_decoder(output, output)
        output = self.dense2(output) #512,256 [256, feedforward: (512, 256)]
        output = self.dense3(output)
        output = self.get_last_stroke(output, num_strokes)
        
        output_gmm = self.gmm(output)

        out = self.gmm.draw_sample(outputs=output_gmm, greedy=True)
        
        return out

class CoSEModel(nn.Module):
    def __init__(self,
                enc_d_model = 64,
                enc_nhead = 4,
                enc_dff = 128,
                enc_n_layers = 6,
                enc_dropout = 0,
                rel_d_model = 64,
                rel_nhead = 4,
                rel_dff = 256,
                rel_n_layers = 6,
                rel_dropout = 0,
                rel_gmm_num_components = 20,
                dec_gmm_num_components = 20,
                layer_features = [512,512,512,512],
                size_embedding = 8,
        ):
        super(CoSEModel, self).__init__()
        self.encoder = Encoder(d_model=enc_d_model, nhead=enc_nhead, dff=enc_dff
                                , nlayers=enc_n_layers, size_embedding=size_embedding, dropout=enc_dropout)
        self.decoder = Decoder(size_embedding=size_embedding, num_components=dec_gmm_num_components
                                , out_units=2, layer_features=layer_features)
        self.position_predictive_model = TransformerGMM(d_model = rel_d_model,nhead=rel_nhead,
                                                        dff=rel_dff, nlayers=rel_n_layers,
                                                        input_size= size_embedding + 2, num_components= rel_gmm_num_components,
                                                        out_units = 2, dropout = rel_dropout
                                                        )
        self.embedding_predictive_model = TransformerGMM(d_model = rel_d_model, nhead = rel_nhead,
                                                         dff = rel_dff, nlayers = rel_n_layers,
                                                         input_size= size_embedding + 4, num_components = rel_gmm_num_components,
                                                         out_units = size_embedding, dropout = rel_dropout
                                                         )

    def tranform2image(self, stroke):
        pass

    def forward(self, diagrama):
        
        out = self.encoder(diagrama, stroke_lengths, src_mask)
        out = self.position_predictive_model(out)
        out = self.embedding_predictive_model(out)
        out = self.decoder(out)
        stroke_image = self.tranform2image(out)

        return stroke_image        
        
    def fit(self, n_epochs:int, trainloader):
        raise NotImplementedError()
        
    def load_weights(self, path_weights):
        pass


