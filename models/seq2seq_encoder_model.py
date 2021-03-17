import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from math import pi
from .gmm import *
from utils import *
import os
from torch.autograd import Variable



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, encoder_dim, num_layers, device, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.encoder_dim = encoder_dim
        self.device = device
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size*2, encoder_dim)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.lstm.weight)

    def get_last_time_step(self, tensor, stroke_lengths):
        
        embeddingd_lt = []
        
        for pos_embedding in range(tensor.shape[0]):
            embedding = tensor[pos_embedding, stroke_lengths[pos_embedding]-1,:]
            embeddingd_lt.append(embedding)
        
        embeddingd_lt = torch.vstack(embeddingd_lt) 
        
        return embeddingd_lt        

        
    def forward(self, input, stroke_lengths, src_mask):
        
        h0 = Variable(torch.randn(self.num_layers*2, input.size(1), self.hidden_size).to(self.device))
        c0 = Variable(torch.randn(self.num_layers*2, input.size(1), self.hidden_size).to(self.device))
        #h0, c0 = self.init_hidden(input.size(0))
        
        encoded_input, hidden = self.lstm(input, (h0, c0))
        
        encoded_input = encoded_input.contiguous().view(-1, self.hidden_size)
        
        encoded_input = encoded_input.reshape((input.size(0), input.size(1), self.hidden_size*2))
        encoded_input = encoded_input.permute(1,0,2)
        
        encoded_input = self.get_last_time_step(encoded_input, stroke_lengths)
        
        encoded_input = self.dropout(encoded_input)
        encoded_input = F.relu(self.fc1(encoded_input))
        
        return encoded_input
    


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, t_input_size, output_size, encoder_dim, num_layers, device, dim_layer, num_components, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.encoder_dim = encoder_dim
        self.device = device
        self.t_input_size = t_input_size
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(encoder_dim, hidden_size)        
        self.lstm2 = nn.LSTM(hidden_size, output_size, num_layers, dropout=dropout,
                             batch_first=True)
        self.fc2 = nn.Linear(t_input_size*output_size, dim_layer)
        self.gmm = OutputModelGMMDense(input_size= dim_layer, out_units= 2, num_components=num_components)
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.lstm2.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

        
    def forward(self, encoded_input):
        h0 = Variable(torch.randn(self.num_layers, encoded_input.size(0), self.output_size).to(self.device))
        c0 = Variable(torch.randn(self.num_layers, encoded_input.size(0), self.output_size).to(self.device))
        
        encoded_input = encoded_input.unsqueeze(dim = 1).repeat(1, self.t_input_size, 1)
        dec = F.relu(self.fc1(encoded_input))
        dec = self.dropout(dec)
        
        lstm_enc, hidden = self.lstm2(dec, (h0, c0))
                
        lstm_enc = lstm_enc.reshape((encoded_input.size(0),-1))
        
        x = self.fc2(lstm_enc)
        
        out_mu, out_sigma, out_pi = self.gmm(x)
        strokes_out = self.gmm.draw_sample(out_mu, out_sigma, out_pi, greedy=True)
        
        return strokes_out, out_mu, out_sigma, out_pi


