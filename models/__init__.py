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
from utils import *
import wandb
from tqdm import tqdm
import os
torch.cuda.empty_cache()


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

    def get_last_time_step(self, tensor, stroke_lengths):
        
        embeddingd_lt = []
        
        for pos_embedding in range(tensor.shape[0]):
            embedding = tensor[pos_embedding, stroke_lengths[pos_embedding]-1,:]
            embeddingd_lt.append(embedding)
        
        embeddingd_lt = torch.vstack(embeddingd_lt) 
        
        return embeddingd_lt
    
    def forward(self, src, stroke_lengths, src_mask):

        output = self.dense1(src) #init embedding
        output = self.pos_encoder(output)
        output = self.transformer_encoder(output, src_mask).permute(1,0,2) #positional encoding + encoder
     
        output = self.get_last_time_step(output, stroke_lengths) #fetch last step
        
        output = self.dense2(output) #output deterministc embedding (#TODO configurable)
        
        return output
    
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


class CoSEModel(nn.Module):
    def __init__(self,
            config_file,
            use_wandb=True
        ):
        super(CoSEModel, self).__init__()
    
        self.use_wandb = use_wandb

        if self.use_wandb:
            wandb.init(project="CoSE_Pytorch")
            wandb.watch_called = False

        self.config = configure_model(config_file, self.use_wandb)

        self.device = torch.device("cuda:1" if self.config.use_gpu and torch.cuda.is_available() else "cpu")
        self.encoder, self.decoder, self.position_predictive_model ,self.embedding_predictive_model = self.init_model(self.device, self.config, self.use_wandb)


    def tranform2image(self, stroke):
        pass

    def forward(self, diagrama):
        
        out = self.encoder(diagrama, stroke_lengths, src_mask)
        out = self.position_predictive_model(out)
        out = self.embedding_predictive_model(out)
        out = self.decoder(out)
        stroke_image = self.tranform2image(out)

        return stroke_image        



    def init_model(self, device, config, use_wandb=True):

        encoder = Encoder(d_model=self.config.enc_d_model, nhead=self.config.enc_nhead, dff=self.config.enc_dff,
                          nlayers=self.config.enc_n_layers, size_embedding=self.config.size_embedding, dropout=self.config.enc_dropout)
                    
        decoder = Decoder(size_embedding=self.config.size_embedding, num_components=self.config.dec_gmm_num_components,
                          out_units=2, layer_features=self.config.dec_layer_features)
        
        position_predictive_model = TransformerGMM(d_model=self.config.rel_d_model,nhead=self.config.rel_nhead,
                                                   dff=self.config.rel_dff, nlayers=self.config.rel_n_layers,
                                                   input_size= self.config.size_embedding + 2,
                                                   num_components= self.config.rel_gmm_num_components,
                                                   out_units = 2, dropout = self.config.rel_dropout
                                                  )

        embedding_predictive_model = TransformerGMM(d_model = self.config.rel_d_model, nhead = self.config.rel_nhead,
                                                    dff = self.config.rel_dff, nlayers = self.config.rel_n_layers,
                                                    input_size= self.config.size_embedding + 4,
                                                    num_components = self.config.rel_gmm_num_components,
                                                    out_units = self.config.size_embedding, dropout = self.config.rel_dropout
                                                  )

        encoder.to(device)
        decoder.to(device)
        position_predictive_model.to(device)
        embedding_predictive_model.to(device)

        if use_wandb:
            wandb.watch(encoder, log="all")
            wandb.watch(decoder, log="all")
            wandb.watch(position_predictive_model, log="all")
            wandb.watch(embedding_predictive_model, log="all")

        return (encoder, decoder, position_predictive_model, embedding_predictive_model)



    def init_optimizers(self):
        list_autoencoder = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer_ae = torch.optim.Adam(list_autoencoder, lr=self.config.lr_ae)

        list_pos_pred = list(self.position_predictive_model.parameters())
        optimizer_pos_pred = torch.optim.Adam(list_pos_pred, lr=self.config.lr_pos_pred)

        list_emb_pred = list(self.embedding_predictive_model.parameters())
        optimizer_emb_pred = torch.optim.Adam(list_emb_pred, lr=self.config.lr_emb_pred)

        
        return (optimizer_ae, optimizer_pos_pred, optimizer_emb_pred)

    def train_step(self, train_loader, optimizers):

        optimizer_ae, optimizer_pos_pred, optimizer_emb_pred = optimizers    

        self.encoder.train()
        self.decoder.train()
        self.embedding_predictive_model.train()
        self.position_predictive_model.train()

        for batch_input, batch_target in iter(train_loader):

            self.encoder.zero_grad()
            self.decoder.zero_grad()
            self.embedding_predictive_model.zero_grad()
            self.position_predictive_model.zero_grad()
            
            # Parsing inputs
            enc_inputs, t_inputs, stroke_len_inputs, inputs_start_coord, inputs_end_coord, num_strokes_x_diagram_tensor = parse_inputs(batch_input,self.device)
            t_target_ink = parse_targets(batch_target,self.device)
            # Creating sequence length mask
            _, look_ahead_mask, _ = generate_3d_mask(enc_inputs, stroke_len_inputs, self.device)
            # Encoder forward
            encoder_out = self.encoder(enc_inputs.permute(1,0,2), stroke_len_inputs, look_ahead_mask)
            # decoder forward
            encoder_out_reshaped = encoder_out.unsqueeze(dim=1).repeat(1,t_inputs.shape[1],1).reshape(-1, encoder_out.shape[-1])
            t_inputs_reshaped = t_inputs.reshape(-1,1)
            decoder_inp = torch.cat([encoder_out_reshaped, t_inputs_reshaped], dim = 1)
            strokes_out, ae_mu, ae_sigma, ae_pi= self.decoder(decoder_inp)
            
            # Random/Ordered Sampling
            pred_inputs, pred_input_seq_len, context_pos, pred_targets, target_pos = random_index_sampling(encoder_out,inputs_start_coord,
                                                                                            inputs_end_coord,num_strokes_x_diagram_tensor,
                                                                                            input_type =self.config.input_type,
                                                                                            num_predictive_inputs = self.config.num_predictive_inputs,
                                                                                            replace_padding = self.config.replace_padding,
                                                                                            end_positions = self.config.end_positions
                                                                                            , device = self.device)
            # Detaching gradients of pred_targets (Teacher forcing)
            if self.config.stop_predictive_grad:
                pred_inputs = pred_inputs.detach()
                pred_input_seq_len = pred_input_seq_len.detach()
                context_pos = context_pos.detach()
                pred_targets = pred_targets.detach()
                target_pos = target_pos.detach() #Detaching gradients of pred_inputs (No influence of Relational Model)
            # Concatenating inputs for relational model
            pos_model_inputs = torch.cat([pred_inputs, context_pos], dim = 2)
            pred_model_inputs = torch.cat([pred_inputs, context_pos, target_pos.unsqueeze(dim = 1).repeat(1, pred_inputs.shape[1], 1)], dim = 2)
            # Predictive model Teacher forcing
            emb_pred_mu, emb_pred_sigma, emb_pred_pi = self.embedding_predictive_model(pred_model_inputs, pred_input_seq_len.int(), None)
            # Position model 
            pos_pred_mu, pos_pred_sigma, pos_pred_pi = self.position_predictive_model(pos_model_inputs, pred_input_seq_len.int(), None)
            
            loss_ae = -1*(logli_gmm_logsumexp(t_target_ink, ae_mu, ae_sigma, ae_pi).sum())
            loss_pos_pred = -1*(logli_gmm_logsumexp(target_pos, pos_pred_mu, pos_pred_sigma, pos_pred_pi).sum())
            loss_emb_pred = -1*(logli_gmm_logsumexp(pred_targets, emb_pred_mu, emb_pred_sigma, emb_pred_pi).sum())
            
            loss_total = loss_pos_pred + loss_emb_pred + loss_ae
            #sys.exit(0)
            loss_total.backward()

            optimizer_pos_pred.step()
            optimizer_emb_pred.step()
            optimizer_ae.step()

        return (loss_ae, loss_pos_pred, loss_emb_pred, loss_total)


    def save_weights(self, path_gen, path_sub, use_wandb=True):

        torch.save(self.encoder.state_dict(), os.path.join(path_sub, 'encoder.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(path_sub, 'decoder.pth'))
        torch.save(self.embedding_predictive_model.state_dict(), os.path.join(path_sub, 'emb_pred.pth'))
        torch.save(self.position_predictive_model.state_dict(), os.path.join(path_sub, 'pos_pred.pth'))

        if use_wandb:
            wandb.save(os.path.join(path_sub,'*.pth'),base_path='/'.join(path_gen.split('/')[:-2]))


    def load_weights(self):
        self.e_shared.load_state_dict(
            torch.load(self.config.model_path + 'encoder.pth',map_location=torch.device(self.device)))

        self.d_shared.load_state_dict(
            torch.load(self.config.model_path + 'decoder.pth',map_location=torch.device(self.device)))

        self.d2.load_state_dict(
            torch.load(self.config.model_path + 'emb_pred.pth',map_location=torch.device(self.device)))

        self.denoiser.load_state_dict(
            torch.load(self.config.model_path + 'pos_pred.pth',map_location=torch.device(self.device)))


    def fit(self, n_epochs:int = 1):
                        


        if self.config.use_gpu and torch.cuda.is_available():
            print("Training in " + torch.cuda.get_device_name(0))  
        else:
            print("Training in CPU")

        if self.config.save_weights:
            path_save_weights = self.config.root_path + self.config.save_path
        try:
            os.mkdir(path_save_weights)
        except OSError:
            pass

        #optimizer_ae, optimizer_pos_pred, optimizer_emb_pred 
        optimizers = self.init_optimizers()
    

        train_loader = get_batch_iterator(self.config.dataset_path)

        for epoch in tqdm(range(self.config.num_epochs)):
            loss_ae, loss_pos_pred, loss_emb_pred, loss_total = self.train_step(train_loader, optimizers)
            #TODO valid_loader shape: (n_ejemplos, num_strokesxdiagrama, num_puntos, 2)
            #generated_strokes = test_strokes(valid_loader)

            if self.use_wandb:
                wandb.log({"train_epoch":epoch+1,
                            #"Generated strokes": [wandb.Image(img) for img in generated_strokes],
                            "loss_ae":loss_ae.item(),
                            "loss_pos_pred":loss_pos_pred.item(),
                            "loss_emb_pred":loss_emb_pred.item(), 
                            "loss_total":loss_total.item()})

            if self.config.save_weights and ((epoch+1)% int(self.config.num_epochs/self.config.num_backups))==0:
                path_save_epoch = path_save_weights + 'epoch_{}'.format(epoch+1)
                
                try:
                    os.mkdir(path_save_epoch)
                except OSError:
                    pass

                self.save_weights(path_save_weights, path_save_epoch, self.use_wandb)

            print("Losses")
            print('Epoch [{}/{}], Loss autoencoder: {:.4f}'.format(epoch+1, self.config.num_epochs, loss_ae.item()))
            print('Epoch [{}/{}], Loss position prediction: {:.4f}'.format(epoch+1, self.config.num_epochs, loss_pos_pred.item()))
            print('Epoch [{}/{}], Loss embedding prediction: {:.4f}'.format(epoch+1, self.config.num_epochs, loss_emb_pred.item()))
            print('Epoch [{}/{}], Loss total: {:.4f}'.format(epoch+1, self.config.num_epochs, loss_total.item()))
        
        if self.use_wandb:
            wandb.finish()
        




