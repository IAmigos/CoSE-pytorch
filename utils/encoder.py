import torch

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.float().masked_fill(mask == 0, float(1)).masked_fill(mask == 1, float(0.0))
    return mask

def generate_3d_mask(seq_input, stroke_len_inputs, device, n_heads):
    look_ahead_mask = generate_square_subsequent_mask(seq_input.shape[1]).to(device)
    seq_mask = 1 - (torch.arange(stroke_len_inputs.max().item()).to(device)[None, :] < stroke_len_inputs[:, None]).float()
    #seq_mask  = seq_mask.masked_fill(seq_mask == 1, float('-inf')).unsqueeze(dim=1).repeat(1,seq_mask.shape[1],1)
    seq_mask  = seq_mask.unsqueeze(dim=1).repeat(1,seq_mask.shape[1],1)
    #return torch.minimum(seq_mask, look_ahead_mask).repeat_interleave(n_heads, dim = 0), look_ahead_mask, seq_mask
    return torch.maximum(seq_mask, look_ahead_mask).repeat_interleave(n_heads, dim = 0), look_ahead_mask, seq_mask

def decode_embedding_all_components(models, emb_pred_mu, emb_pred_sigma, emb_pred_pi, seq_len, device, given_strokes = 2):
    pred_emb_model, decoder = models
    emb_sample, emb_pi = pred_emb_model.gmm.draw_sample_every_component(emb_pred_mu, emb_pred_sigma, emb_pred_pi, greedy=False)
    # reshaping
    emb_samples = emb_sample[:, 0, :, :].detach()
    emb_pis = emb_pi[:, 0, :].detach().cpu().numpy()
    n_components = emb_pis.shape[1]
    # create possible predictions for all components
    emb_samples_compwise = emb_samples.permute(1,0,2).reshape(-1, emb_samples.size(-1)) #first: first component prediction for all batches, second: second component prediction for all batches, so forth
    emb_stroke_lens = seq_len[given_strokes:].repeat(n_components) #first: stroke lenghts for all batches in first component, and so forth
    #decoding strokes
    recon_strokes_pred = decode_sequence(decoder, emb_samples_compwise, emb_stroke_lens, device)
    
    return recon_strokes_pred.detach(), emb_stroke_lens, emb_pis, emb_samples

def predict_embedding_ordered_batch(pred_emb_model, device, embedding, start_coord, given_strokes = 2, rel_nhead = 4):
    '''
    Returns predictions and target for the ordered batch embedding in input
    Args:
        pred_emb_model (Model): the prediction embedding model
        device (str): the device to allocate tensors
        embedding (Tensor): the embedding from input batch
        start_coord (Tensor): the positions from input batch
        given_strokes (int): number of initial given strokes
    Outputs:
        pred_emb (Tensor): predicted embeddings
        target_emb (Tensor): target embeddings
    '''
    n_strokes = embedding.size(0)
    n_predictions = n_strokes -  given_strokes
    embedding_unsqueezed = embedding.unsqueeze(dim = 0)
    
    # create data with padding for autoregressive prediction
    inp_emb = embedding_unsqueezed.repeat(n_predictions, 1,1)[:,:-1]
    mask = (torch.triu(torch.ones(n_strokes, n_strokes-1), diagonal = 1) == 0).float()
    mask = mask[given_strokes-1:-1].to(device)
    
    # create input variables to model
    ## inputs
    inp_emb = inp_emb*mask.unsqueeze(-1).repeat(1,1, inp_emb.shape[-1])
    inp_pos = start_coord.unsqueeze(dim = 0).repeat(n_predictions,1,1)[:,:-1]
    seq_len_inputs = torch.arange(given_strokes, n_strokes).int().to(device)
    ## targets
    target_pos = start_coord[given_strokes:]
    target_emb = embedding[given_strokes:]
    
    # create mask for TransformerModel in Predictive Embedding Model
    seq_mask_rel  = 1 - (torch.arange(seq_len_inputs.max().item()).to(device)[None, :] < seq_len_inputs[:, None]).float()
    seq_mask_rel  = seq_mask_rel.masked_fill(seq_mask_rel == 1, float('-inf')).unsqueeze(dim=1).repeat(1,seq_mask_rel.shape[1],1).repeat_interleave(rel_nhead,dim = 0)
    
    # pass to Predictive Model
    pos_model_inputs = torch.cat([inp_emb, inp_pos], dim = -1)
    emb_pred_mu, emb_pred_sigma, emb_pred_pi = pred_emb_model(pos_model_inputs, seq_len_inputs, target_pos, src_mask  = seq_mask_rel)
    pred_emb = pred_emb_model.draw_sample(emb_pred_mu, emb_pred_sigma, emb_pred_pi)
    
    return emb_pred_mu, emb_pred_sigma, emb_pred_pi, pred_emb, target_emb


def batch_to_real_stroke_list(batch_strokes, batch_start_pos, batch_seq_len, std_channel, mean_channel, device):
    recon_strokes_pred = batch_strokes*torch.tensor(std_channel).to(device) + torch.tensor(mean_channel).to(device)
    recon_strokes_pred = recon_strokes_pred + batch_start_pos
    strokes_list = []
    for stroke_idx in range(recon_strokes_pred.shape[0]):
        strokes_list.append(recon_strokes_pred[stroke_idx, :batch_seq_len[stroke_idx]].detach().cpu().numpy())
    return strokes_list


def decode_sequence(decoder_model, embedding, seq_len, device):
    """Decodes an stroke into a sequence by mapping t in [0,1] to seq_len.

    Args:
      embedding (Tensor): stroke sample or dict of shape (1, latent_units).
      seq_len (np.array): # of sequence steps.

    Returns:
      decoded_seq (Tensor): decoded sequence
    """
    n_strokes = embedding.size(0)
    n_latent = embedding.size(1)
    max_len = seq_len.max().item()
    
    # reshaping embedding input
    embedding_inp = embedding.unsqueeze(1).repeat(1,max_len,1).reshape(-1, n_latent)
    
    # creating t_inputs
    t_vals = []
    for seq_index in range(len(seq_len)):
        range_t = torch.linspace(start = 0.0, end = 1.0, steps = seq_len[seq_index]) #tf.expand_dims(tf.linspace(0.0, 1.0, seq_len_tf[sid]), axis=1)
        t_ = torch.zeros(max_len,1)
        t_[:seq_len[seq_index],0] = range_t
        t_vals.append(t_)
    t_inputs = torch.cat(t_vals, axis=0)
    
    # reshaping t_inputs and concatenating
    t_inputs_reshaped = t_inputs.reshape(-1,1).to(device)
    decoder_inp = torch.cat([embedding_inp, t_inputs_reshaped], dim = 1)
    
    #pass to decoder
    
    strokes_out, ae_mu, ae_sigma, ae_pi= decoder_model(decoder_inp)
    
    #reshaping to strokes x diagram shape
    recon_stroke  = strokes_out.reshape(n_strokes, -1, strokes_out.shape[-1])
    
    return recon_stroke