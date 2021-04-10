import torch
from utils import *

def pred_pos_emb(cose, sampled_input_emb, sampled_input_start_pos, sampled_target_start_pos, sampled_seq_len_emb):
    pos_model_inputs = torch.cat([sampled_input_emb, sampled_input_start_pos], dim = 2)
    tgt_cond = sampled_target_start_pos.squeeze(dim = 1)
    seq_len_inputs = sampled_seq_len_emb.int().to(cose.device)
    #
    look_ahead_mask = generate_square_subsequent_mask(pos_model_inputs.shape[1]).to(cose.device)
    seq_mask_rel = 1 - (torch.arange(seq_len_inputs.max().item()).to(cose.device)[None, :] < seq_len_inputs[:, None]).float()
    seq_mask_rel  = seq_mask_rel.unsqueeze(dim=1).repeat(1,seq_mask_rel.shape[1],1)
    #seq_mask_rel  = seq_mask_rel.masked_fill(seq_mask_rel == 1, float('-inf')).unsqueeze(dim=1).repeat(1,seq_mask_rel.shape[1],1).repeat_interleave(cose.config.rel_nhead,dim = 0)
    seq_mask_rel  = torch.maximum(seq_mask_rel, look_ahead_mask).repeat_interleave(cose.config.rel_nhead, dim = 0)
    #
    emb_pred_mu, emb_pred_sigma, emb_pred_pi = cose.embedding_predictive_model(pos_model_inputs, seq_len_inputs, tgt_cond, src_mask  = seq_mask_rel)
    # Position model
    pos_pred_mu, pos_pred_sigma, pos_pred_pi = cose.position_predictive_model(pos_model_inputs, seq_len_inputs, None, src_mask  = seq_mask_rel)

    out_pos_sample = cose.position_predictive_model.draw_sample(pos_pred_mu, pos_pred_sigma, pos_pred_pi, greedy = False)

    emb_pred = cose.embedding_predictive_model.draw_sample(emb_pred_mu, emb_pred_sigma, emb_pred_pi, greedy = False)
    return tgt_cond, emb_pred