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