import torch

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_3d_mask(seq_input, stroke_len_inputs):
    look_ahead_mask = generate_square_subsequent_mask(seq_input.shape[1])
    seq_mask = 1 - (torch.arange(stroke_len_inputs.max().item())[None, :] < stroke_len_inputs[:, None]).float()
    seq_mask  = seq_mask.masked_fill(seq_mask == 1, float('-inf')).unsqueeze(dim=2).repeat(1,1,seq_mask.shape[1])
    return torch.minimum(seq_mask, look_ahead_mask), look_ahead_mask, seq_mask