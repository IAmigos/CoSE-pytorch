#from chamferdist import ChamferDistance
import torch
from models import logli_gmm_logsumexp
import numpy as np
from .encoder import *
from .utils import *
import time

def eval_parse_input(batch_input, device):
    '''
    parsing batch_input
    Args:
        batch_input (Tensor)
        device (str)
    '''
    #parse inputs
    encoder_inputs = batch_input['encoder_inputs'].squeeze(dim = 0)
    num_strokes = batch_input['num_strokes'].squeeze(dim = 0)
    strok_len_inputs = batch_input['seq_len'].squeeze(dim = 0)
    start_coord = batch_input['start_coord'].squeeze(dim = 0).squeeze()
    end_coord = batch_input['end_coord'].squeeze(dim = 0).squeeze()
    #save to devices
    encoder_inputs = encoder_inputs.to(device)
    num_strokes = num_strokes.to(device)
    strok_len_inputs = strok_len_inputs.to(device)
    start_coord = start_coord.to(device)
    end_coord = end_coord.to(device)

    return encoder_inputs, num_strokes, strok_len_inputs, start_coord, end_coord

def eval_parse_target(batch_target, device):
    '''
    parsing batch_target
    Args:
        batch_target (Tensor)
        device (str)
    '''
    #parse targets
    target_ink = batch_target['t_target_ink'].squeeze(dim = 0)
    target_strok_len = batch_target["seq_len"].squeeze(dim = 0)
    target_pos = batch_target["start_coord"].squeeze(dim =  0)
    target_end_coord = batch_target["end_coord"].squeeze(dim =  0)
    target_strokes = batch_target["stroke"].squeeze(dim = 0)
    target_num_strokes = batch_target["num_strokes"].squeeze(dim = 0)
    #save to devices
    target_ink = target_ink.to(device)
    target_strok_len = target_strok_len.to(device)
    target_pos = target_pos.to(device)
    target_strokes = target_strokes.to(device)
    target_num_strokes = target_num_strokes.to(device)

    return target_ink, target_strok_len, target_pos, target_strokes, target_num_strokes

def draw_pred_strokes_ar_step(models, stroke_i, context_embeddings, ar_start_pos, mean_channel, std_channel, rel_nhead, device ,decoded_length = 50):
    #parse models
    position_predictive_model, embedding_predictive_model, decoder = models
    # inputs positions up unitl stroke_index
    input_pos = np.concatenate(ar_start_pos[:stroke_i], axis=1)
    
    # define seq_len and n_strokes
    n_strokes_ = context_embeddings.size(1)
    seq_len = (torch.ones_like(context_embeddings[:, 0, 0])*n_strokes_).int()
    
    # inputs to position predictive model
    pos_ar_inputs = torch.cat([context_embeddings, torch.tensor(input_pos).to(device)], dim = 2)
    
    # mask for position predictive model
    seq_mask_rel = 1 - (torch.arange(seq_len.max().item()).to(device)[None, :] < seq_len[:, None]).float()
    seq_mask_rel  = seq_mask_rel.masked_fill(seq_mask_rel == 1, float('-inf')).unsqueeze(dim=1).repeat(1,seq_mask_rel.shape[1],1).repeat_interleave(rel_nhead,dim = 0)
    
    # pass to position predictive model
    pos_pred_mu, pos_pred_sigma, pos_pred_pi = position_predictive_model(pos_ar_inputs, seq_len, None, seq_mask_rel)
    
    # draw sample to position predictive model
    out_pos_sample = position_predictive_model.draw_sample(pos_pred_mu, pos_pred_sigma, pos_pred_pi, greedy = False)
    
    # additional inputs to embedding predictive model
    target_pos = out_pos_sample.detach().cpu().numpy()
    
    # pass to embedding predictive model with same masks
    emb_pred_mu, emb_pred_sigma, emb_pred_pi = embedding_predictive_model(pos_ar_inputs, seq_len, torch.tensor(target_pos).to(device), src_mask  = seq_mask_rel)
    
    # draw sample to embedding predictive model
    emb_pred = embedding_predictive_model.draw_sample(emb_pred_mu, emb_pred_sigma, emb_pred_pi, greedy = False)
    
    # add to already available embeddings and positions
    context_embeddings = torch.cat([context_embeddings, emb_pred.unsqueeze(dim = 0)], dim  =1)
    ar_start_pos.append(np.expand_dims(target_pos, axis = 0))
    
    # preparing to draw
    emb_ = context_embeddings[0]

    draw_seq_len = np.array([decoded_length]*(stroke_i + 1))

    predicted_batch_stroke = decode_sequence(decoder, emb_, draw_seq_len, device)

    predicted_batch_strat_pos = torch.tensor(np.transpose(np.concatenate(ar_start_pos[:stroke_i+1], axis=1), [1,0,2])).to(device)
 
    return context_embeddings, ar_start_pos, predicted_batch_stroke, predicted_batch_strat_pos, draw_seq_len

def quantitative_ae_step(encoder_out, out_eval_parse_input, out_eval_parse_target, models, stats_channels, eval_loss, device, rel_nhead ,recons_analysis = True, pred_analysis = True, include_diagram_loss = False):
    '''
    Evaluation step
    Args:
        eval_loss (AggregatedAvg)
        batch_input (Tensor)
        batch_target (Tensor)
        device (str)
        recons_analysis (bool): whether to perform reconstruction analysis
        pred_analysis (bool): whether to perform predictive analysis
        include_diagram_loss (bool): whether to include diagram loss for the reconstruction analysis
    Returns:

    '''
    # parsing models
    embedding_predictive_model, decoder = models
    # parsing batch_inputs and targets
    encoder_inputs, num_strokes, strok_len_inputs, start_coord, end_coord = out_eval_parse_input
    target_ink, target_strok_len, target_pos, target_strokes, target_num_strokes = out_eval_parse_target
    # parsing statistics variables
    mean_channel, std_channel = stats_channels
    # passing inputs to encoding
    # comb_mask, look_ahead_mask, _ = generate_3d_mask(encoder_inputs, strok_len_inputs,device, enc_nhead)
    # encoder_out = encoder(encoder_inputs.permute(1,0,2), strok_len_inputs, comb_mask)
    #losses saved in a dict and aggregated by eval_loss
    losses = dict()

    if recons_analysis:
        # decoding sequences
        embedding = encoder_out.detach().clone()
        seq_len = target_strok_len.detach().clone()
        recon_stroke = decode_sequence(decoder, embedding, seq_len, device)
        # padded sequences to list of strokes
        target_strokes_list = batch_to_real_stroke_list(target_strokes, target_pos, seq_len, std_channel, mean_channel, device)
        recons_strokes_list = batch_to_real_stroke_list(recon_stroke, target_pos, seq_len, std_channel, mean_channel, device)
        # chamfer loss for reconstructed strokes
        recon_chamfer = evaluate_chamfer(recons_strokes_list, target_strokes_list)
        # saving loss for reconstructed strokes
        losses["rc_chamfer_stroke"]  = recon_chamfer

        if include_diagram_loss:
            # diagram level arrays
            target_diagram = np.vstack(gt_strokes)
            recons_diagram = np.vstack(recon_strokes)
            # chamfer loss for diagram
            recon_diag_chamfer = evaluate_chamfer(target_diagram, recons_diagram)
            # saving loss for diagram
            losses["rc_chamfer_diagram"] = recon_diag_chamfer

    return eval_loss.add(losses), recon_chamfer

def qualitative_ae_step(encoder_out, out_eval_parse_input, out_eval_parse_target, models, stats_channels, device, rel_nhead, num_extra_pred = 5):
    '''
    Qualititive eval
    Args:
        ....
        idx: index to eval
    '''
    # parsing models
    position_predictive_model, embedding_predictive_model, decoder = models
    # parsing batch_inputs and targets
    encoder_inputs, num_strokes, strok_len_inputs, start_coord, end_coord = out_eval_parse_input
    target_ink, target_strok_len, target_pos, target_strokes, target_num_strokes = out_eval_parse_target
    # parsing statistics variables
    mean_channel, std_channel = stats_channels

    embedding = encoder_out.detach().clone()
    seq_len = target_strok_len.detach().clone()

    embedding = embedding.reshape(num_strokes.size(0),-1, embedding.size(1))

    emb_ = embedding[0][:num_strokes[0]]

    print(emb_.shape)

    draw_seq_len = np.array([50]*(emb_.size(0)))

    predicted_batch_stroke = decode_sequence(decoder, emb_, draw_seq_len, device)

    return predicted_batch_stroke, target_pos, draw_seq_len


def qualitative_eval_step(encoder_out, out_eval_parse_input, out_eval_parse_target, models, stats_channels, device, rel_nhead, num_extra_pred = 5):
    '''
    Qualititive eval
    Args:
        ....
        idx: index to eval
    '''
    # parsing models
    position_predictive_model, embedding_predictive_model, decoder = models
    # parsing batch_inputs and targets
    encoder_inputs, num_strokes, strok_len_inputs, start_coord, end_coord = out_eval_parse_input
    target_ink, target_strok_len, target_pos, target_strokes, target_num_strokes = out_eval_parse_target
    # parsing statistics variables
    mean_channel, std_channel = stats_channels

    embedding = encoder_out.detach().clone()
    seq_len = target_strok_len.detach().clone()

    target_strokes_list = batch_to_real_stroke_list(target_strokes, target_pos, seq_len, std_channel, mean_channel, device)

    all_strokes = np.concatenate(target_strokes_list)

    embedding = embedding.reshape(num_strokes.size(0),-1, embedding.size(1))

    # Auto-regressive prediction
    n_strokes = target_num_strokes[0].item()
    n_strokes += num_extra_pred
    context_ids = 2
    context_embeddings = embedding[:, :context_ids]
    start_positions = np.transpose(target_pos.detach().cpu().numpy(), [1, 0, 2])

    ar_start_pos = np.split(start_positions[:, 0:context_ids], context_ids, axis=1)

    for stroke_i in range(context_ids, n_strokes):
        context_embeddings, ar_start_pos, predicted_batch_stroke, predicted_batch_strat_pos, draw_seq_len = draw_pred_strokes_ar_step(models, stroke_i, context_embeddings, ar_start_pos, mean_channel, std_channel, rel_nhead, device, decoded_length = 50)
    
    return predicted_batch_stroke, predicted_batch_strat_pos, draw_seq_len

def quantitative_eval_step(encoder_out, out_eval_parse_input, out_eval_parse_target, models, stats_channels, eval_loss, device, rel_nhead ,recons_analysis = True, pred_analysis = True, include_diagram_loss = False):
    '''
    Evaluation step
    Args:
        eval_loss (AggregatedAvg)
        batch_input (Tensor)
        batch_target (Tensor)
        device (str)
        recons_analysis (bool): whether to perform reconstruction analysis
        pred_analysis (bool): whether to perform predictive analysis
        include_diagram_loss (bool): whether to include diagram loss for the reconstruction analysis
    Returns:

    '''
    # parsing models
    embedding_predictive_model, decoder = models
    # parsing batch_inputs and targets
    encoder_inputs, num_strokes, strok_len_inputs, start_coord, end_coord = out_eval_parse_input
    target_ink, target_strok_len, target_pos, target_strokes, target_num_strokes = out_eval_parse_target
    # parsing statistics variables
    mean_channel, std_channel = stats_channels
    # passing inputs to encoding
    # comb_mask, look_ahead_mask, _ = generate_3d_mask(encoder_inputs, strok_len_inputs,device, enc_nhead)
    # encoder_out = encoder(encoder_inputs.permute(1,0,2), strok_len_inputs, comb_mask)
    #losses saved in a dict and aggregated by eval_loss
    losses = dict()

    if recons_analysis:
        # decoding sequences
        embedding = encoder_out.detach().clone()
        seq_len = target_strok_len.detach().clone()
        recon_stroke = decode_sequence(decoder, embedding, seq_len, device)
        # padded sequences to list of strokes
        target_strokes_list = batch_to_real_stroke_list(target_strokes, target_pos, seq_len, std_channel, mean_channel, device)
        recons_strokes_list = batch_to_real_stroke_list(recon_stroke, target_pos, seq_len, std_channel, mean_channel, device)
        # chamfer loss for reconstructed strokes
        recon_chamfer = evaluate_chamfer(recons_strokes_list, target_strokes_list)
        # saving loss for reconstructed strokes
        losses["rc_chamfer_stroke"]  = recon_chamfer

        if include_diagram_loss:
            # diagram level arrays
            target_diagram = np.vstack(gt_strokes)
            recons_diagram = np.vstack(recon_strokes)
            # chamfer loss for diagram
            recon_diag_chamfer = evaluate_chamfer(target_diagram, recons_diagram)
            # saving loss for diagram
            losses["rc_chamfer_diagram"] = recon_diag_chamfer

    if pred_analysis:
        # setting 
        given_strokes  = 2 #TODO: consider this config variable in json
        emb_pred_mu, emb_pred_sigma, emb_pred_pi, pred_emb, target_emb = predict_embedding_ordered_batch(embedding_predictive_model, device, embedding, start_coord, given_strokes = given_strokes, rel_nhead = rel_nhead)
        # nll loss
        nll_embedingg_loss = -1*logli_gmm_logsumexp(target_emb, emb_pred_mu, emb_pred_sigma, emb_pred_pi)[:,0].detach().cpu().numpy()
        # saving loss
        losses["nll_embedding"] = nll_embedingg_loss
        # decode all posible embeddings
        all_pred_strokes, all_pred_stroke_lens, all_emb_pi, all_emb_samples = decode_embedding_all_components([embedding_predictive_model, decoder], emb_pred_mu, emb_pred_sigma, emb_pred_pi, seq_len, device, given_strokes)
        # retrieving important values
        n_components = all_emb_pi.shape[1]
        all_target_pos = target_pos[given_strokes:].repeat(n_components,1,1)
        # padded sequences to list of strokes
        pred_strokes_list = batch_to_real_stroke_list(all_pred_strokes, all_target_pos, all_pred_stroke_lens, std_channel, mean_channel, device)
        target_pred_strokes_list = target_strokes_list[given_strokes:]*n_components
        # chamfer loss for predicted strokes
        results_pred = evaluate_chamfer(pred_strokes_list, target_pred_strokes_list)
        #
        all_comp_chamfer = np.transpose(np.reshape(np.array(results_pred), [n_components, -1]), [1, 0])
        min_chamfer = np.min(all_comp_chamfer, axis=1)
        min_comp_id = np.argmin(all_comp_chamfer, axis=1)
        # best embeddings according to their chamfer distance
        best_embedding_idx = torch.vstack([torch.arange(all_emb_samples.shape[0]), torch.tensor(min_comp_id)]).T
        # retrieving best embeddings
        best_embeddings = torch.stack([all_emb_samples[index][min_comp_id[index]] for index in range(all_emb_samples.size(0))])
        # saving loss and best embeddings 
        #TODO there is an additional processing to order the embeddings in lowest error (On hold)
        losses["pred_chamfer_stroke"] =min_chamfer
        
    return eval_loss.add(losses), recon_chamfer, min_chamfer


def _get_reconstruction_metrics(expected_strokes, expected_start_coord, pred_embedding, recon_start_coord, strok_len_inputs, decoder, mean_channel, std_channel,device, skip_rows = 0):
    loss_ae = 0
    mean_chamfer_dist = 0
    q = 0
    chamferDist = ChamferDistance()
    total_chamfer_dist = 0
    recons_strokes = []
    for i, stroke_embedding in enumerate(pred_embedding):
        if strok_len_inputs[i] > 0:
            t_inputs = torch.linspace(0, 1, steps=strok_len_inputs[i]).to(device)
            t_inp = t_inputs.reshape(1,-1)
            stroke_emb = stroke_embedding.reshape(1,-1).repeat(t_inp.size(-1), 1)
            t_inp = t_inp.reshape(-1,1)
            recons_input = torch.cat([stroke_emb, t_inp], dim = 1)
            recons_stroke, ae_mu, ae_sigma, ae_pi= decoder(recons_input)
            orig_stroke_ = expected_strokes[i:i+1,:strok_len_inputs[i],:2]
            orig_pos = expected_start_coord[i:i+1,:]
            recon_stroke_ = recons_stroke.reshape(-1, strok_len_inputs[i],2)
            recon_pos = recon_start_coord[i:i+1,:]
            if i >= skip_rows:
                recons_strokes.append(recon_stroke_.squeeze(dim = 0))
                loss_ae += -1*(logli_gmm_logsumexp(expected_strokes[i:i+1,:strok_len_inputs[i],:2], ae_mu, ae_sigma, ae_pi).mean())
                orig_stroke_ = torch.from_numpy(orig_stroke_.cpu().detach().numpy()*std_channel + mean_channel).to(device) + orig_pos
                recon_stroke_ = torch.from_numpy(recon_stroke_.cpu().detach().numpy()*std_channel + mean_channel).to(device) + recon_pos
                total_chamfer_dist += chamferDist(orig_stroke_, recon_stroke_)
                q+=1
            else:
                recons_strokes.append(orig_stroke_.squeeze(dim = 0))
    loss_ae = loss_ae/q
    mean_chamfer_dist = total_chamfer_dist/q
    return loss_ae, mean_chamfer_dist, recons_strokes

def _get_prediction_metrics(encoder_inputs, strok_len_inputs, diagram_embedding, start_pos_base, num_strokes, models, device, mean_channel, std_channel, use_autoregressive = False):
    decoder, position_predictive_model, embedding_predictive_model = models
    # Num_strokes_iniciales
    n_strokes_init = 2
    #for every diagram
    loss_pos_pred =  []
    loss_emb_pred = []
    pred_cd = []
    reconstructed_diagrams_strokes = []
    reconstructed_diagrams_start_pos = []
    num_diagrams = diagram_embedding.size(0)
    for index_diagram in range(num_diagrams):
        #diagram inputs
        num_strokes_one = num_strokes[index_diagram]
        one_diagram = diagram_embedding[index_diagram].unsqueeze(dim =0)
        start_pos_one = start_pos_base[index_diagram].unsqueeze(dim = 0)
        expected_output_one = encoder_inputs.reshape(num_diagrams,-1,encoder_inputs.size(1), encoder_inputs.size(2))[index_diagram,:num_strokes_one,:,:]
        stroke_len_one = strok_len_inputs.reshape(num_diagrams,-1)[index_diagram,:]
        #starting config
        inp_diagram_cum = one_diagram[:,:n_strokes_init,:]
        inp_start_pos_cum = start_pos_one[:,:n_strokes_init,:]
        #iterating to predict next stroke every time
        for j in range(n_strokes_init, num_strokes[index_diagram]):
            #inputs
            if not use_autoregressive:
                inp_diagram = one_diagram[:,:j,:]
                inp_start_pos = start_pos_one[:,:j,:]
            else:
                inp_diagram = inp_diagram_cum[:,:j,:]
                inp_start_pos = inp_start_pos_cum[:,:j,:]
            #targets
            target_diagram = one_diagram[:,j,:]
            target_start_pos = start_pos_one[:,j,:]
            #predictions
            inp_pos_model = torch.cat([inp_diagram, inp_start_pos], dim = 2)
            inp_num_strokes = torch.tensor([inp_diagram.size(1)])
            #pos prediction
            pos_pred_mu, pos_pred_sigma, pos_pred_pi = position_predictive_model(inp_pos_model, inp_num_strokes, None)
            pos_pred = position_predictive_model.draw_sample(pos_pred_mu, pos_pred_sigma, pos_pred_pi)  
            #next embedding prediction
            #pred_model_inputs = torch.cat([inp_diagram, inp_start_pos, pos_pred.unsqueeze(dim = 1).repeat(1, inp_num_strokes, 1)], dim = 2)
            tgt_cond = pos_pred.squeeze(dim = 1)
            emb_pred_mu, emb_pred_sigma, emb_pred_pi = embedding_predictive_model(inp_pos_model, inp_num_strokes, tgt_cond)
            emb_pred = embedding_predictive_model.draw_sample(emb_pred_mu, emb_pred_sigma, emb_pred_pi)
            #losses
            loss_pos_pred.append(-1*(logli_gmm_logsumexp(inp_start_pos, pos_pred_mu, pos_pred_sigma, pos_pred_pi).mean()).item())
            loss_emb_pred.append(-1*(logli_gmm_logsumexp(target_diagram, emb_pred_mu, emb_pred_sigma, emb_pred_pi).mean()).item())
            #updating diagrams for autoregressiveness
            inp_diagram_cum = torch.cat([inp_diagram_cum, emb_pred.unsqueeze(dim = 1)], dim = 1)
            inp_start_pos_cum = torch.cat([inp_start_pos_cum, pos_pred.unsqueeze(dim = 1)], dim = 1)
        #print(stroke_len_one)
        _, chamf_dist, recon_strokes = get_reconstruction_metrics(expected_strokes= expected_output_one,
                                                                  expected_start_coord=start_pos_one.squeeze(dim=0)[:num_strokes[index_diagram]],
                                                                  pred_embedding=inp_diagram_cum.squeeze(dim = 0),
                                                                  recon_start_coord= inp_start_pos_cum.squeeze(dim=0),
                                                                  strok_len_inputs=stroke_len_one,
                                                                  decoder= decoder,
                                                                  mean_channel = mean_channel,
                                                                  std_channel = std_channel,
                                                                  device = device,
                                                                  skip_rows = n_strokes_init)

        pred_cd.append(chamf_dist.item())
        reconstructed_diagrams_strokes.append(recon_strokes)
        reconstructed_diagrams_start_pos.append(inp_start_pos_cum)
        
    total_emb_loss = np.array(loss_emb_pred).mean()
    total_pos_loss = np.array(loss_pos_pred).mean()
    total_pred_cd  = np.array(pred_cd).mean()
    
    return total_emb_loss, total_pos_loss, total_pred_cd, reconstructed_diagrams_strokes, reconstructed_diagrams_start_pos