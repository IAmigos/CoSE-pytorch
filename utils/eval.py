from chamferdist import ChamferDistance
import torch
from models import logli_gmm_logsumexp
import numpy as np

def get_reconstruction_metrics(expected_strokes, pred_embedding, strok_len_inputs, decoder, device, skip_rows = 0):
    loss_ae = 0
    chamf_dist = 0
    q = 0
    chamferDist = ChamferDistance()
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
            recon_stroke_ = recons_stroke.reshape(-1, strok_len_inputs[i],2)
            recons_strokes.append(recon_stroke_.squeeze(dim = 0))
            if i >= skip_rows:
                loss_ae += -1*(logli_gmm_logsumexp(expected_strokes[i:i+1,:strok_len_inputs[i],:2], ae_mu, ae_sigma, ae_pi).mean())
                chamfer_dist = chamferDist(orig_stroke_, recon_stroke_).detach().cpu().item()
                q+=1
    loss_ae = loss_ae/q
    chamf_dist = chamfer_dist/q
    return loss_ae, chamf_dist, recons_strokes

def get_prediction_metrics(encoder_inputs, strok_len_inputs, diagram_embedding, start_pos_base, num_strokes, models, use_autoregressive = False):
    decoder, position_predictive_model, embedding_predictive_model = models
    # Num_strokes_iniciales
    n_strokes_init = 2
    #for every diagram
    loss_pos_pred =  []
    loss_emb_pred = []
    pred_cd = []
    reconstructed_diagrams_strokes = []
    reconstructed_diagrams_start_pos = []
    for index_diagram in range(diagram_embedding.size(0)):
        #diagram inputs
        one_diagram = diagram_embedding[index_diagram].unsqueeze(dim =0)
        start_pos_one = start_pos_base[index_diagram].unsqueeze(dim = 0)
        num_strokes_one = num_strokes[index_diagram]
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
            pred_model_inputs = torch.cat([inp_diagram, inp_start_pos, pos_pred.unsqueeze(dim = 1).repeat(1, inp_num_strokes, 1)], dim = 2)
            emb_pred_mu, emb_pred_sigma, emb_pred_pi = embedding_predictive_model(pred_model_inputs, inp_num_strokes, None)
            emb_pred = embedding_predictive_model.draw_sample(emb_pred_mu, emb_pred_sigma, emb_pred_pi)
            #losses
            loss_pos_pred.append(-1*(logli_gmm_logsumexp(inp_start_pos, pos_pred_mu, pos_pred_sigma, pos_pred_pi).mean()).item())
            loss_emb_pred.append(-1*(logli_gmm_logsumexp(target_diagram, emb_pred_mu, emb_pred_sigma, emb_pred_pi).mean()).item())
            #updating diagrams for autoregressiveness
            inp_diagram_cum = torch.cat([inp_diagram_cum, emb_pred.unsqueeze(dim = 1)], dim = 1)
            inp_start_pos_cum = torch.cat([inp_start_pos_cum, pos_pred.unsqueeze(dim = 1)], dim = 1)
        _, chamf_dist, recon_strokes = get_reconstruction_metrics(expected_output_one, inp_diagram_cum.squeeze(dim = 0), stroke_len_one, decoder, skip_rows = n_strokes_init)
        pred_cd.append(chamf_dist)
        reconstructed_diagrams_strokes.append(recon_strokes)
        reconstructed_diagrams_start_pos.append(inp_start_pos_cum)
        
    total_emb_loss = np.array(loss_emb_pred).mean()
    total_pos_loss = np.array(loss_pos_pred).mean()
    total_pred_cd  = np.array(pred_cd).mean()

    return total_emb_loss, total_pos_loss, total_pred_cd, reconstructed_diagrams_strokes, reconstructed_diagrams_start_pos