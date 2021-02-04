import torch

def random_index_sampling(encoder_out,inputs_start_coord,inputs_end_coord,num_strokes_x_diagram_tensor,
                         input_type ="hybrid", num_predictive_inputs = 32, replace_padding = True, end_positions = False, device = None):
    #================================#
    diagram_embedding, padded_max_num_strokes, min_n_stroke, num_diagrams = reshape_stroke2diagram(encoder_out,num_strokes_x_diagram_tensor)
    start_pos_base = inputs_start_coord.reshape(num_diagrams,padded_max_num_strokes,2)
    #num_predictive_inputs = 32
    if input_type == "hybrid":
        num_predictive_inputs //= 2
    #creates indexes to gather from
    # -----------------------#
    all_input_emb = []
    all_input_start_pos = []
    all_seq_len_emb = []
    #-----------------------#
    all_target_emb = []
    all_target_start_pos = []
    all_n_inputs = []

    if input_type in ["random", "hybrid"]:
        for i in range(num_predictive_inputs):
            input_indexes, target_indexes, n_inputs = get_random_inp_target_pairs(num_strokes_x_diagram_tensor,
                                                                                  padded_max_num_strokes,
                                                                                  num_diagrams,
                                                                                  min_n_stroke, device)
            #-------------------------------------#
            input_embedding = torch.stack([diagram_embedding[diagram_index][input_indexes[diagram_index]] for diagram_index in range(num_diagrams)])
            input_start_pos = torch.stack([start_pos_base[diagram_index][input_indexes[diagram_index]] for diagram_index in range(num_diagrams)])
            target_embedding = torch.stack([diagram_embedding[diagram_index][target_indexes[diagram_index]] for diagram_index in range(num_diagrams)]).squeeze()
            target_start_pos = torch.stack([start_pos_base[diagram_index][target_indexes[diagram_index]] for diagram_index in range(num_diagrams)]).squeeze()
            seq_len_embedding = torch.ones([num_diagrams])*n_inputs
            #---------------------------------------#
            all_input_emb.append(input_embedding)
            all_input_start_pos.append(input_start_pos)
            all_target_emb.append(target_embedding)
            all_target_start_pos.append(target_start_pos)
            all_seq_len_emb.append(seq_len_embedding)
            all_n_inputs.append(n_inputs)
    if input_type in ["order", "hybrid"]:
        for i in range(num_predictive_inputs):
            input_indexes, target_indexes, n_inputs = get_ordered_inp_target_pairs(num_strokes_x_diagram_tensor,
                                                                                   padded_max_num_strokes,
                                                                                   num_diagrams,
                                                                                   min_n_stroke, device)
            #----------------------------------------------------#
            input_embedding = torch.stack([diagram_embedding[diagram_index][input_indexes[diagram_index]] for diagram_index in range(num_diagrams)])
            input_start_pos = torch.stack([start_pos_base[diagram_index][input_indexes[diagram_index]] for diagram_index in range(num_diagrams)])
            target_embedding = torch.stack([diagram_embedding[diagram_index][target_indexes[diagram_index]] for diagram_index in range(num_diagrams)]).squeeze()
            target_start_pos = torch.stack([start_pos_base[diagram_index][target_indexes[diagram_index]] for diagram_index in range(num_diagrams)]).squeeze()
            seq_len_embedding = torch.ones([num_diagrams])*n_inputs
            #-----------------------------------------------------#
            all_input_emb.append(input_embedding)
            all_input_start_pos.append(input_start_pos)
            all_target_emb.append(target_embedding)
            all_target_start_pos.append(target_start_pos)
            all_seq_len_emb.append(seq_len_embedding)
            all_n_inputs.append(n_inputs)
            #------------------------------------------------------#
    sampled_seq_len_emb = torch.stack(all_seq_len_emb)
    sampled_target_start_pos = torch.stack(all_target_start_pos)
    sampled_target_emb = torch.stack(all_target_emb)
    #-------------------------------------------------------------#
    sampled_input_start_pos = torch.zeros(32, num_diagrams, max(all_n_inputs),2)
    for i_sample, sampled_data in enumerate(all_input_start_pos):
        for i_diagram, content in enumerate(sampled_data):
            sampled_input_start_pos[i_sample,i_diagram,:int(all_seq_len_emb[i_sample][i_diagram]),:] = content

    sampled_input_emb = torch.zeros(32, num_diagrams, max(all_n_inputs),8)
    for i_sample, sampled_data in enumerate(all_input_emb):
        for i_diagram, content in enumerate(sampled_data):
            sampled_input_emb[i_sample,i_diagram,:int(all_seq_len_emb[i_sample][i_diagram]),:] = content
    #-------------------------------------------------------------#
    sampled_input_start_pos = sampled_input_start_pos.reshape(-1, sampled_input_start_pos.size(-2), sampled_input_start_pos.size(-1))
    sampled_input_emb = sampled_input_emb.reshape(-1, sampled_input_emb.size(-2), sampled_input_emb.size(-1))
    sampled_seq_len_emb = sampled_seq_len_emb.reshape(-1)
    sampled_target_start_pos = sampled_target_start_pos.reshape(-1, sampled_target_start_pos.size(-1))
    sampled_target_emb = sampled_target_emb.reshape(-1, sampled_target_emb.size(-1))
    
    return sampled_input_start_pos, sampled_input_emb, sampled_seq_len_emb, sampled_target_start_pos, sampled_target_emb

def gather_indexes(base_tensor, index_tensor, replace_padding = True):
    '''
    Simulates gather_nd and has an extra property to replace padding with values in first (stroke) in first (diagram)
    Args:
        index_tensor: list of var-len tensors
        base_tensor: tensor from which to sample from
    Returns:
        gathered_tensor_padded: tensor sampled and replaced in padding if option replace_padding = True
    '''
    tensor_var_len = [base_tensor[i,value,:].squeeze() for a in index_tensor for i,value in enumerate(a)]
    gathered_tensor_padded = torch.nn.utils.rnn.pad_sequence(tensor_var_len, batch_first=False, padding_value=0).permute(1,0,2)
    if replace_padding:
        gathered_tensor_padded = replacing_padding_with_embedding(base_tensor, gathered_tensor_padded, index_diagram = 0, index_stroke = 0)    
    return gathered_tensor_padded

def replacing_padding_with_embedding(ref_tensor, tensor_to_mod, index_diagram= 0, index_stroke= 0):
    '''
    replaces padded values with values from dia
    '''
    default_first_base_tensor = ref_tensor[0,0,:].repeat(tensor_to_mod.shape[0],tensor_to_mod.shape[1],1)
    return torch.where(tensor_to_mod != 0.0, tensor_to_mod, default_first_base_tensor)

def reshape_stroke2diagram(stroke_embedding,num_strokes_x_diagram_tensor):
    embedding_size = stroke_embedding.shape[-1]
    padded_max_n_strokes = torch.max(num_strokes_x_diagram_tensor).item()
    min_n_stroke = torch.min(num_strokes_x_diagram_tensor).item()
    num_diagrams = num_strokes_x_diagram_tensor.shape[0]
    diagram_embedding = stroke_embedding.reshape([num_diagrams, padded_max_n_strokes, embedding_size])
    return diagram_embedding, padded_max_n_strokes, min_n_stroke, num_diagrams

def get_random_inp_target_pairs(num_strokes_x_diagram_tensor, padded_max_num_strokes, num_diagrams, min_n_stroke, device):
    #TODO Revisar
    """Get a randomly generated input set and a target."""
    n_inputs = torch.randint(2, min_n_stroke,size = (1,)).to(device).item() #validated with tf
    target_indexes = (torch.rand([num_diagrams]).to(device)*(num_strokes_x_diagram_tensor - 1)).int() #validated with tf
    input_range = torch.arange(start=0, end = padded_max_num_strokes).repeat(num_diagrams,1).to(device) #validated with tf
    mask = ((input_range)< num_strokes_x_diagram_tensor.reshape(-1,1)) & ((input_range -1) != target_indexes.reshape(-1,1)) #validated with tf
    input_indexes = torch.multinomial((input_range*mask).float().to(device),n_inputs) #validated with tf
    return input_indexes, target_indexes, n_inputs

def get_ordered_inp_target_pairs(num_strokes_x_diagram_tensor, padded_max_num_strokes, num_diagrams, min_n_stroke, device, random_target = False):
    #TODO Revisar
    """Get a slice (i.e., window) randomly."""
    n_inputs = torch.randint(2, min_n_stroke,size = (1,)).to(device).item()
    start_index = torch.randint(0, min_n_stroke - n_inputs, size = (1,)).to(device).item()
    target_indexes = torch.tensor([start_index+n_inputs]).repeat(num_diagrams,1).to(device)
    input_range = torch.arange(start=0, end = padded_max_num_strokes).repeat(num_diagrams,1).to(device)    
    mask = ((input_range)< target_indexes) & ((input_range)>= start_index)
    input_indexes = input_range.masked_select(mask).reshape(num_diagrams, n_inputs)
    return input_indexes, target_indexes, n_inputs