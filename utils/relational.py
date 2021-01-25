import torch

def random_index_sampling(encoder_out,inputs_start_coord,inputs_end_coord,num_strokes_x_diagram_tensor,
                         input_type ="hybrid", num_predictive_inputs = 32, replace_padding = True, end_positions = False):
    
    #obtains diagram embedding (batch_strokes, embedding_size) -> (num_diagrams, padded_n_strokes, embedding_size)
    diagram_embedding, padded_max_num_strokes, min_n_stroke, num_diagrams = reshape_stroke2diagram(encoder_out,num_strokes_x_diagram_tensor)
    #creates indexes to gather from
    all_n_inputs = []
    all_input_indexes = []
    all_target_indexes = []
    all_seq_len = []

    if input_type == "hybrid":
        num_predictive_inputs //= 2

    if input_type in ["random", "hybrid"]:
        for i in range(num_predictive_inputs):
            input_indexes, target_indexes, n_inputs = get_random_inp_target_pairs(num_strokes_x_diagram_tensor,
                                                                                  padded_max_num_strokes,
                                                                                  num_diagrams,
                                                                                  min_n_stroke)
            all_input_indexes.append(input_indexes)        
            all_target_indexes.append(target_indexes)
            all_n_inputs.append(n_inputs)
            all_seq_len.append(torch.ones([num_diagrams])*n_inputs)

    if input_type in ["order", "hybrid"]:
        for i in range(num_predictive_inputs):
            input_indexes, target_indexes, n_inputs = get_ordered_inp_target_pairs(num_strokes_x_diagram_tensor,
                                                                                   padded_max_num_strokes,
                                                                                   num_diagrams,
                                                                                   min_n_stroke)
            all_input_indexes.append(input_indexes)
            all_target_indexes.append(target_indexes)
            all_n_inputs.append(n_inputs)
            all_seq_len.append(torch.ones([num_diagrams])*n_inputs)

    #preparing for tensor indexing
    input_range_n_inputs = torch.arange(start=0, end = num_diagrams).repeat(1,len(all_n_inputs)).permute(1,0).squeeze()
    gather_target_index = torch.stack([input_range_n_inputs,
                 torch.cat(all_target_indexes, dim = 0).squeeze()], dim = -1)
    start_pos_base = inputs_start_coord.reshape(num_diagrams,padded_max_num_strokes,2)
    end_pos_base = inputs_end_coord.reshape(num_diagrams,padded_max_num_strokes,2)
    #gathering indexes from base tensors
    pred_targets = torch.stack([diagram_embedding[i][j] for i,j in gather_target_index])
    pred_inputs = gather_indexes(diagram_embedding, all_input_indexes, replace_padding = True)
    pred_input_seq_len = torch.cat(all_seq_len,dim=0)
    if end_positions:
        start_pos = gather_indexes(start_pos_base, all_input_indexes, replace_padding = True)
        end_pos = gather_indexes(end_pos_base, all_input_indexes, replace_padding = True)
        context_pos = torch.cat([start_pos, end_pos], dim=-1)
    else:
        context_pos = gather_indexes(start_pos_base, all_input_indexes, replace_padding = True)
    return pred_inputs, pred_input_seq_len, context_pos, pred_targets

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

def get_random_inp_target_pairs(num_strokes_x_diagram_tensor, padded_max_num_strokes, num_diagrams, min_n_stroke):
    """Get a randomly generated input set and a target."""
    n_inputs = torch.randint(2, (min_n_stroke+1),size = (1,)).item()
    target_indexes = (torch.rand([num_diagrams])*num_strokes_x_diagram_tensor).int().reshape(-1,1)
    input_range = torch.arange(start=1, end = padded_max_num_strokes + 1).repeat(num_diagrams,1)
    mask = (input_range <= num_strokes_x_diagram_tensor.reshape(-1,1)) & (input_range != target_indexes)
    input_indexes = torch.multinomial((input_range*mask).float(),n_inputs) - 1
    return input_indexes, target_indexes, n_inputs

def get_ordered_inp_target_pairs(num_strokes_x_diagram_tensor, padded_max_num_strokes, num_diagrams, min_n_stroke, random_target = False):
    """Get a slice (i.e., window) randomly."""
    n_inputs = torch.randint(2, (min_n_stroke+1),size = (1,)).item()
    start_index = torch.randint(0, min_n_stroke - n_inputs + 1, size = (1,)).item()
    if not random_target:
        target_indexes = torch.tensor([start_index+n_inputs]).repeat(num_diagrams,1)
    else:
        target_indexes = (torch.rand([num_diagrams])*num_strokes_x_diagram_tensor).int().reshape(-1,1)
    input_range = torch.arange(start=1, end = padded_max_num_strokes + 1).repeat(num_diagrams,1)
    mask = ((input_range - 1)< target_indexes) & ((input_range - 1)>= start_index)
    input_indexes = input_range.masked_select(mask).reshape(num_diagrams, n_inputs)
    return input_indexes, target_indexes, n_inputs