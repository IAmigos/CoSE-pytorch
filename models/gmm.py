import torch
import torch.nn as nn
import torch.nn.functional as F


def adjust_temp(pi_tensor, temp):
    #UPDATE to log functions were made to support 0 values
    pi_tensor = torch.log1p(pi_tensor)/temp
    pi_tensor -= torch.max(pi_tensor, dim = -1, keepdim=True)[0]
    pi_tensor = torch.exp(pi_tensor) - 1.0
    pi_tensor /= torch.sum(pi_tensor, dim = -1, keepdim = True)
    return pi_tensor

class OutputModelGMMDense(nn.Module):

    def __init__(self,
                input_size,
                out_units, # dim of mu and sigma
                num_components):
                # sigma_activation=,
                # kernel_regularizer=None,
                # bias_regularizer=None,
                # prefix=""
        super(OutputModelGMMDense, self).__init__()

        self.out_units = out_units
        self.num_components = num_components

        # out_size = (mu, sigma) * num_components + num_components ( para K)
        self.component_size = out_units * num_components
        self.out_size = 2 * self.num_components + num_components
        self.layer_out = nn.Linear(input_size, self.out_size, bias=False)
        
    def forward(self, inputs):
        out_ = self.layer_out(inputs)
        out_dict = dict()
        out_mu, out_sigma, out_pi = torch.tensor_split(
            out_, [self.component_size, 2*self.component_size]) # (self.component_size, self.component_size, self.num_components)
        out_dict["mu"] = out_mu 
        out_dict["sigma"] = torch.exp(out_sigma)
        out_dict["pi"] = torch.nn.functional.softmax(out_pi)
        return out_dict

    def draw_sample(self, outputs: dict, greedy:bool=False, greedy_mu:bool=True, temp:float =0.5):
        '''
        Obtains 2D strokes from mu, sigma, pis returned by GMM
        '''
        is_2d = True
        if outputs["mu"].dim() == 3:
            is_2d = False
        out_shape = outputs["mu"].size()
        batch_size = out_shape[0]
        seq_len = 1 if is_2d else out_shape[1]
        comp_shape = (batch_size, seq_len, self.out_units, self.num_components)
        #reshapes mu and sigma according to comp_shape
        pi = outputs["pi"]
        mu = outputs["mu"].reshape(comp_shape)
        sigma = outputs["sigma"].reshape(comp_shape)
        #permute variables (?, seq_len, out_units, num_components)
        mu = mu.permute(0, 1, 3, 2) 
        sigma = sigma.permute(0, 1, 3, 2)
        probs = pi.reshape(-1, self.num_components)
        #when greedy: select mus, and sigmas according to max probabilites
        if greedy:
            logits = torch.log1p(probs) 
            comp_indexes = logits.max(dim = 1, keepdim = True)[1]
        #when not greedy: selects mus, and sigmas according to a categorial distribution with probabilities
        else:
            probs_adjusted = adjust_temp(pi_pdf, temp)
            logits = torch.log1p(probs_adjusted)
            comp_indexes = torch.multinominal(logits, 1).reshape(-1, seq_len) #multinomial distribution is categorical
        #selects components of mu and sigma according to indexes selected
        component_mu = mu.index_select(dim = 3, index = comp_indexes[-1]).squeeze(dim = 3)
        component_sigma = sigma.index_select(dim = 3, index = comp_indexes[-1]).squeeze(dim = 3)
        #when greedy_mu component mu is the one calculated
        if greedy_mu:
            sample = component_mu
        #when not greedy_mu component mu is equal to a normal distribution with mean=component_mu and std=component_sigma*temp^2
        else:
            sample = torch.normal(mean = component_mu, std = component_sigma*(temp^2))
        if is_2d:
            return sample.squeeze() #output shape: (?, 2)
        else:
            return sample

    def draw_sample_every_component(self, outputs:dict, greedy:bool=False):
        """Draws a sample from every GMM component.
    
        Args:
        outputs: a dictionary containing mu, sigma and pi. mu and sigma are of
            shape (batch_size, seq_len, feature_size*n_components).
            pi is of shape (batch_size, seq_len, n_components).
        greedy: whether to return mu directly or sample.

        Returns:
        sample tensor - (batch_size, seq_len, n_components, feature_size)
        pi values - (batch_size, seq_len, n_components)
        """
        mu = outputs["mu"]
        sigma = outputs["sigma"]
        pi = outputs["pi"]

        out_shape = mu.shape
        seq_len = 1
        if mu.dim() == 3:
            seq_len = out_shape[1]
        batch_size = out_shape[0]
        comp_shape = (batch_size, seq_len, self.out_units, self.num_components)

        mu = outputs["mu"].reshape(comp_shape).permute(0, 1, 3, 2)
        sigma = outputs["sigma"].reshape(comp_shape).permute(0, 1, 3, 2)
        pi = outputs["pi"].reshape(batch_size, seq_len, self.num_components)

        if greedy:
            sample = mu
        else:
            sample = torch.normal(mu, sigma/4.0)
        return sample, pi

    def draw_sample_from_nth(self, outputs:dict, n: int, greedy:bool=False):
        """Draws a sample from the nth component.

        Args:
        outputs: a dictionary containing mu, sigma and pi. mu and sigma are of
            shape (batch_size, seq_len, feature_size*n_components).
            pi is of shape (batch_size, seq_len, n_components).
        n: component id.
        greedy: whether to return mu directly or sample.

        Returns:
        sample tensor - (batch_size, seq_len, feature_size)
        pi values - (batch_size, seq_len)
        """
        assert n < self.num_components

        mu = outputs["mu"]
        sigma = outputs["sigma"]
        pi = outputs["pi"]

        out_shape = mu.shape
        seq_len = 1
        if mu.dim() == 3:
            seq_len = out_shape[1]
        batch_size = out_shape[0]
        comp_shape = (batch_size, seq_len, self.out_units, self.num_components)

        mu = outputs["mu"].reshape(comp_shape)[:, :, :, n]
        sigma = outputs["sigma"].reshape(comp_shape)[:, :, :, n]
        pi = outputs["pi"].reshape(batch_size, seq_len, self.num_components)[:, :, n]

        if greedy:
            sample = mu
        else:
            sample = torch.normal(mu, sigma/4.0)
        return sample, pi

