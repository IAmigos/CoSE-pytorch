import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputModelGMMDense(nn.Module):

    def __init__(self,
                input_size,
                out_units, # dim of mu and sigma
                num_components,
                # sigma_activation=,
                # kernel_regularizer=None,
                # bias_regularizer=None,
                # prefix=""
                )
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

    def draw_sample(self, outputs, greedy=False, greedy_mu, temp=0.5):
        def adjust_temp(pi_pdf, temp):
            pi_pdf = torch.log(pi_pdf)/temp
            # pi_pdf -= tf.reduce_max(pi_pdf, axis=-1)
            # pi_pdf = tf.math.exp(pi_pdf)
            # pi_pdf /= tf.reduce_sum(pi_pdf, axis=-1)
            return pi_pdf

        is_2d = True
        if outputs["mu"].dim() == 3:
            is_2d = False
        out_shape = outputs["mu"].shape()
        batch_size = out_shape[0]
        seq_len = 1 if is_2d else out_shape[1]
        comp_shape = (batch_size, seq_len, self.out_units, self.num_components)

        pi = outputs["pi"]
        mu = outputs["mu"].shape(comp_shape)
        sigma = outputs["sigma"].shape(comp_shape)

        # mu = tf.transpose(a=mu, perm=[0, 1, 3, 2])
        # sigma = tf.transpose(a=sigma, perm=[0, 1, 3, 2])

        # probs = tf.reshape(pi, (-1, self.num_components))

