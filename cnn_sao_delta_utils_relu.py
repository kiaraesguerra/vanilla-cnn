from itertools import product
import torch
import numpy as np
from scipy.linalg import orth
from cnn_sao_utils import *
from sao_utils import *
from pruning_methods import *

def generate_orthogonal_matrix(height, width):
    rand_matrix = torch.randn((height, width))
    q, _ = torch.qr(rand_matrix)
    orthogonal_matrix = q[:, :width]
    return orthogonal_matrix

def Gen_W0(W_0):
    return torch.concat([torch.concat([W_0, torch.negative(W_0)], axis=0), torch.concat([torch.negative(W_0), W_0], axis=0)],
                        axis=1)


def delta_ortho_dense_relu(module):
    out_ch = module.out_channels // 2
    in_ch = module.in_channels // 2
    ortho_c = torch.zeros(module.weight.shape).to('cuda')

    if module.in_channels == 3:
        ortho_tensor = torch.tensor(orth(np.random.randn(module.out_channels, module.in_channels)))
    else:
        if in_ch > out_ch:
            ortho_tensor = Gen_W0(torch.tensor(orth(np.random.randn(in_ch, out_ch))).T)
        else:
            ortho_tensor = Gen_W0(torch.tensor(orth(np.random.randn(out_ch, in_ch))))
            
    left = range(out_ch)
    right = range(in_ch)
    
    with torch.no_grad():
        for i,j in product(left, right):
            ortho_c[i,j, 1, 1] = ortho_tensor[i,j]

    return ortho_c

    
def delta_ortho_sao_relu(module, degree):
    
    out_ch = module.out_channels // 2
    in_ch = module.in_channels // 2
    deg = degree // 2
    
    if out_ch > in_ch:
        mask = d_regular_copy(in_ch, out_ch, degree=deg)
        sao_matrix = Gen_W0(sao_2(mask).T)
        mask = torch.abs(Gen_W0(mask.T))
    else:
        mask = d_regular_copy(out_ch, in_ch, degree=deg)
        sao_matrix = Gen_W0(sao_2(mask))
        mask =  torch.abs(Gen_W0(mask))

    ortho_conv = torch.zeros(module.weight.shape).to('cuda')
    conv_mask = torch.zeros(module.weight.shape).to('cuda')
    
    left = range(module.out_channels)
    right = range(module.in_channels)

    with torch.no_grad():
        for i,j in product(left, right):
            ortho_conv[i,j, 1, 1] = sao_matrix[i,j]
            conv_mask[i, j] = mask[i, j]

    return ortho_conv, conv_mask


