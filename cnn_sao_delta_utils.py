import torch.nn as nn
import torch.nn.utils.prune as prune
from cnn_sao_utils import *
from sao_utils import *
from pruning_methods import *

def delta_ortho_dense(module):
    ortho_tensor = torch.tensor(orth(np.random.randn(max(module.in_channels, module.out_channels), 
                                                     min(module.in_channels, module.out_channels)))).to('cuda')
    ortho_tensor = ortho_tensor.T if module.in_channels > module.out_channels else ortho_tensor
    ortho_c = torch.zeros(module.weight.shape).to('cuda')
    with torch.no_grad():
        ortho_c[:, :, 1, 1] = ortho_tensor

    return ortho_c.to('cuda')

def Delta_Ortho_Dense(model, gain):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain)
        elif isinstance(module, torch.nn.Conv2d):
            if module.kernel_size[0] != 1:
                module.weight = nn.Parameter(delta_ortho_dense(module)*gain)      
    return model.to('cuda')
    
    
def delta_ortho_sao(module, degree):
    in_ch = module.in_channels
    out_ch = module.out_channels
    
    if out_ch > in_ch:
        mask = d_regular_copy(in_ch, out_ch, degree)
        sao_matrix = sao_2(mask).T
        mask = mask.T
    else:
        mask = d_regular_copy(out_ch, in_ch, degree)
        sao_matrix = sao_2(mask)

    ortho_c = torch.zeros(module.weight.shape).to('cuda')
    conv_mask = torch.zeros(module.weight.shape).to('cuda')
    ortho_c[:,:, 1, 1] = sao_matrix
    for i,j in product(range(out_ch), range(in_ch)):
        conv_mask[i, j] = mask[i, j]
    return ortho_c, conv_mask


