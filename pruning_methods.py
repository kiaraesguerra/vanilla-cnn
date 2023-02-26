import torch.nn as nn
import torch
import torch.nn.utils.prune as prune
from cnn_sao_utils import *
from sao_utils import *
from cnn_sao_delta_utils import *
from cnn_sao_delta_utils_relu import *
def degree_from_sparsity(sparsity, module):
    larger_dim = max(module.in_channels, module.out_channels)
    return int((1-sparsity)*larger_dim)

def sparsity_from_degree(degree, module):
    larger_dim = max(module.in_channels, module.out_channels)
    return (1-degree/larger_dim)



def lmp(model, sparsity):        
    for _, module in model.named_modules():            
        if isinstance(module, torch.nn.Conv2d):
            if module.in_channels != 3 and module.kernel_size[0] !=1:    
                with torch.no_grad():
                    mask = (prune_vanilla_kernelwise(module.weight, sparsity) == 0)*1
                    prune.custom_from_mask(module, name='weight', mask=mask)
    return model    


def d_regular_pruning(model, sparsity):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels != 3:
            degree = degree_from_sparsity(sparsity, module)
            if module.out_channels > module.in_channels:
                prune.custom_from_mask(module, name='weight', mask=mask_structured(module, degree*2))
            else:
                prune.custom_from_mask(module, name='weight', mask=mask_structured(module, degree))
    return model


def Delta_SAO(model, sparsity, gain):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain)
        elif isinstance(module, torch.nn.Conv2d) and module.kernel_size[0] != 1:    
            if module.in_channels == 3:
                module.weight = nn.Parameter(delta_ortho_dense(module)*gain)
            else:
                degree = degree_from_sparsity(sparsity, module)
                weight, mask = delta_ortho_sao(module, degree=degree)
                module.weight = nn.Parameter(weight*gain)
                with torch.no_grad(): 
                    prune.custom_from_mask(module, name='weight', mask=mask)
            
    return model.to('cuda')


def Delta_ReLU_SAO(model, sparsity, gain):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain)
        elif isinstance(module, torch.nn.Conv2d) and module.kernel_size[0] != 1:
            if module.in_channels == 3:
                module.weight = nn.Parameter(delta_ortho_dense(module)*gain)
            else:
                degree = degree_from_sparsity(sparsity, module)
                weight, mask = delta_ortho_sao_relu(module, degree=degree)
                module.weight = nn.Parameter(weight*gain)        
                with torch.no_grad(): 
                    prune.custom_from_mask(module, name='weight', mask=mask)
    return model.to('cuda')
    

def d_regular_pruning_same_deg(model, degree):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.in_channels != 3 and module.kernel_size[0] !=1:
                if module.out_channels > module.in_channels:
                    prune.custom_from_mask(module, name='weight', mask=mask_structured(module, degree*2))
                else:
                    prune.custom_from_mask(module, name='weight', mask=mask_structured(module, degree))

    return model
    

def lmp_same_deg(model, degree):        
    for _, module in model.named_modules():            
        if isinstance(module, torch.nn.Conv2d):
            if module.in_channels != 3 and module.kernel_size[0] !=1:
                if module.out_channels > module.in_channels:
                    deg = degree * 2
                else:
                    deg = degree
                sparsity = sparsity_from_degree(deg, module)         
                with torch.no_grad():
                    mask = (prune_vanilla_kernelwise(module.weight, sparsity) == 0)*1
                    prune.custom_from_mask(module, name='weight', mask=mask)
    return model


def Delta_SAO_same_deg(model, degree, gain):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain)
        elif isinstance(module, torch.nn.Conv2d) and module.kernel_size[0] != 1:
            if module.in_channels == 3:
                module.weight = nn.Parameter(delta_ortho_dense(module)*gain)
            else:
                if module.out_channels > module.in_channels:
                    weight, mask = delta_ortho_sao(module, degree=degree*2)
                else:
                    weight, mask = delta_ortho_sao(module, degree=degree)
                module.weight = nn.Parameter(weight*gain)
                with torch.no_grad(): 
                    prune.custom_from_mask(module, name='weight', mask=mask)
    return model.to('cuda')    


def Delta_ReLU_SAO_same_deg(model, degree, gain):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain)
        elif isinstance(module, torch.nn.Conv2d) and module.kernel_size[0] != 1:
            if module.in_channels == 3:
                module.weight = nn.Parameter(delta_ortho_dense(module)*gain)
            else:
                if module.out_channels > module.in_channels:
                    weight, mask = delta_ortho_sao_relu(module, degree=degree*2)
                else:
                    weight, mask = delta_ortho_sao_relu(module, degree=degree)
                module.weight = nn.Parameter(weight*gain)        
                with torch.no_grad(): 
                    prune.custom_from_mask(module, name='weight', mask=mask)
    return model.to('cuda')


    

