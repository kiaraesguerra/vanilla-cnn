from cnn_sao_delta_utils import *
from cnn_sao_utils import *


def custom_pruner_degree(model, pruning_method, degree):
    if pruning_method == 'SAO':
        model = Delta_SAO_same_deg(model, degree, 1)
    elif pruning_method == 'SAO-relu':
        model = Delta_ReLU_SAO_same_deg(model, degree, 1)
    elif pruning_method == 'LMP-S':
        model = lmp_same_deg(model, degree)
    elif pruning_method == 'RG-S':
        model = d_regular_pruning_same_deg(model, degree)      
    return model

def custom_pruner(model, pruning_method, sparsity):
    if pruning_method == 'SAO':
        model = Delta_SAO(model, sparsity, 1)
    elif pruning_method == 'SAO-relu':
        model = Delta_ReLU_SAO(model, sparsity, 1)
    elif pruning_method == 'LMP-S':
        model = lmp(model, sparsity)
    elif pruning_method == 'RG-S':
        model = d_regular_pruning(model, sparsity)      
    return model

def remove_parameters(model):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model



