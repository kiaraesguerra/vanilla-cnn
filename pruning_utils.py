from cnn_sao_delta_utils import *
from cnn_sao_utils import *


def custom_pruner_degree(model, args):
    if pruning_method == 'SAO':
        model = Delta_SAO_same_deg(model, args.degree, args.gain)
    elif pruning_method == 'SAO-relu':
        model = Delta_ReLU_SAO_same_deg(model,  args.degree, args.gain)
    elif pruning_method == 'LMP-S':
        model = lmp_same_deg(model,  args.degree)
    elif pruning_method == 'RG-S':
        model = d_regular_pruning_same_deg(model,  args.degree)      
    return model

def custom_pruner(model, args):
    if pruning_method == 'SAO':
        model = Delta_SAO(model, args.sparsity, args.gain)
    elif pruning_method == 'SAO-relu':
        model = Delta_ReLU_SAO(model, args.sparsity, args.gain)
    elif pruning_method == 'LMP-S':
        model = lmp(model, args.sparsity)
    elif pruning_method == 'RG-S':
        model = d_regular_pruning(model, args.sparsity)      
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



