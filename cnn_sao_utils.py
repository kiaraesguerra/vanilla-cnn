from sao_utils import *
import torch.nn as nn
import math
from sao_utils import *


def ortho_conv(module, degree=None):
    k = module.kernel_size[0]
    in_ch = module.in_channels
    out_ch = module.out_channels

    List1 = []
    List2 = []

    for i, j in product(range(k), range(k)):
        eqi, eqj = give_equiv(k, i, j)
        List1.append([i, j])
        List2.append([eqi, eqj])

    for i in List1:
        index1 = List1.index(i)
        index2 = List2.index(i)

        if index1 > index2:
            List1[index1] = -1

    List1 = [x for x in List1 if x != -1] #use this list to assign the unique ortho matrix
    List2 = [x for x in List2 if x not in List1] #use this list to get the equiv index then copy the originally assigned ortho matrices

    ortho_tensor = ortho_gen(in_ch, out_ch, k, degree=degree)
    A = torch.zeros(k, k, out_ch, in_ch)

    for i in range(len(List1)):
        p, q = List1[i]
        A[p, q] = ortho_tensor[i]

    for i in range(len(List2)):
        p, q = List2[i]
        equivi, equivj = give_equiv(k, p, q)
        A[p, q] = A[equivi, equivj]

    weight_mat = torch.zeros(out_ch,in_ch,k, k)

    for i, j in product(range(out_ch), range(in_ch)):
        weight_mat[i, j] = torch.fft.ifft2(A[:, :, i, j])

    return weight_mat.to('cuda')

def ortho_gen(in_ch, out_ch, k, degree=None):
    L = (k**2 + 1)//2
    ortho_tensor = torch.zeros(L, out_ch, in_ch)
    if degree != None:
        if out_ch > in_ch:
            mask = d_regular_copy(in_ch, out_ch, degree)
        else:
            mask = d_regular_copy(out_ch, in_ch, degree)
            
    for i in range(L):
        if degree == None:
            if in_ch > out_ch:
                ortho_tensor[i] = torch.tensor(orth(np.random.randn(in_ch, out_ch))).T
            else:
                ortho_tensor[i] = torch.tensor(orth(np.random.randn(out_ch, in_ch)))    
        else:
            if out_ch > in_ch:     
                ortho_tensor[i] = sao_2(mask).T      
            else:
                ortho_tensor[i] = sao_2(mask)
                
    return ortho_tensor.to('cuda')


def give_equiv(kernel_size, i, j):
    k = kernel_size
    i_equiv = (k-i)%k
    j_equiv = (k-j)%k
    return i_equiv, j_equiv

def orthogonal_dense(model, gain):
  for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.weight = nn.Parameter(ortho_conv(module, degree=None)*gain)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain)

  return model.to('cuda')

def mask_structured(module, degree):
  in_ch = module.in_channels
  out_ch = module.out_channels

  if out_ch > in_ch:
    mask = d_regular_copy(in_ch, out_ch, degree).T
  else:
    mask = d_regular_copy(out_ch, in_ch, degree)
  
  mask_conv = torch.zeros(module.weight.shape).to('cuda')
  left = range(out_ch)
  right = range(in_ch)
  with torch.no_grad():
      for i,j in product(left, right):
          mask_conv[i,j] = mask[i,j]

  return mask_conv


def prune_vanilla_kernelwise(param, sparsity, fn_importance=lambda x: x.norm(1, -1)):
    assert param.dim() >= 3
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        return torch.zeros_like(param).byte()
    num_kernels = param.size(0) * param.size(1)
    param_k = param.view(num_kernels, -1)
    param_importance = fn_importance(param_k)
    num_pruned = int(math.ceil(num_kernels * sparsity))
    _, topk_indices = torch.topk(param_importance, k=num_pruned,
                                 dim=0, largest=False, sorted=False)
    mask = torch.zeros_like(param).byte()
    mask_k = mask.view(num_kernels, -1)
    param_k.index_fill_(0, topk_indices, 0)
    mask_k.index_fill_(0, topk_indices, 1)
    return mask