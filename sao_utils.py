# Generating SAO matrices
import torch
from scipy.linalg import orth
import random
import numpy as np
from itertools import product

def sao_2(mask):
    c = int(torch.sum(mask, 0)[0])
    d = int(torch.sum(mask, 1)[0])
    degree = c if c > d else d
    sao_matrix = torch.zeros(mask.shape).to('cuda') 
    num_ortho = int(degree*mask.shape[0]/mask.shape[1])

    _, inv, counts = torch.unique(mask, dim=0, return_inverse=True, return_counts=True)
    row_index = [tuple(torch.where(inv == i)[0].tolist()) for i, c, in enumerate(counts) if counts[i] > 1]
    
    if num_ortho == 1:
        to_iterate = inv.reshape(inv.shape[0], 1)
    else:
        to_iterate = row_index

    for i in to_iterate:

        indices = torch.tensor(i).to('cuda') 
        identical_row = mask[indices]

        M = np.random.rand(degree, degree) 
        O = torch.tensor(orth(M), dtype=torch.float).T

        for j in range(identical_row.shape[0]):
                nonzeros = torch.nonzero(identical_row[j])
                identical_row[j, nonzeros] = O[j].reshape(O.shape[1], 1).to('cuda') 

        sao_matrix[indices] = identical_row
        
    return sao_matrix.to('cuda')

def d_regular_copy(n_rows, n_columns, degree):
    ''' This forms the d-regular mask {0, 1} of the weight matrix.
    When n_rows > n_columns (output > input, upscaling), the column degree is greater than the specified degree.

    '''
    
    m = int(n_columns/degree)
    n = n_columns
    

    row_deg = (1/2)*torch.ones(n) #actually degree of each column
    degree_tensor = torch.ones(n)

    while not torch.equal(row_deg, degree_tensor):
            weight = torch.zeros(m, n)
            columns = list(range(n)) 

            for _ in range(2):
                for i in range(weight.shape[0]):   
                    index = random.choice(columns)
                    while torch.sum(weight[:, index]) == 1:
                        index = random.choice(columns)
                    weight[i, index] = 1
                
            for i, j in product(range(m), range(n)):
                if weight[i, j] != 1 and torch.sum(weight[i]) < degree and torch.sum(weight[:, j]) != 1:
                    weight[i, j] = 1

                row_deg = torch.sum(weight, 0)
            
    weight_copied = torch.zeros(n_rows, n_columns)   
    a = int((n_rows*degree)/n_columns)
    
    for i in range(a):
        weight_copied[m*i:m*i + m] = weight.reshape(m, n)
                
    return torch.tensor(weight_copied).reshape(n_rows, n_columns).to('cuda')