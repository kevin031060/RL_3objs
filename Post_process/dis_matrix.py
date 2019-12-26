import numpy as np
import torch

def dis_matrix(static, s_size, obj3_if=False):
    static = static.squeeze(0)

    # [2,20]
    obj1 = static[:2, :]
    obj2 = static[2, :]
    obj3 = static[3, :]

    l = obj1.size()[1]
    obj1_matrix = np.zeros((l, l))
    obj2_matrix = np.zeros((l, l))
    obj3_matrix = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            if i != j:
                obj1_matrix[i,j] = torch.sqrt(torch.sum(torch.pow(obj1[:, i] - obj1[:, j], 2))).detach()
                obj2_matrix[i, j] = torch.abs(obj2[i] - obj2[j]).detach()
                obj3_matrix[i, j] = torch.abs(obj3[i] - obj3[j]).detach()


    if obj3_if:
        return obj1_matrix, obj2_matrix, obj3_matrix
    else:
        return obj1_matrix, obj2_matrix