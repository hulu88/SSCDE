import torch
import numpy as np

# x = torch.tensor([[1,8,1],[2,1,7]],dtype=torch.float32).cuda()
def Orthonorm(x, epsilon=1e-7):
    # x_2 = torch.mm(x.t(),x)
    # x_2 +=torch.eye(x.shape[1]) * epsilon
    # L = torch.cholesky(x_2)
    # ortho_weights = torch.inverse(L).t() * torch.sqrt(torch.tensor(x.shape[0], dtype=torch.float32))
    # print(ortho_weights.shape)
    #
    # ortho_weights = torch.inverse(L).t()
    # l =torch.mm(x, ortho_weights)
    # print(l)

    q,r = torch.qr(x)
    # print(torch.mm(q.t(),q))
    # print(ortho_weights)
    # exit()
    return q

def squared_distance(X, Y=None, W=None):
    if Y is None:
        Y = X
    # distance = squaredDistance(X, Y)
    dim =2
    X = torch.unsqueeze(X, dim=1)

    if W is not None:
        # if W provided, we normalize X and Y by W
        D_diag = torch.unsqueeze(torch.sqrt(torch.sum(W, dim=1)), dim=1)
        X /= D_diag
        Y /= D_diag
    distance = torch.sum(torch.pow(X-Y,2), dim=2)
    return distance





