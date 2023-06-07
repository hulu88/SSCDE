from __future__ import division
from __future__ import print_function
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.sparse as sp
import torch
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data():
    DATA_ASD = sio.loadmat('ASD_connectivity.mat')
    DATA= DATA_ASD['connectivities']
    ASD_label = sio.loadmat('ASD_labels.mat')
    ASD_label = ASD_label['VarName1']
    label = np.squeeze(ASD_label)
    k = 19+1
    n = 3
    name = "ASD"

    adj = c_adj(DATA, k)
    s_data = DATA
    s_y = label
    s_n = n
    s_adj =adj.todense()
    s_name = name

    data = sio.loadmat('ABIDE-MS.mat') # 988x1357  3
    k = 22+1
    n = 3
    name = "ADIBE-MS"
    data = data['data']
    label = data[:, 0]
    DATA = data[:, 1:]
    adj = c_adj(DATA, k)

    t_data = DATA
    t_y = label
    t_n = n
    t_adj = adj.todense()
    t_name = name

    print("源域的数据集为：",s_name,   "目标域的数据集为",t_name)
    s_data = torch.tensor(s_data, dtype=torch.float32)
    t_data = torch.tensor(t_data, dtype=torch.float32)
    s_adj = torch.Tensor(s_adj)
    t_adj = torch.Tensor(t_adj)

    s_y = s_y - np.min(s_y)
    t_y = t_y - np.min(t_y)
    return s_data, s_y, s_n, t_data, t_y, t_n, s_adj, t_adj, t_name

def c_adj(data,k):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    # features = data.detach().cpu().numpy()
    features = data
    standardizer = StandardScaler()
    features_standardized = standardizer.fit_transform(features)
    nearestneighbors_euclidean = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(features_standardized)
    adj= nearestneighbors_euclidean.kneighbors_graph(features_standardized).toarray()
    from scipy import sparse
    adj = sparse.csr_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize(adj):
    """Symmetrically normalize adjacency matrix."""

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def plot_embeddings(embeddings, Features, Labels):

    emb_list = []
    for k in range(Features.shape[0]):
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2, init="pca")
    # model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)
    color_idx = {}
    for i in range(Features.shape[0]):
        color_idx.setdefault(Labels[i][0], [])
        color_idx[Labels[i][0]].append(i)
    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s=6)
    plt.axis('off')
    # plt.legend()
    plt.gca.legend_ = None
    # plt.savefig(name)
    plt.show()

class DiffLoss(torch.nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss
