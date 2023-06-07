from model import *
from Metrics import eva
from utils import *
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')


s_data, s_y, s_n, t_data, t_y, t_n, s_adj, t_adj, name = load_data()
s_label = np.expand_dims(s_y,axis=1)
t_label = t_y
t_label = np.expand_dims(t_y,axis=1)

model = deepnet(s_data, t_data, s_n,s_adj, t_adj)
net = torch.load('tl_model.{}.pkl'.format(name))
model.load_state_dict(net)

s_y = torch.tensor(s_y, dtype=torch.int64)
t_y = torch.tensor(t_y, dtype=torch.int64)
s_m, t_m, s_class, t_label, loss2 = model(s_label)

SC = SpectralClustering(n_clusters=t_n)
Y_pred_OK = SC.fit_predict(t_m.detach().numpy())
Labels_K = np.array(t_y).flatten()
print('源域的性能为：')
eva(Y_pred_OK, Labels_K)

#-------------------------------------------------------------
model_main = net2(Y_pred_OK, t_data, t_adj, t_n)
net2 = torch.load('model2_ing.{}.pkl'.format(name))
model_main.load_state_dict(net2)
G, H, G_, H_, Z, X_dec, Y_cls, Y_SCNET = model_main()

kmeans = KMeans(n_clusters=t_n)
Y_pred_OK = kmeans.fit_predict(Z.detach().numpy())
Labels_K = np.array(t_y).flatten()
print('目标域的性能为：')
eva(Y_pred_OK, Labels_K)