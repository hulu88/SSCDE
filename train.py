from utils import load_data, DiffLoss
from model import *
from Metrics import eva
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
from orth import *

Learning_Rate = 0.005 #0.001
Epoch_Num =200
seed=1
sigma = 0.8
l1 = 0.005 #lossc
l2 = 1    #dec
l3 = 0.5  #loss1
l4 = 0.0001 #loss_or
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
#random.seed(seed)
torch.backends.cudnn.benckmark = False
torch.backends.cudnn.deterministic = True

s_data, s_y, s_n, t_data, t_y, t_n, s_adj, t_adj, name = load_data()
s_label = np.expand_dims(s_y,axis=1)
t_label = t_y
t_label = np.expand_dims(t_y,axis=1)



if s_n!=t_n:
    exit()

model = deepnet(s_data, t_data, s_n,s_adj, t_adj)
net = torch.load('tl_model.{}.pkl'.format(name))
model.load_state_dict(net)

cross_loss = torch.nn.CrossEntropyLoss()
klloss = torch.nn.KLDivLoss()
mse_loss = torch.nn.MSELoss()
dif_loss = DiffLoss()
s_y = torch.tensor(s_y, dtype=torch.int64)
t_y = torch.tensor(t_y, dtype=torch.int64)

s_m, t_m, s_class, t_label, loss2 = model(s_label)

SC = SpectralClustering(n_clusters=t_n)
Y_pred_OK = SC.fit_predict(t_m.detach().numpy())
Labels_K = np.array(t_y).flatten()
eva(Y_pred_OK, Labels_K)

model_main = net2(Y_pred_OK, t_data, t_adj, t_n)
optimzer = torch.optim.Adam(model_main.parameters(), lr=Learning_Rate)

Y_pred_OK = torch.tensor(Y_pred_OK, dtype=torch.int64)



for epoch in range(Epoch_Num):
    G, H, G_, H_, Z, X_dec, Y_cls, Y_SCNET = model_main()
    x = Orthonorm(Y_SCNET)
    Y = squared_distance(x)
    W = torch.exp(-squared_distance(Z) / (2 * (sigma* 2)))
    W = W.view(-1)
    B = Y.view(-1)
    lossc = torch.matmul(W, B) / (t_data.shape[0])*l1
    loss_dec = mse_loss(X_dec, t_data)*l2
    loss1 = cross_loss(Y_cls, Y_pred_OK)*l3
    loss_or = dif_loss(G_, H_)*l4
    loss = loss_dec + loss_or + loss1 + lossc
 
    print('                                     loss{: .4f}   || 交叉熵 {: .4f} | or {: .4f} | dec {: .4f} | C {: .4f}|'.format(loss, loss1, loss_or, loss_dec, lossc))
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    kmeans = KMeans(n_clusters=t_n)
    pred = kmeans.fit_predict(Z.detach().numpy())
    Labels = np.array(t_y).flatten()
    acc = eva(pred, Labels, epoch=epoch)
    Y_t = pred

    kmeans = KMeans(n_clusters=t_n)
    pred = kmeans.fit_predict(G.detach().numpy())
    Labels = np.array(t_y).flatten()
    eva(pred, Labels, epoch=epoch)


    kmeans = KMeans(n_clusters=t_n)
    pred = kmeans.fit_predict(H.detach().numpy())
    Labels = np.array(t_y).flatten()
    eva(pred, Labels, epoch=epoch)






