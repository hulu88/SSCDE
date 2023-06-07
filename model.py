import torch.nn.functional as F
from mmd import *

def get_weight_initial(d1, d2):
    bound = torch.sqrt(torch.Tensor([6.0 / (d1 + d2)]))
    nor_W = -bound + 2*bound*torch.rand(d1, d2)
    return torch.Tensor(nor_W)

class myGAE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2):
        super(myGAE, self).__init__()

        self.gconv1 = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1),
            torch.nn.ReLU(inplace=True)
        )
        self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)

        self.gconv2 = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2),
            # torch.nn.Dropout(0.5)
        )
        self.gconv2[0].weight.data = get_weight_initial(d_2, d_1)

    def Encoder(self, Adjacency_Modified, H_0):
        H_1 = self.gconv1(torch.matmul(Adjacency_Modified, H_0))
        H_2 = self.gconv2(torch.matmul(Adjacency_Modified, H_1))
        return H_2


    def forward(self, Adjacency_Modified, H_0):
        Latent_Representation = self.Encoder(Adjacency_Modified, H_0)
        return Latent_Representation

class deepnet(nn.Module):
    def __init__(self, s_data, t_data, s_n, s_adj, t_adj ):
        super(deepnet, self).__init__()
        self.s_data = s_data
        self.t_data = t_data
        self.s_adj = s_adj
        self.t_adj = t_adj
        self.encoder_s = nn.Sequential(nn.Linear(s_data.shape[1], 500),nn.ReLU(inplace=True),nn.Linear(500, 64))
        self.encoder_t = nn.Sequential(nn.Linear(t_data.shape[1], 500), nn.ReLU(inplace=True), nn.Linear(500, 64))
        self.cls_model = nn.Sequential(nn.Linear(64, s_n))
        self.mmd = MMD_loss()

    def forward(self, s_label):
        s_m = self.encoder_s(self.s_data)
        t_m = self.encoder_t(self.t_data)
        s_class = self.cls_model(s_m)
        t_label = self.cls_model(t_m)

        loss_m = self.mmd.marginal(s_m, t_m)
        loss_c = self.mmd.conditional(s_m, t_m, s_label, F.softmax(t_label,dim=1))
        # mu = 0*dynamic_factor.estimate_mu(s_m.detach().numpy(),\
        #                                 s_label,\
        #                                 t_m.detach().numpy(), \
        #                                 torch.max(t_label, 1)[1].detach().numpy())
        print(loss_m,loss_c)
        loss = loss_c + 0.5*loss_m

        # exit()
        return s_m, t_m, s_class, t_label, loss

class net2(nn.Module):
    def __init__(self,  y_pros, t_data, t_adj,t_n):
        super(net2, self).__init__()
        self.t_data = t_data
        self.t_adj = t_adj
        self.y_pros = y_pros
        self.encoder_2 = myGAE(t_data.shape[1], 256, 32)
        self.encoder_se = torch.nn.Sequential(nn.Linear(t_data.shape[1], 512), nn.ReLU(inplace=True), nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 32))
        self.decoder_2 = torch.nn.Sequential(nn.Linear(32, 512), nn.ReLU(inplace=True), nn.Linear(512, t_data.shape[1]))
        self.encoder_pro = torch.nn.Sequential(nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Linear(16, t_n))
        self.scnet = torch.nn.Sequential(nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Linear(16, t_n))
        self.org = torch.nn.Sequential(nn.Linear(32, 32), nn.Softmax(dim=1))
        self.orh = torch.nn.Sequential(nn.Linear(32, 32), nn.Softmax(dim=1))

    def forward(self):
        G = self.encoder_2(self.t_adj, self.t_data)
        H = self.encoder_se(self.t_data)
        Z = F.softmax(G, dim=1) + F.softmax(H, dim=1)
        G_ = self.org(G)
        H_ = self.orh(H)
        Y_cls = self.encoder_pro(Z)
        X_dec = self.decoder_2(Z)
        Y_SCNET = self.scnet(Z)
        return G, H, G_, H_, Z, X_dec, Y_cls, Y_SCNET