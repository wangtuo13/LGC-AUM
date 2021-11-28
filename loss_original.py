import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class lgc_loss(nn.Module):
    def __init__(self, change=30, n_cluster=10, hidden_dim=10, eps=0.00001, total=2000, dim=784):
        super(lgc_loss, self).__init__()
        #self.in_dim = dim[0]
        #self.nlayer = len(dim) - 1
        #self.layers = dim
        #self.numpen = numpen
        #self.encoder = self.build_net(growthRate, numpen)
        self.total = total
        self.mu = Parameter(torch.Tensor(n_cluster, hidden_dim))
        # self.mu = torch.Tensor(nClasses, dim[-1])
        self.num_class = n_cluster
        self.relu = nn.ReLU()
        #self.mseloss = nn.MSELoss()
        self.eps = eps
        self.alpha = 1
        #self.view1 = nn.Linear(dim[-1], 100)
        #self.view2 = nn.Linear(100, 2)
        self.sigmas = torch.zeros(total, 1).to(device)
        self.rhos = torch.zeros(total, 1).to(device)
        self.sigmas_x_u = torch.zeros(total, 1)
        self.rhos_x_u = torch.zeros(total, 1)
        self.start = torch.zeros(total, 1)
        self.sigmas_u_x = torch.zeros(1, 10)
        self.rhos_u_x = torch.zeros(1, 10)
        self.mu_sigma = 1
        self.mu_rho = 1
        self.indices_x = 1
        self.indices_z = 1
        self.change = change

    def loss(self, q1, q2, D1, D2, one=True, pur=False):
        if one:
            if pur:
                return torch.mean(torch.sum(D1 * torch.log(1 / q1), dim=1))
            else:
                return torch.mean(
                    torch.sum(D1 * torch.log(1 / q1) + (1 - D2 + self.eps) * torch.log(1 / (1 - q2 + self.eps)), dim=1))
        else:
            if pur:
                return torch.mean(torch.sum(D1 * torch.log((D1 + self.eps) / (q1 + self.eps)), dim=1))
            else:
                return torch.mean(torch.sum(
                    D1 * torch.log((D1 + self.eps) / (q1 + self.eps)) + (1 - D2 + self.eps) * torch.log(
                        (1 - D2 + self.eps) / (1 - q2 + self.eps)), dim=1))

    def compute_S_(self, z, epoch):
        # Affinity_Z = torch.sum((z.unsqueeze(1) - z + self.eps) ** 2, dim=2).sqrt()
        # d_ = torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2)#######################################.sqrt()
        d_ = torch.sum((z.unsqueeze(1) - self.mu + self.eps) ** 2, dim=2).sqrt()
        # d_1 = torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2).sqrt()
        # d_ = torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2).sqrt() / (torch.sum(z ** 2, dim=1).sqrt().unsqueeze(1) * torch.sum(self.mu ** 2, dim=1).sqrt().unsqueeze(0))
        # d_ = d_1 * d_2

        k = 5
        m = 1

        Relation_z = d_.topk(k=k, dim=1, largest=False)
        Relation_u = d_.t().topk(k=k, dim=1, largest=False)

        if epoch >= 0:  ## very important caculate the proper rho and sigma for new mu
            rho_u = Relation_u[0][:, 0].unsqueeze(0)
            if epoch >= 0:
                rho_z = Relation_z[0][:, 0].unsqueeze(1)
                self.rhos_x_u = rho_z
            else:
                rho_z = self.rhos
                self.rhos_x_u = rho_z
            self.rhos_u_x = rho_u

            sample_size = z.size(0)
            lo = torch.zeros(sample_size, 1)
            hi = 10000 * torch.ones(sample_size, 1)
            mid_z = torch.cat([lo, hi], dim=1).cuda()
            sigma_z = torch.ones(sample_size, 1).cuda()

            mu_size = self.mu.size(0)
            lo = torch.zeros(mu_size, 1)
            hi = 10000 * torch.ones(mu_size, 1)
            mid_u = torch.cat([lo, hi], dim=1).cuda()
            sigma_u = torch.ones(1, mu_size).cuda()

            target = 1.5  # np.log2(k) * 1
            for i in range(k * 30):
                current_u = torch.sum((-(self.relu(Relation_u[0].t() - rho_u) / sigma_u)).exp(), dim=0)

                mid_u[range(mu_size), (current_u > target) * 1] = sigma_u[0, :]

                #temp_mid_u = mid_u.detach().cpu().numpy()
                #temp_sigma_u = sigma_u.detach().cpu().numpy()
                #temp_mid_u[range(mu_size), (current_u > target).cpu().numpy() * 1] = temp_sigma_u[0, :]
                #mid_u = torch.Tensor(temp_mid_u).to(device)

                sigma_u = torch.mean(mid_u, dim=1).unsqueeze(0)

                current_z = torch.sum((-(self.relu(Relation_z[0] - rho_z) / sigma_z)).exp(), dim=1)
                # current_z = torch.sum((-(self.relu(Relation_z[0] - rho_z) / sigma_z)).exp(), dim=1)
                mid_z[range(sample_size), (current_z > target) * 1] = sigma_z[:, 0]
                sigma_z = torch.mean(mid_z, dim=1).unsqueeze(1)
            self.sigmas_x_u = sigma_z
            self.sigmas_u_x = sigma_u
        else:
            rho_z = self.rhos_x_u  # _x_u
            rho_u = self.rhos_u_x
            sigma_z = self.sigmas_x_u
            sigma_u = self.sigmas_u_x
        # else:
        #    rho_x = self.rhos[batch_idx * sample_size: min((batch_idx + 1) * sample_size, num)]
        #    sigma_x = self.sigmas[batch_idx * sample_size: min((batch_idx + 1) * sample_size, num)]

        if epoch >= 0:
            rho_z = Relation_z[0][:, 0].unsqueeze(1)
            # self.rhos_x_u = rho_z

        '''
        # nearest_  = Affinity_Z.topk(k=10, dim=1, largest=False)  # Nearest k samples from Z_i
        near_mu_ = d_.topk(k=10, dim=0, largest=False)  # Nearest k samples from U_j
        Z_near_mu = d_.topk(k=10, dim=1, largest=False)  # Nearest k centers from Z_i



        # rho_ = nearest_[0][:, 1]
        rho_mu_ = near_mu_[0][0, :]
        rho_mu_z = Z_near_mu[0][:, 0]
        rho_mu_ = rho_mu_.detach()
        rho_mu_z = rho_mu_z.detach()

        # theta_ = nearest_[0][:, 3]
        theta_mu_ = near_mu_[0][5, :]  # k + 1
        theta_mu_z = Z_near_mu[0][:, 1]

        # W1_ = (- self.relu(d_ - rho_.unsqueeze(1))     / theta_.unsqueeze(1)).exp()
        # W2_ = (-          (d_ - rho_mu_.unsqueeze(0))  / theta_mu_.unsqueeze(0)).exp()
        # W3_ = (-          (d_ - rho_mu_z.unsqueeze(1)) / theta_mu_z.unsqueeze(1)).exp()
        # W2_ = (          theta_mu_.unsqueeze(0) / (d_ - rho_mu_.unsqueeze(0) + 0.1))
        # W3_ = (          theta_mu_z.unsqueeze(1) / (d_ - rho_mu_z.unsqueeze(1) + theta_mu_z.unsqueeze(1)))
        # W3_ = (theta_mu_z.unsqueeze(1) / (d_ - rho_mu_z.unsqueeze(1) + 1))
        # W2_ = (theta_mu_.unsqueeze(0)  -  (d_ - rho_mu_.unsqueeze(0))).exp()
        W2_ = (theta_mu_.unsqueeze(0) - (d_ - rho_mu_.unsqueeze(0))).exp()
        #W3_ = (theta_mu_z.unsqueeze(1) - (d_ - rho_mu_z.unsqueeze(1))).exp()  # useful
        W3_ = ( - (d_ - rho_mu_z.unsqueeze(1))).exp()  # useful
        # W3_ = (-          (d_ - rho_mu_z.unsqueeze(1)) / (torch.log(theta_mu_z.unsqueeze(1) + 1))).exp()
        # W3_ = (-          (d_)).exp()

        # S = W1_ + W2_ - W1_ * W2_

        # W1_ = W1_ / torch.sum(W1_, dim=1, keepdim=True)
        W2_ = W2_ / torch.sum(W2_, dim=1, keepdim=True)
        W3_ = W3_ / torch.sum(W3_, dim=1, keepdim=True)

        # D1_ = W1_ ** 2 / torch.sum(W1_, dim=0, keepdim=True)
        # D1 = D1_ / torch.sum(D1_, dim=1, keepdim=True)
        # D1 = W1_ / torch.sum(W1_, dim=1, keepdim=True)

        D2_ = W2_ ** 2 / torch.sum(W2_, dim=0)
        D2 = D2_ / torch.sum(D2_, dim=1, keepdim=True)
        # D2 = W2_ / torch.sum(W2_, dim=1, keepdim=True)

        D3 = W3_ ** 2 / torch.sum(W3_, dim=0)  ############################################################## keepdim=True
        D3 = D3 / torch.sum(D3, dim=1, keepdim=True)
        # D3 = W3_ / torch.sum(W3_, dim=1, keepdim=True)

        # S = S / torch.sum(S, dim=1, keepdim=True)
        '''

        rho_z = rho_z.detach()
        rho_u = rho_u.detach()

        W1 = (- self.relu(d_ - rho_z) / sigma_z).exp()
        # S = W1  # + W2 - W1 * W2
        # S = (- self.relu(d_ - rho_z) / sigma_z).exp()
        # S = S / torch.sum(S, dim=1, keepdim=True)
        # W1 = (- (d_ - rho_z) / sigma_z).exp()
        # W1 = (sigma_z / (self.relu(d_ - rho_z) + self.eps))
        # W1 = 1 / (self.relu(d_ - rho_z) / sigma_z + 1)
        # W1 = 1 / (1 + self.relu(d_ - rho_z) ** 2 / sigma_z)
        # W1 = 1 / (1 + ((d_ - rho_z) / sigma_z) ** 2)
        # W1 = 1.0 / ((self.relu(d_ - rho_z) / sigma_z) + 1)
        # W1 = W1 ** (self.alpha + 1.0) / 2.0
        # W1 = W1 / torch.sum(W1, dim=1, keepdim=True)
        # W1 = 1.0 / (1.0 + self.relu(d_ - rho_z) / sigma_z)
        # N = W1 / torch.sum(W1, dim=1, keepdim=True)
        # M = N ** 2 / torch.sum(N, dim=0)
        # M = M / torch.sum(M, dim=1, keepdim=True)
        W2 = (- self.relu(d_ - rho_u) / sigma_u).exp()

        S = W1  # + W2 - W1 * W2
        # S = (- self.relu(d_ - rho_z) / sigma_z).exp()
        S = S / torch.sum(S, dim=1, keepdim=True)

        D = S ** 2 / torch.sum(S, dim=0)
        D = D / torch.sum(D, dim=1, keepdim=True)
        # D =(- 1 - self.relu(d_ - rho_z) / sigma_z).exp()
        # D = D / torch.sum(D, dim=1, keepdim=True)

        # D = S ** 2
        return W1, S, W1, D

    def compute_batch(self, z, batch_idx, batch_size, num, epoch):
        if epoch >= 0:
            d_z = torch.sum((z.unsqueeze(1) - z + self.eps) ** 2, dim=2).sqrt().detach()
            Relation_x = d_z.topk(k=2, dim=1, largest=False)
            rho_z = Relation_x[0][:, 1].unsqueeze(1).detach()
            # self.rhos[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)] = rho_z

        d_ = torch.sum((z.unsqueeze(1) - self.mu + self.eps) ** 2, dim=2).sqrt()
        # d_ = torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2)
        # rho_z = self.rhos[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
        # sigma_z = self.sigmas[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
        k = 10
        m = 10
        '''
        Relation_z = d_.topk(k=k, dim=1, largest=False)
        Relation_u = d_.t().topk(k=k, dim=1, largest=False)

        rho_u = Relation_u[0][:, 0].unsqueeze(0)

        mu_size = self.mu.size(0)
        lo = torch.zeros(mu_size, 1)
        hi = 10 * torch.ones(mu_size, 1)
        mid_u = torch.cat([lo, hi], dim=1).cuda()
        sigma_u = torch.ones(1, mu_size).cuda()

        target = np.log2(k) * 1
        for i in range(k * 3):
            current_u = torch.sum((-((Relation_u[0] - rho_u) / sigma_u)).exp(), dim=0)
            mid_u[range(mu_size), (current_u > target) * 1] = sigma_u[:, 0]
            sigma_u = torch.mean(mid_u, dim=1).unsqueeze(0)
        '''

        '''
        # nearest_  = Affinity_Z.topk(k=10, dim=1, largest=False)  # Nearest k samples from Z_i
        near_mu_ = d_.topk(k=10, dim=0, largest=False)  # Nearest k samples from U_j
        Z_near_mu = d_.topk(k=10, dim=1, largest=False)  # Nearest k centers from Z_i

        # rho_ = nearest_[0][:, 1]
        rho_mu_ = near_mu_[0][0, :]
        rho_mu_z = Z_near_mu[0][:, 0]
        rho_mu_ = rho_mu_.detach()
        rho_mu_z = rho_mu_z.detach()

        # theta_ = nearest_[0][:, 3]
        theta_mu_ = near_mu_[0][5, :]  # k + 1
        theta_mu_z = Z_near_mu[0][:, 1]

        # W1_ = (- self.relu(d_ - rho_.unsqueeze(1))     / theta_.unsqueeze(1)).exp()
        # W2_ = (-          (d_ - rho_mu_.unsqueeze(0))  / theta_mu_.unsqueeze(0)).exp()
        # W3_ = (-          (d_ - rho_mu_z.unsqueeze(1)) / theta_mu_z.unsqueeze(1)).exp()
        # W2_ = (          theta_mu_.unsqueeze(0) / (d_ - rho_mu_.unsqueeze(0) + 0.1))
        # W3_ = (          theta_mu_z.unsqueeze(1) / (d_ - rho_mu_z.unsqueeze(1) + theta_mu_z.unsqueeze(1)))
        # W3_ = (theta_mu_z.unsqueeze(1) / (d_ - rho_mu_z.unsqueeze(1) + 1))
        # W2_ = (theta_mu_.unsqueeze(0)  -  (d_ - rho_mu_.unsqueeze(0))).exp()
        W2_ = (theta_mu_.unsqueeze(0) - (d_ - rho_mu_.unsqueeze(0))).exp()
        #W3_ = (theta_mu_z.unsqueeze(1) - (d_ - rho_mu_z.unsqueeze(1))).exp()  # useful
        W3_ = ( - (d_ - rho_mu_z.unsqueeze(1))).exp()  # useful
        # W3_ = (-          (d_ - rho_mu_z.unsqueeze(1)) / (torch.log(theta_mu_z.unsqueeze(1) + 1))).exp()
        # W3_ = (-          (d_)).exp()

        # S = W1_ + W2_ - W1_ * W2_

        # W1_ = W1_ / torch.sum(W1_, dim=1, keepdim=True)
        W2_ = W2_ / torch.sum(W2_, dim=1, keepdim=True)
        W3_ = W3_ / torch.sum(W3_, dim=1, keepdim=True)

        # D1_ = W1_ ** 2 / torch.sum(W1_, dim=0, keepdim=True)
        # D1 = D1_ / torch.sum(D1_, dim=1, keepdim=True)
        # D1 = W1_ / torch.sum(W1_, dim=1, keepdim=True)

        D2_ = W2_ ** 2 / torch.sum(W2_, dim=0)
        D2 = D2_ / torch.sum(D2_, dim=1, keepdim=True)
        # D2 = W2_ / torch.sum(W2_, dim=1, keepdim=True)

        D3 = W3_ ** 2 / torch.sum(W3_,
                                  dim=0)  ############################################################## keepdim=True
        D3 = D3 / torch.sum(D3, dim=1, keepdim=True)
        # D3 = W3_ / torch.sum(W3_, dim=1, keepdim=True)

        # S = S / torch.sum(S, dim=1, keepdim=True)
        '''

        # rho_z = self.rhos_x_u[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]   #self.rhos_x_u
        sigma_z = self.sigmas_x_u[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]  # self.sigmas_x_u
        rho_u = self.rhos_u_x
        sigma_u = self.sigmas_u_x

        rho_u = rho_u.detach()
        rho_z = rho_z.detach()

        # W1 = (- self.relu(d_ - rho_z) / sigma_z).exp()
        # W1 = (- (d_ - rho_z) / sigma_z).exp()
        # W1 = 1 / (1 + self.relu(d_ - rho_z) / sigma_z)
        # W1 = 1 / (1 + self.relu(d_ - rho_z) ** 2 / sigma_z)
        # W1 = 1 / (1 + ((d_ - rho_z) / sigma_z) ** 2)
        # W1 = 1.0 / ((self.relu(d_ - rho_z) / sigma_z) + 1)
        W1 = 1.0 / (1.0 + self.relu(d_ - rho_z) / sigma_z)
        W2 = (- self.relu(d_ - rho_u) / sigma_u).exp()

        S = W1  # + W2 - W1 * W2

        left = np.arange(z.size(0))
        left = np.expand_dims(left, axis=1)
        left = np.repeat(left, m, axis=1)
        # D = Dbatch.topk(k=m, dim=1)
        # S = S[left, D[1]]
        S = S / torch.sum(S, dim=1, keepdim=True)
        # D = Dbatch#[0]

        # D = S ** 2 / torch.sum(S, dim=0)
        # D = D / torch.sum(D, dim=1, keepdim=True)
        return S

    def Similarity_Mantain_Norm(self, zbatch, z, batch_idx, batch_size, num, epoch, m, divide):
        d_z = torch.sum((z.unsqueeze(1) - z + self.eps) ** 2, dim=2).sqrt()
        d_x = torch.sum((zbatch.unsqueeze(1) - zbatch + self.eps) ** 2, dim=2).sqrt()

        d_ = torch.sum((zbatch.unsqueeze(1) - self.mu + self.eps) ** 2, dim=2).sqrt()
        wide = z.size(0)
        m = m  # 5
        divide = wide * divide
        Try = False
        # start = int((wide * 0.13))

        Relation_x = d_x.topk(k=2, dim=1, largest=False)
        # Relation_z = d_z.topk(k=k+1, dim=1, largest=False)
        Relation_x_u = d_.topk(k=self.mu.size(0), dim=1, largest=False)

        if epoch == 0:
            rho_x = Relation_x[0][:, 1].unsqueeze(1)
            # rho_z = Relation_z[0][:, 1].unsqueeze(1)
            self.rhos[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)] = rho_x

            lo = torch.zeros(wide, 1)
            hi = 10000 * torch.ones(wide, 1)
            mid_x = torch.cat([lo, hi], dim=1).cuda()
            sigma_x = torch.ones(wide, 1).cuda()
            mid_z = torch.cat([lo, hi], dim=1).cuda()
            sigma_z = torch.ones(wide, 1).cuda()
            target = np.log2(self.mu.size(0)) * 1
            for i in range(self.mu.size(0) * 30):
                current_x = torch.sum((-(self.relu(Relation_x_u[0] - rho_x) / sigma_x)).exp(), dim=1)
                mid_x[range(wide), (current_x > target) * 1] = sigma_x[:, 0]
                sigma_x = torch.mean(mid_x, dim=1).unsqueeze(1)

                # current_z = torch.sum((-((Relation_z[0][:, 1:k + 1] - rho_z) / sigma_z)).exp(), dim=1)
                # mid_z[range(batch_size), (current_z > target) * 1] = sigma_z[:, 0]
                # sigma_z = torch.mean(mid_z, dim=1).unsqueeze(1)
            self.sigmas[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)] = sigma_x
        else:
            rho_x = self.rhos[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)]
            sigma_x = self.sigmas[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)]

        rho_x = rho_x.detach()

        # theta_x = Relation_x[0][:, 2]
        # theta_z = Relation_z[0][:, 2]
        # theta_x = self.sigmas[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]

        # Wx = (- self.relu(d_x - rho_x.unsqueeze(1)) / theta_x.unsqueeze(1)).exp()
        # Wz = (- self.relu(d_z - rho_z.unsqueeze(1)) / theta_z.unsqueeze(1)).exp()
        Wx = (- self.relu(d_x - rho_x) / sigma_x).exp()
        Wz = (- self.relu(d_z - rho_x) / sigma_x).exp()

        if epoch >= 0:
            limit = (-((Relation_x_u[0] - rho_x) / sigma_x)).exp().topk(k=3, dim=1)[0][:, 1]  # 1
            self.start[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)] = torch.sum(
                (Wx - limit.unsqueeze(1) > 0) * 1, dim=1).unsqueeze(1) + divide

        if Try:
            start = self.start[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)].numpy().mean()
            start = int(start)
        else:
            start = self.start[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)].numpy()
        # print(start)
        # start = 130
        Sx = Wx + Wx.t() - Wx * Wx.t()
        Sz = Wz + Wz.t() - Wz * Wz.t()

        left = np.arange(z.size(0))
        left = np.expand_dims(left, axis=1)
        left = np.repeat(left, m, axis=1)
        if Try:
            Sx = Sx.topk(k=int(start) + m, dim=1)
        else:
            Sx = Sx.topk(k=int(start.max()) + m, dim=1)
        Sz1 = Sz[left, Sx[1][:, 1:m + 1]]
        Sx1 = Sx[0][:, 1:m + 1]

        # m=2
        # start = 150#130 150
        # left = np.arange(z.size(0))
        # left = np.expand_dims(left, axis=1)
        # left = np.repeat(left, m, axis=1)
        # Sx = Sx.topk(k=m, dim=1, largest=False)

        if Try:
            Sz2 = Sz[left, Sx[1][:, start:start + m]]
            Sx2 = Sx[0][:, start:start + m]
        else:
            up = np.arange(m)
            up = np.expand_dims(up, axis=0)
            up = np.repeat(up, wide, axis=0)
            Sz2 = Sz[left, Sx[1][left, up + start]]
            Sx2 = Sx[0][left, up + start]

        '''
        tmp = Sx.topk(k=z.size(0), largest=True)
        left = np.arange(z.size(0))
        left = np.expand_dims(left, axis=1)
        left = np.repeat(left, z.size(0) - 1, axis=1)
        maxWx = tmp[0][:, 1:z.size(0)]
        maxWz = Sz[left, tmp[1][:, 1:z.size(0)]]

        maxWx = maxWx / torch.sum(maxWx, dim=1, keepdim=True)
        maxWz = maxWz / torch.sum(maxWz, dim=1, keepdim=True)

        # minWx = tmp[0][:, z.size(0)-5:z.size(0)]
        # minWz = Sz[left, tmp[1][:, z.size(0)-5:z.size(0)]]
        '''

        # return torch.mean(torch.sum(maxWx * torch.log((1+maxWx+self.eps)/(1+maxWz+self.eps)), dim=1))
        # return torch.mean(torch.sum(Sx1 * torch.log(Sx1 / (Sz1 + self.eps)) + (1 - Sx2) * torch.log((1 - Sx2 + self.eps) / (1 - Sz2 + self.eps)), dim=1))
        # return torch.mean(torch.sum(Sx1 * torch.log(Sx1 / (Sz1 + self.eps)), dim=1)) #+ 0.5 * (1 - Sx1) * torch.log((1 - Sx1 + self.eps) / (1 - Sz1 + self.eps)), dim=1))
        # return torch.mean(torch.sum(Sx1 * torch.log(Sx1 / (Sz1 + self.eps)) + (1 - Sx1) * torch.log((1 - Sx1 + self.eps) / (1 - Sz1 + self.eps)), dim=1))
        # return torch.mean(torch.sum(Sx1 * torch.log(Sx1 / (Sz1 + self.eps)), dim=1)) + 0.2 * torch.mean(torch.sum((1 - Sx2) * torch.log((1 - Sx2 + self.eps) / (1 - Sz2 + self.eps)), dim=1))
        # return torch.mean(torch.sum(Sx1 * torch.log(Sx1 / (Sz1 + self.eps)), dim=1))
        return self.loss(Sz1, Sz2, Sx1, Sx2, one=False, pur=False)
        # return torch.mean(torch.sum((1 - Sx2) * torch.log((1 - Sx2 + self.eps) / (1 - Sz2 + self.eps)), dim=1))
        # return self.loss(maxWz, maxWz, maxWx, maxWx, one=False, pur=False)
        # return self.mseloss(maxWx, maxWz)
        # return torch.mean(torch.sum(maxWx - maxWz, dim=1)) #+ torch.mean(torch.sum(minWz - minWx, dim=1))
        # return self.mseloss(Sx, Sz)
        # return torch.mean(torch.sum(Wx * torch.log((Wx + self.eps) / (Wz + self.eps)), dim=1) + torch.sum((1-Wx) * torch.log((1-Wx+self.eps)/(1-Wz+self.eps)), dim=1))
