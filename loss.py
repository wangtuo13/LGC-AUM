import torch
from torch import nn
import numpy as np
import math
from utils import positive, negtive
import torch.nn.functional as F
from torch.nn import Parameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class lgc_loss(nn.Module):
    def __init__(self, n_cluster=10, hidden_dim=10, eps=0.000001, total=2000, intra_batchsize=256, inter_batchsize=256):
        super(lgc_loss, self).__init__()
        #self.in_dim = dim[0]
        #self.nlayer = len(dim) - 1
        #self.layers = dim
        #self.numpen = numpen
        #self.encoder = self.build_net(growthRate, numpen)
        self.total = total
        self.mu = Parameter(torch.Tensor(n_cluster, hidden_dim))
        torch.nn.init.normal_(self.mu, mean=0, std=1)
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
        self.softmax = nn.Softmax(dim=1)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.Citer = torch.zeros(n_cluster, n_cluster).to(device)
        self.mseloss = nn.MSELoss()
        self.F = Parameter(torch.Tensor(n_cluster, n_cluster).to(device))
        self.G = Parameter(torch.Tensor(total, n_cluster).to(device))
        self.eye_K = torch.eye(self.num_class).to(device)
        self.position_K = (torch.ones(self.num_class, self.num_class).to(device) - self.eye_K).to(device)
        self.just_other_intra = (torch.ones(intra_batchsize, intra_batchsize) - torch.eye(intra_batchsize)).to(device)
        self.just_other_inter = (torch.ones(inter_batchsize, inter_batchsize) - torch.eye(inter_batchsize)).to(device)

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
                return torch.mean(
                    torch.sum(D1 * torch.log((D1 + self.eps) / (q1 + self.eps)), dim=1) +
                    torch.sum((1 - D2 + self.eps) * torch.log((1 - D2 + self.eps) / (1 - q2 + self.eps)), dim=1)
                )

    def load_model(self, pretrained_dict):
        self.mu.data.copy_(pretrained_dict.mu)

    def loss2(self, q1, q2, D1, D2, pur=False):
        # q3 = torch.cat([q1, q2*D1.expand(-1,q2.shape[1])], dim=1)
        # q1 =q1*torch.exp((1-D1))
        # q3 = torch.cat([q1, q2*torch.exp(D1.expand(-1,q2.shape[1]))], dim=1)
        # try
        # q3 = torch.cat([q1*(1-D1), D1.expand(-1, q2.shape[1]) * q2], dim=1)
        q3 = torch.cat([torch.exp(-D1)*q1, torch.exp(-D2)*q2], dim=1)
        #q3 = torch.exp(-D2) * q2 + eps
        #q4 = 1-q3
        #q5=(torch.exp(torch.exp(-D1)*q1)/ (torch.sum(torch.exp(q3), dim=1, keepdim=True) + self.eps))
        #q5 = (torch.exp(-D1) * q1) / (torch.sum(q3, dim=1, keepdim=True) + self.eps)+ self.eps
        # q3 = torch.cat([q1, D1.expand(-1, q2.shape[1]) * torch.exp(q2)], dim=1)
        # q3 = torch.diag(0.5*torch.sum(q3+torch.transpose(q3, 1, 0))) - q3
        # q4=torch.cat([q1,1-q2],dim=1)
        if pur:
            return torch.mean(- torch.sum(D1 * torch.log(q1), dim=1))
        else:
            #return torch.mean(
            #    - torch.sum(D1 * torch.log(q1), dim=1)
            #    - torch.sum((1 - D2 + self.eps) * torch.log(1 - q2 + self.eps), dim=1)
            #)

            ''''''
            ''''''
            #return torch.mean(
                #- torch.sum( torch.log(torch.exp(q1)/(torch.sum(torch.exp(q3), dim=1, keepdim=True)+self.eps)), dim=1)
               # - torch.sum( torch.log(torch.exp(1 - D2)/(torch.sum(torch.exp(1-q3), dim=1, keepdim=True)+self.eps)), dim=1)
            #)

            return torch.mean(

                    - torch.sum(D1 * torch.log(torch.exp(torch.exp(-D1)*q1)/(torch.sum(torch.exp(q3), dim=1, keepdim=True) + self.eps)), dim=1)
                    ##torch.exp(- torch.sum(D1 * torch.log((torch.exp(-D1) * q1)/((
                    ##            torch.sum(q3, dim=1, keepdim=True) + self.eps) * D1)), dim=1))
                    ##+torch.exp(- torch.sum(q5 * torch.log(D1/q5)))
                #   - torch.sum(D1 * torch.log(torch.exp(q1)/ (torch.sum(torch.exp(q3), dim=1, keepdim=True) + self.eps)), dim=1)# only one 90 at iter 50 with q3 = torch.cat([q1, q2], dim=1)
                #   - torch.sum((1 - D2) * torch.log(torch.exp((1 - q2)*(1 - D2))/(torch.sum(torch.exp(q4), dim=1, keepdim=True) + self.eps)), dim=1) # two terms failure
                # 2021-11-13 potential available
                # torch.exp(- torch.sum(torch.log((q1) / (torch.sum((q3), dim=1, keepdim=True) + self.eps)), dim=1))

                #- torch.sum(torch.log((q1) / (torch.sum((q3), dim=1, keepdim=True) + self.eps)), dim=1)
                #- torch.sum(torch.log((q1) / (torch.sum((q3), dim=1, keepdim=True) + self.eps)), dim=1) # relatively stable low performance with q1 =q1*torch.exp((1-D1)) q3 = torch.cat([q1, q2*torch.exp(D1.expand(-1,q2.shape[1]))], dim=1)
                # 2021-11-14 test
                #torch.exp(- torch.sum(torch.log( torch.exp(q1) / (torch.sum( torch.exp(q3), dim=1, keepdim=True) + self.eps)), dim=1))
                #just use (- torch.sum(torch.log(torch.exp(q1) / (torch.sum(torch.exp(q3), dim=1, keepdim=True) + self.eps)),dim=1)) ## 这个这个这个
                #(- torch.sum(torch.log(torch.exp(q1) / (torch.sum(torch.exp(q3), dim=1, keepdim=True) + self.eps)),dim=1))
                # the below is worthy trying more times again
                #(- torch.sum( torch.log(torch.exp(q1 * (1 - D1)) / (torch.sum(torch.exp(q3), dim=1, keepdim=True) + self.eps)),dim=1)) #with q3 = torch.cat([q1*(1-D1), q2], dim=1)
                # the below loss term seemingly works well !!
                #(- torch.sum(torch.log(torch.exp(q1) / (torch.sum(torch.exp(q3), dim=1, keepdim=True) + self.eps)),dim=1)) # with q3 = torch.cat([q1*(1-D1), q2], dim=1)
                #(- torch.sum(torch.log(torch.exp(q1) / (torch.sum(torch.exp(q3), dim=1, keepdim=True) + self.eps)), dim=1))
                # 2021-11-14 test not best
                #+ (- torch.sum( torch.log(torch.exp(1 - q2)/(torch.sum(torch.exp(1-q3), dim=1, keepdim=True)+self.eps)), dim=1))
            )

            #return torch.mean(
            #    - torch.sum(torch.log(q4 / (torch.sum(q3, dim=1, keepdim=True) + self.eps)), dim=1)
            #)
            # 可以试一下这个思想。利用增广样本，知道h的标签，也就是利用h写一个loss
            #

    def Lmc_Lmc_r(self, z1, z2):
        self.Citer = 0.1 * self.Citer.detach() + z1.T @ z2
        Lmc = (torch.sum(self.Citer * self.position_K)) / self.num_class
        #self.Citer[position, position] = 0
        #Lmc = torch.sum(self.Citer) / self.num_class
        one = torch.ones(z1.size(0), z1.size(0)).to(device)
        Lmc_r = torch.sum((z1.T @ one @ z2) * self.eye_K) / z1.size(0)
        return Lmc_r + Lmc

    def Lmc_Lmc_r_n(self, z1, z2):
        n = z1.size(0)
        position_n = (- 1 * torch.eye(n)).to(device)
        Citer_n12 = z1 @ z2.T
        Citer_n11 = z1 @ z1.T
        Citer_n12 = Citer_n12 / Citer_n12.sum(dim=1)
        Citer_n11 = Citer_n11 / Citer_n11.sum(dim=1)
        Lmc = (torch.sum(Citer_n12 * position_n)) / n + (torch.sum(Citer_n11 * position_n)) / n
        return - torch.log(Lmc)

    def init_G(self, D, y):
        self.G[range(self.total), y] = 1
        self.G = self.G + 0.2

        #self.G[range(self.total), torch.max(D, dim=1)[1]] = 1
        #self.G = self.G + 0.2

        #self.G = torch.rand(self.total, self.num_class).to(device)

        return

    def _NMF(self, z, alpha, rho):
        z = z.T
        # Update F
        #self.F = z @ self.G @ (self.G.T @ self.G).inverse()

        # Update G
        #self.G = self.G * ((positive(z.T @ self.F) + self.G @ negtive(negtive(self.F.T @ self.F))) /
        #                   (negtive(z.T @ self.F) + self.G @ positive(positive(self.F.T @ self.F))+self.eps)).sqrt()

        return 0.5 * torch.norm(z - self.F @ self.G.T) + alpha * rho * torch.norm(self.G, p=1) + alpha * (1 - rho) / 2 * torch.norm(self.G)

    def compute_S_(self, z, epoch, h_mu=None):
        # Affinity_Z = torch.sum((z.unsqueeze(1) - z + self.eps) ** 2, dim=2).sqrt()
        # d_ = torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2)#######################################.sqrt()
        if h_mu == None:
            d_ = torch.sum((z.unsqueeze(1) - self.mu + self.eps) ** 2, dim=2).sqrt()
        else:
            d_ = torch.sum((z.unsqueeze(1) - h_mu + self.eps) ** 2, dim=2).sqrt()
        # d_1 = torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2).sqrt()
        # d_ = torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2).sqrt() / (torch.sum(z ** 2, dim=1).sqrt().unsqueeze(1) * torch.sum(self.mu ** 2, dim=1).sqrt().unsqueeze(0))
        # d_ = d_1 * d_2

        k = min(5, self.mu.size(0)) #5
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
            mid_z = torch.cat([lo, hi], dim=1).to(device)
            sigma_z = torch.ones(sample_size, 1).to(device)

            mu_size = self.mu.size(0)
            lo = torch.zeros(mu_size, 1)
            hi = 10000 * torch.ones(mu_size, 1)
            mid_u = torch.cat([lo, hi], dim=1).to(device)
            sigma_u = torch.ones(1, mu_size).to(device)

            target = np.log2(k) * 1 - 1 # 1.5
            target1 = np.log2(k) * 1
            for i in range(k * 30):
                current_u = torch.sum((-(self.relu(Relation_u[0].t() - rho_u) / sigma_u)).exp(), dim=0)

                mid_u[range(mu_size), (current_u > target1) * 1] = sigma_u[0, :]

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

        S = W1 + W2 - W1 * W2
        # S = (- self.relu(d_ - rho_z) / sigma_z).exp()
        S = S / torch.sum(S, dim=1, keepdim=True)

        D = S ** 2 / torch.sum(S, dim=0)
        D = D / torch.sum(D, dim=1, keepdim=True)
        # D =(- 1 - self.relu(d_ - rho_z) / sigma_z).exp()
        # D = D / torch.sum(D, dim=1, keepdim=True)

        # D = S ** 2
        return W1, S, W1, D

    def compute_batch(self, z, batch_idx, batch_size, num, epoch, h_mu=None):
        if epoch >= 0:
            d_z = torch.sum((z.unsqueeze(1) - z + self.eps) ** 2, dim=2).sqrt().detach()
            Relation_x = d_z.topk(k=2, dim=1, largest=False)
            rho_z = Relation_x[0][:, 1].unsqueeze(1).detach()
            # self.rhos[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)] = rho_z

        if h_mu == None:
            d_ = torch.sum((z.unsqueeze(1) - self.mu + self.eps) ** 2, dim=2).sqrt()
        else:
            d_ = torch.sum((z.unsqueeze(1) - h_mu + self.eps) ** 2, dim=2).sqrt()
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
        mid_u = torch.cat([lo, hi], dim=1).to(device)
        sigma_u = torch.ones(1, mu_size).to(device)

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
        '''
        Z_near_mu = d_.topk(k=10, dim=1, largest=False)  # Nearest k centers from Z_i
        rho_mu_z = Z_near_mu[0][:, 0].unsqueeze(1)
        rho_mu_z = rho_mu_z.detach()
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

    def Norm_U(self):
        I = torch.eye(self.num_class).cuda()
        L = self.mu/(torch.sum(self.mu ** 2, dim=1, keepdim=True).sqrt())
        L = L.mm(L.t())
        return self.mseloss(L, I)
        #Angle = torch.sum(self.mu.unsqueeze(1) * self.mu, dim=2) / (
        #            torch.sum(self.mu ** 2, dim=1).sqrt().unsqueeze(1) * torch.sum(self.mu ** 2,
        #                                                                           dim=1).sqrt().unsqueeze(0))
        #return torch.sum(Angle.topk(k=2, dim=1)[0][:, 1])

    def Similarity_Mantain_Norm(self, zbatch, z, batch_idx, batch_size, num, epoch, m, divide, stage, z1, decay=1):
        d_z = torch.sum((z.unsqueeze(1) - z1 + self.eps) ** 2, dim=2).sqrt()
        d_x = torch.sum((zbatch.unsqueeze(1) - zbatch + self.eps) ** 2, dim=2).sqrt()

        d_ = torch.sum((zbatch.unsqueeze(1) - self.mu.detach() + self.eps) ** 2, dim=2).sqrt()
        wide = z.size(0)
        m = m  # 5
        divide = int(wide * divide)
        Try = False
        # start = int((wide * 0.13))

        Relation_x = d_x.topk(k=2, dim=1, largest=False)
        # Relation_z = d_z.topk(k=k+1, dim=1, largest=False)
        Relation_x_u = d_.topk(k=self.mu.size(0), dim=1, largest=False)

        if epoch >= 0:
            rho_x = Relation_x[0][:, 1].unsqueeze(1)
            #rho_x = Relation_x_u[0][:, 0].unsqueeze(1)
            # rho_z = Relation_z[0][:, 1].unsqueeze(1)
            self.rhos[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)] = rho_x

            lo = torch.zeros(wide, 1)
            hi = 10000 * torch.ones(wide, 1)
            mid_x = torch.cat([lo, hi], dim=1).to(device)
            sigma_x = torch.ones(wide, 1).to(device)
            mid_z = torch.cat([lo, hi], dim=1).to(device)
            sigma_z = torch.ones(wide, 1).to(device)
            #target = np.log2(15) #np.log2(self.mu.size(0)) * 1 - 1
            target = np.log2(self.mu.size(0)) * 1 - 1
            for i in range(self.mu.size(0) * 30):
                current_x = torch.sum((-(self.relu(Relation_x_u[0]  - rho_x) / sigma_x)).exp(), dim=1)
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

        if epoch == 0:

            limit = (-((Relation_x_u[0] - rho_x) / sigma_x)).exp().topk(k=3, dim=1)[0][:, 1]  # 1
            self.start[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)] = torch.sum(
                (Wx - limit.unsqueeze(1) > 0) * 1, dim=1).unsqueeze(1) + divide
            '''
            limit = (-((Relation_x_u[0] - rho_x) / sigma_x)).exp().topk(k=min(stage+2, self.num_class), dim=1)[0][:, min(stage+1, self.num_class-1)]  # min(stage+1, self.num_class)  1
            max_num = z.size(0) * torch.ones(z.size(0)).to(device).int() - m - 1
            self.start[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)] = \
                torch.min(torch.sum((Wx - limit.unsqueeze(1) > 0) * 1, dim=1).int() + divide, max_num).unsqueeze(1)
            '''
        if Try:
            start = self.start[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)].numpy().mean()
            start = int(start)
        else:
            start = self.start[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)].numpy()
        # print(start)
        # start = 130
        Sx = Wx + Wx.t() - Wx * Wx.t()
        Sz = Wz + Wz.t() - Wz * Wz.t()

        New_ele_fenmu = torch.pow(Sz, Sx)

        left = np.arange(z.size(0))
        left = np.expand_dims(left, axis=1)
        left = np.repeat(left, m, axis=1)
        if Try:
            Sx = Sx.topk(k=int(start) + m, dim=1)
        else:
            Sx = Sx.topk(k=min(int(start.max()) + m, Sx.size(1)), dim=1)
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
        N = Sz1.size(0)
        New_ele =  torch.pow(Sz1, Sx1)
        New_ele = torch.log(New_ele / (torch.sum(New_ele_fenmu, dim=1, keepdim=True) + self.eps))
        return - torch.sum(New_ele)
        # return torch.mean(torch.sum(maxWx * torch.log((1+maxWx+self.eps)/(1+maxWz+self.eps)), dim=1))
        # return torch.mean(torch.sum(Sx1 * torch.log(Sx1 / (Sz1 + self.eps)) + (1 - Sx2) * torch.log((1 - Sx2 + self.eps) / (1 - Sz2 + self.eps)), dim=1))
        # return torch.mean(torch.sum(Sx1 * torch.log(Sx1 / (Sz1 + self.eps)), dim=1)) #+ 0.5 * (1 - Sx1) * torch.log((1 - Sx1 + self.eps) / (1 - Sz1 + self.eps)), dim=1))
        # return torch.mean(torch.sum(Sx1 * torch.log(Sx1 / (Sz1 + self.eps)) + (1 - Sx1) * torch.log((1 - Sx1 + self.eps) / (1 - Sz1 + self.eps)), dim=1))
        # return torch.mean(torch.sum(Sx1 * torch.log(Sx1 / (Sz1 + self.eps)), dim=1)) + 0.2 * torch.mean(torch.sum((1 - Sx2) * torch.log((1 - Sx2 + self.eps) / (1 - Sz2 + self.eps)), dim=1))
        # return torch.mean(torch.sum(Sx1 * torch.log(Sx1 / (Sz1 + self.eps)), dim=1))
        #return self.loss(Sz1, Sz2, Sx1, Sx2, one=False, pur=False)
        # return torch.mean(torch.sum((1 - Sx2) * torch.log((1 - Sx2 + self.eps) / (1 - Sz2 + self.eps)), dim=1))
        # return self.loss(maxWz, maxWz, maxWx, maxWx, one=False, pur=False)
        # return self.mseloss(maxWx, maxWz)
        # return torch.mean(torch.sum(maxWx - maxWz, dim=1)) #+ torch.mean(torch.sum(minWz - minWx, dim=1))
        # return self.mseloss(Sx, Sz)
        # return torch.mean(torch.sum(Wx * torch.log((Wx + self.eps) / (Wz + self.eps)), dim=1) + torch.sum((1-Wx) * torch.log((1-Wx+self.eps)/(1-Wz+self.eps)), dim=1))

    def maintain_similarity(self, z0, z1, zbatch, zbatch0, zbatch1, epoch, batch_idx, batch_size):
        wide = z1.size(0)
        # compute inner product as the similarity for simplicity
        mask = torch.ones((wide, wide)).to(device)
        mask = mask.fill_diagonal_(0).bool()
        sim_inner = torch.mm(z0, z1.T)
        sim_inner_same = torch.diag(sim_inner).reshape(wide, 1)
        sim_inner_diff = sim_inner[mask].reshape(wide, -1)


        # 样本x的特征z, 样本x的增广样本x0的特征z0; 它们两者间的距离
        #d_z = (torch.sum((z0.unsqueeze(1) - z) ** 2, dim=2) + self.eps).sqrt() # eps应该放在平方的外面吧，否则岂不是引入了误差

        # 就是这个d_x，这个zbatch是每个batch才重新计算的新的，其流形随着网络参数改变而不断变化
        # 而之前的代码，是在一个epoch开始阶段，把每个batch计算后，保存起来，期间就不变了，甚至多个epoch，也保持同一个zbatch
        # loss上升的问题一直存在，或重或轻，但之前是上升一下下，就开始降了，前期的论文的代码
        d_x = (torch.sum((torch.abs(zbatch0.unsqueeze(1) - zbatch)) ** 2, dim=2)+self.eps).sqrt()


        if wide == batch_size:
            d_x = d_x * self.just_other_intra #
        else:
            d_x = d_x * (torch.ones(wide, wide) - torch.eye(wide)).to(device) # 非对角元素值

        mask = torch.ones((wide, wide)).to(device)
        mask = mask.fill_diagonal_(0).bool()

        #idx = d_x.topk(k=50, dim=1, largest=False)[1][:, 1:50] #距离最小值的索引
        # 距离最小值
        topkk = min(3, wide)
        Relation_x_ = d_x.topk(k=topkk, dim=1, largest=False)[0][:, 1:topkk]
        #Relation_z_ = d_z.topk(k=3, dim=1, largest=False)[0][:, 1:3]

        re_x1 = (torch.sum((zbatch0 - zbatch1) ** 2, dim=1) + self.eps).sqrt().unsqueeze(1)
        #re_x2 = (torch.sum((zbatch0 - zbatch2 ) ** 2, dim=1) + self.eps).sqrt().unsqueeze(1)
        #re_z1 = (torch.sum((z0 - z1) ** 2, dim=1) + self.eps).sqrt().unsqueeze(1)
        #re_z2 = (torch.sum((z0 - z2) ** 2, dim=1) + self.eps).sqrt().unsqueeze(1)

        #re_x = torch.cat((re_x1, re_x2), dim=1)   # 2021-11-13 comment this
        #re_z = torch.cat((re_z1, re_z2), dim=1)   # 2021-11-13 comment this
        re_x = re_x1
        #re_z = re_z1
        Relation_x_ = Relation_x_ + re_x.max(dim=1)[0].unsqueeze(1) # 2021-11-16 uncomment this
        #Relation_z = Relation_z_ + re_x.max(dim=1)[0].unsqueeze(1)

        Relation_x = torch.cat((re_x, Relation_x_), dim=1)  # 2021-11-13 Relation_x->Relation_x_

        #if epoch % 3 == 0:  # 2021.11.14 19：03 被王拓修改前
        if epoch % 1 == 0:   # 2021.11.14 19：03 被王拓修改后
            rho_x = torch.min(Relation_x_.min(dim=1)[0], re_x.min(dim=1)[0]).unsqueeze(1)
            #rho_z = re_z.min(dim=1)[0].unsqueeze(1)
            self.rhos[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)] = rho_x

            lo = torch.zeros(wide, 1)
            hi = 10000 * torch.ones(wide, 1)
            mid_x = torch.cat([lo, hi], dim=1).to(device)
            sigma_x = torch.ones(wide, 1).to(device)
            mid_z = torch.cat([lo, hi], dim=1).to(device)
            sigma_z = torch.ones(wide, 1).to(device)
            target = np.log2(4) #np.log2(self.mu.size(0)) * 1 - 1
            #target = np.log2(self.mu.size(0)) * 1 - 1
            for i in range(self.mu.size(0) * 30):
                current_x = torch.sum((-(self.relu(Relation_x - rho_x) / sigma_x)).exp(), dim=1)
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
        #rho_z = rho_z.detach()

        Wx = (- self.relu(re_x - rho_x) / sigma_x).exp()
        #Wz = (- self.relu(re_z - rho_z) / sigma_x).exp()

        #left = np.arange(wide)
        #left = np.expand_dims(left, axis=1)
        #left = np.repeat(left, 49, axis=1)

        Sx1 = Wx
        #Sz1 = Wz

        Wx2 = (- self.relu(d_x + re_x.max(dim=1)[0].unsqueeze(1) - rho_x) / sigma_x).exp()
        #Wz2 = (- self.relu(d_z + re_z.max(dim=1)[0].unsqueeze(1) - rho_z) / sigma_x).exp()

        Sx2 = Wx2 + Wx2.T - Wx2 * Wx2.t()
        #Sz2 = Wz2 + Wz2.T - Wz2 * Wz2.t()

        Sx2 = Sx2[mask].reshape(wide, -1) #/ (wide - 1) comments 2021.11.13 23:16
        #Sz2 = Sz2[mask].reshape(wide, -1) #/ (wide - 1)

        #return self.loss(Sz1, Sz2, Sx1, Sx2, one=True, pur=False)
        Sz1 = sim_inner_same
        Sz2 = sim_inner_diff
        return self.loss2(Sz1, Sz2, Sx1, Sx2, pur=False)

    #def sim_preserve(self, z, z1, z2, z0, zbatch, zbatch1, zbatch2, zbatch0, epoch, batch_idx, batch_size):
    def sim_preserve(self, z1, z0):
        wide = z1.size(0)
        mask = torch.ones((wide, wide)).to(device)
        mask = mask.fill_diagonal_(0).bool()


        # compute inner product as the similarity for simplicity
        sim_inner = torch.mm(z0, z1.T)
        sim_inner_same = torch.diag(sim_inner).reshape(wide, 1)
        sim_inner_diff = sim_inner[mask].reshape(wide, -1)

        '''
        ############ compute the uniform similarity ######
        d_z = (torch.sum((z0.unsqueeze(1) - z) ** 2, dim=2) + self.eps).sqrt()  # eps应该放在平方的外面吧，否则岂不是引入了误差
        # Relation_z_ = d_z.topk(k=3, dim=1, largest=False)[0][:, 1:3]

        re_z1 = (torch.sum((z0 - z1) ** 2, dim=1) + self.eps).sqrt().unsqueeze(1)
        re_z2 = (torch.sum((z0 - z2) ** 2, dim=1) + self.eps).sqrt().unsqueeze(1)
        re_z = re_z1
        # re_z = torch.cat((re_z1, re_z2), dim=1)   # 2021-11-13 comment this
        # Relation_z = Relation_z_ + re_x.max(dim=1)[0].unsqueeze(1)


        d_x = (torch.sum((torch.abs(zbatch0.unsqueeze(1) - zbatch)) ** 2, dim=2) + self.eps).sqrt()
        if wide == batch_size:
            d_x = d_x * self.just_other  #
        else:
            d_x = d_x * (torch.ones(wide, wide) - torch.eye(wide)).to(device)  # 非对角元素值

        # idx = d_x.topk(k=50, dim=1, largest=False)[1][:, 1:50] #距离最小值的索引
        # 距离最小值
        Relation_x_ = d_x.topk(k=3, dim=1, largest=False)[0][:, 1:3]
        re_x1 = (torch.sum((zbatch0 - zbatch1) ** 2, dim=1) + self.eps).sqrt().unsqueeze(1)
        re_x2 = (torch.sum((zbatch0 - zbatch2) ** 2, dim=1) + self.eps).sqrt().unsqueeze(1)

        # re_x = torch.cat((re_x1, re_x2), dim=1)   # 2021-11-13 comment this
        re_x = re_x1
        # Relation_x = Relation_x_ + re_x.max(dim=1)[0].unsqueeze(1) # 2021-11-13 comment this


        Relation_x = torch.cat((re_x, Relation_x_), dim=1)  # 2021-11-13 Relation_x->Relation_x_

        if epoch % 3 == 0:
            rho_z = re_z.min(dim=1)[0].unsqueeze(1)

            rho_x = torch.min(Relation_x_.min(dim=1)[0], re_x.min(dim=1)[0]).unsqueeze(1)
            self.rhos[batch_idx * batch_size: min((batch_idx + 1) * batch_size, self.total)] = rho_x

            lo = torch.zeros(wide, 1)
            hi = 10000 * torch.ones(wide, 1)
            mid_x = torch.cat([lo, hi], dim=1).to(device)
            sigma_x = torch.ones(wide, 1).to(device)
            mid_z = torch.cat([lo, hi], dim=1).to(device)
            sigma_z = torch.ones(wide, 1).to(device)
            target = np.log2(4)  # np.log2(self.mu.size(0)) * 1 - 1
            # target = np.log2(self.mu.size(0)) * 1 - 1
            for i in range(self.mu.size(0) * 30):
                current_x = torch.sum((-(self.relu(Relation_x - rho_x) / sigma_x)).exp(), dim=1)
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
        rho_z = rho_z.detach()


        Wx = (- self.relu(re_x - rho_x) / sigma_x).exp()
        Sx1 = Wx
        Wz = (- self.relu(re_z - rho_z) / sigma_x).exp()
        Sz1 = Wz


        ### Compute Sx, Sz
        Wx2 = (- self.relu(d_x + re_x.max(dim=1)[0].unsqueeze(1) - rho_x) / sigma_x).exp()
        Sx2 = Wx2 + Wx2.T - Wx2 * Wx2.t()
        Sx2 = Sx2[mask].reshape(wide, -1) / (wide - 1)

        Wz2 = (- self.relu(d_z + re_z.max(dim=1)[0].unsqueeze(1) - rho_z) / sigma_x).exp()
        Sz2 = Wz2 + Wz2.T - Wz2 * Wz2.t()
        Sz2 = Sz2[mask].reshape(wide, -1) / (wide - 1)
        '''

        # return self.loss(Sz1, Sz2, Sx1, Sx2, one=True, pur=False)

        return self.selfcontrastive_loss(sim_inner_same, sim_inner_diff, pur = False)

    def selfcontrastive_loss(self, sim_inner_same, sim_inner_diff,  pur=True):
        #   q3 = torch.cat([q1, q2], dim=1)
        #    q3 = torch.diag(0.5*torch.sum(q3+torch.transpose(q3, 1, 0))) - q3
        # q4=torch.cat([q1,1-q2],dim=1)
        if pur:
            return torch.mean(- torch.sum(D1 * torch.log(q1), dim=1))
        else:
            #return torch.mean(
            #    - torch.sum(D1 * torch.log(q1), dim=1)
            #    - torch.sum((1 - D2 + self.eps) * torch.log(1 - q2 + self.eps), dim=1)
            #)

            return torch.mean(
                - torch.sum(torch.log(torch.exp(sim_inner_same)/ (torch.sum(torch.exp(torch.cat([sim_inner_same, sim_inner_diff],dim=1)), dim=1, keepdim=True) + self.eps)), dim=1)
                #- torch.sum( torch.log(q3[:, 1]/(torch.sum(q3, dim=1, keepdim=True)+self.eps)), dim=1)
                #- torch.sum( torch.log((1 - D2)/(torch.sum(1-q3, dim=1, keepdim=True)+self.eps)), dim=1)
            )
