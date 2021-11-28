import sys
#sys.path.append(r'E:\Data')
from sklearn.cluster import KMeans
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
#from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from model import *
from CNN_VAE2 import *
from loss import *
from load_data import *
from utils import *
from sklearn.manifold import TSNE
import numpy as np
#import umap
#import math
import math
import scipy.io
'''
#test = 'Palm'
#test = 'Mnist'
#test = 'Fashion'
#test = 'USPS'
#test = 'Coil_20'
#test = 'Yale'
#test = 'Cifar_10'
if test=='Palm':
    Data = Data_style(name='Palm',      input_size=256, n_cluster=100, hidden_dim=100, shuffle=False, pretrain=False,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None,
                      cluster_batchsize=1000, cluster_epoch=100,  cluster_lr=0.0001, change=1,  m=10, divide=0.03, total=2000)
elif test == 'Mnist':
    Data = Data_style(name='Mnist',     input_size=784, n_cluster=10, hidden_dim=10, shuffle=False, pretrain=False,
                      pretrain_batchsize=256, pretrain_epoch=800, pretrain_lr=0.01, noise=0.2, trans=None,
                      cluster_batchsize=1000, cluster_epoch=5,  cluster_lr=0.001, change=2,  m=10, divide=0.1, total=70000)
elif test == 'Yale':
    Data = Data_style(name='Yale',      input_size=1024, n_cluster=38, hidden_dim=38, shuffle=False, pretrain=False,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None,
                      cluster_batchsize=500, cluster_epoch=100,  cluster_lr=0.001, change=30,  m=5, divide=0.01, total=2000)
elif test == 'Fashion':
    Data = Data_style(name='Fashion',   input_size=784, n_cluster=10, hidden_dim=10, shuffle=False, pretrain=False,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None,
                      cluster_batchsize=1000, cluster_epoch=100,  cluster_lr=0.001, change=30,  m=10, divide=0.1, total=70000)
elif test == 'USPS':
    Data = Data_style(name='USPS',      input_size=256, n_cluster=10, hidden_dim=10, shuffle=False, pretrain=False,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None,
                      cluster_batchsize=1000, cluster_epoch=100,  cluster_lr=0.001, change=30,  m=10, divide=0.04, total=9298)
elif test == 'Coil_20':
    Data = Data_style(name='Coil_20',   input_size=16384, n_cluster=20, hidden_dim=20, shuffle=False, pretrain=False,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None,
                      cluster_batchsize=1000, cluster_epoch=100,  cluster_lr=0.001, change=30,  m=10, divide=0.1, total=2000)
elif test == 'Cifar_10':
    Data = Data_style(name='Cifar_10',  input_size=3072, n_cluster=10, hidden_dim=10, shuffle=False, pretrain=False,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None,
                      cluster_batchsize=1000, cluster_epoch=100,  cluster_lr=0.001, change=30,  m=10, divide=0.1, total=2000)
'''

import argparse
parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', default='None', type=str)
parser.add_argument('--input_size', default=-1, type=int)
parser.add_argument('--n_cluster', default=-1, type=int)
parser.add_argument('--hidden_dim', default=-1, type=int)
parser.add_argument('--shuffle', action='store_true', help='Run or not')
parser.add_argument('--pretrain', action='store_true', help='Run or not')
parser.add_argument('--pretrain_batchsize', default=-1, type=int)
parser.add_argument('--pretrain_epoch', default=-1, type=int)
parser.add_argument('--pretrain_lr', default=-1, type=float)
parser.add_argument('--pretrain_optim', default='None', type=str)
parser.add_argument('--noise', default=-1, type=float)
parser.add_argument('--trans', action='store_true', help='Run or not')
parser.add_argument('--num_workers', default=-1, type=int)
parser.add_argument('--net', default='None', type=str)
parser.add_argument('--channel', default=-1, type=int)
parser.add_argument('--wide', default=-1, type=int)
parser.add_argument('--Acc', default=-1, type=float)

parser.add_argument('--cluster_batchsize', default=-1, type=int)
parser.add_argument('--cluster_epoch', default=-1, type=int)
parser.add_argument('--cluster_lr', default=-1, type=float)
parser.add_argument('--change', default=-1, type=int)
parser.add_argument('--m', default=-1, type=int)
parser.add_argument('--divide', default=-1, type=float)
parser.add_argument('--total', default=-1, type=int)
parser.add_argument('--cluster_optim', default='None', type=str)
parser.add_argument('--manifold_lr', default=-1, type=float)
parser.add_argument('--plot', action='store_true', help='Run or not')
parser.add_argument('--times', default=1, type=int)
args = parser.parse_args()

Data = Data_style(name=args.name, input_size=args.input_size, n_cluster=args.n_cluster, hidden_dim=args.hidden_dim, shuffle=args.shuffle,
                  pretrain=args.pretrain, pretrain_batchsize=args.pretrain_batchsize, pretrain_epoch=args.pretrain_epoch,
                  pretrain_lr=args.pretrain_lr, pretrain_optim=args.pretrain_optim, noise=args.noise, trans=args.trans,
                  num_workers=args.num_workers, net=args.net, channel=args.channel, wide=args.wide, manifold_lr=args.manifold_lr,
                  cluster_batchsize=args.cluster_batchsize, cluster_epoch=args.cluster_epoch, cluster_lr=args.cluster_lr,
                  change=args.change, m=args.m, divide=args.divide, total=args.total, cluster_optim=args.cluster_optim)

Data.show_para()
#SetSeed(0)
train_loader = find_data(Data, train_mode=True)
#val_loader   = find_data(Data, train_mode=False)

if Data.net == 'CNN_VAE2':
    lgc = LGC_CNN(channel_in=Data.channel, z=Data.hidden_dim, cluster_num=Data.n_cluster, wide=Data.wide).to(device)
elif Data.net == 'VAE':
    lgc = LGC(hidden=Data.hidden_dim, input_size=Data.input_size, cluster=Data.n_cluster).to(device)
manifold = lgc_loss(change=Data.change,  n_cluster=Data.n_cluster, hidden_dim=Data.hidden_dim, eps=0.00001, total=Data.total).to(device)


if Data.net == 'CNN_VAE2':
    pretrain_dir = '../pretrain_cnn_model/pre_cnn_'
elif Data.net == 'VAE':
    pretrain_dir = '../pretrain_model/pre_vae_'
pretrain_model = Data.name+'_Acc_'+str(round(args.Acc, 2))+'_hidim_'+str(Data.hidden_dim)+'_'+str(Data.pretrain_optim)\
                 +'_lr_'+str(Data.pretrain_lr)+'_n_work_'+str(Data.num_workers)\
                 +'_noise_'+str(Data.noise)+'_btsize_'+str(Data.pretrain_batchsize)+'_epoh_'+str(Data.pretrain_epoch) + '.pt' #str(Data.num_workers)\
#if Data.net == 'CNN_VAE2':
#    pretrain_dir = '../pretrain_cnn_model/'
#elif Data.net == 'VAE':
#    pretrain_dir = '../pretrain_model/pre_vae_'
#pretrain_model = 'FC_Mnist_Ac_97.26_Nm_93.91_Ar_94.05_m_10_divid_0.1_chang_2_clr_0.001_mlr_0.0001_btsize_1000_epoh_5000_Fr_Mnist_Acc_91.62_hidim_10_Adam_lr_0.0001_n_work_0_noise_0.2_btsize_256_epoh_200.pt'

fig_dir = '../f_fig/' + str(args.times)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

path = pretrain_dir + pretrain_model
print('\nloading pretrain weight from ', path, '\n')

lgc.load_model(path)
#lgc.Global_share_para()

if Data.cluster_optim == 'SGD':
    bias_params = filter(lambda x: ('bias' in x[0]) and (x[1].requires_grad), lgc.named_parameters())
    bias_params = list(map(lambda x: x[1], bias_params))
    nonbias_params = filter(lambda x: ('bias' not in x[0]) and (x[1].requires_grad), lgc.named_parameters())
    nonbias_params = list(map(lambda x: x[1], nonbias_params))
    optimizer = optim.SGD([{'params': bias_params, 'lr': Data.cluster_lr}, {'params': nonbias_params}],
                          lr=Data.cluster_lr, momentum=0.9, weight_decay=0.0, nesterov=True)
elif Data.cluster_optim == 'Adam':
    optimizer = optim.Adam([{'params': lgc.parameters()}, {'params': manifold.mu}], lr=Data.cluster_lr)

#data = torch.Tensor(dataset.x).to(device)
if Data.net == 'VAE':
    print('encoder1: ', lgc.encoder1, '\nmanifold: ', lgc.encoder2, '\nclustering: ', lgc.global_, '\n')
elif Data.net == 'CNN_VAE2':
    print('encoder: ', lgc.encoder, '\nclustering: ', lgc.global_, '\n')
kmeans0 = KMeans(Data.n_cluster, n_init=30)
kmeans1 = KMeans(Data.n_cluster, n_init=2)

plot_AD = []
plot_AC = []
plot_Nmi = []
plot_Ari = []
plot_stage = []
plot_AK = []
plot_stage_num = []
plot_loss = []

record = Show(pretrain=False)
stage = 'global2'
old_mu = torch.Tensor(Data.n_cluster, Data.hidden_dim).to(device)
local_end = 0
global1_end = 0
global2_end = -1
small_epoch = 0
stage_num = 0
first_mu = torch.Tensor(Data.n_cluster, Data.hidden_dim).to(device)
kmeans_mu = torch.Tensor(Data.n_cluster, Data.hidden_dim).to(device)
last_loss = torch.zeros(1).to(device)
for epoch in range(Data.cluster_epoch):
    total_loss = 0

    small_batch = []
    X1 = []
    H = []
    small_batch_c = []
    Y = []
    for batch_idx, (x, y) in enumerate(train_loader):
        if Data.net == 'CNN_VAE2':
            x = x.to(device).view(-1, Data.channel, Data.wide, Data.wide)
        elif Data.net == 'VAE':
            x = x.to(device).view(-1, Data.input_size)
        z = lgc(x, local=True)
        h = lgc.cluster_indice(z) + manifold.eps
        z_c = h#manifold.softmax(h)
        #X1_ = lgc.decoder(z)
        #X1.append(X1_.data)
        H.append(h.data)
        Y.append(y.data)
        small_batch.append(z.data)
        small_batch_c.append(z_c.data)
        #X.append(x.data)
    z = torch.cat(small_batch, dim=0)
    #X = torch.cat(X, dim=0)
    #X1 = torch.cat(X1, dim=0)
    H = torch.cat(H, dim=0).T
    p_z = z
    y_c = torch.cat(small_batch_c, dim=0)
    Y = torch.cat(Y, dim=0).to(device)

    if small_epoch == 0 or stage == 'local':
        if epoch == 0:
            y_pred = kmeans0.fit_predict(z.data.cpu().numpy())
        else:
            y_pred = kmeans1.fit_predict(z.data.cpu().numpy())
    if epoch == 0:
        first_mu.data.copy_(torch.Tensor(kmeans0.cluster_centers_))
        manifold.mu.data.copy_(torch.Tensor(kmeans0.cluster_centers_))
    if small_epoch == 0:
        Z = z

    if epoch==0:
        kmeans_mu.data.copy_(torch.Tensor(kmeans0.cluster_centers_))
    else:
        kmeans_mu.data.copy_(torch.Tensor(kmeans1.cluster_centers_))

    old_mu.data.copy_(manifold.mu.data)

    ''''''
    # Copy the mu from X'H.T
    if stage == 'global1':
        #y_c = (y_c - y_c.min(dim=1, keepdim=True)[0]) / (y_c.max(dim=1, keepdim=True)[0] - y_c.min(dim=1, keepdim=True)[0])
        y_c = y_c / torch.sum(y_c, dim=1, keepdim=True)
        new_center = z.T @ (y_c / torch.sum(y_c, dim=0, keepdim=True))
        manifold.mu.data.copy_(new_center.T.data)

        #if small_epoch == 0:
        #    torch.nn.init.kaiming_normal_(lgc.global_[0].weight, a=math.sqrt(5))
        #    torch.nn.init.kaiming_normal_(lgc.global_[2].weight, a=math.sqrt(5))

        #    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(lgc.global_[0].weight)
        #    bound = 1 / math.sqrt(fan_in)

        #    torch.nn.init.uniform_(lgc.global_[0].bias, -bound, bound)
        #    torch.nn.init.uniform_(lgc.global_[2].bias, -bound, bound)
    ''''''

    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - manifold.mu) ** 2, dim=2) / 1)
    q = q / torch.sum(q, dim=1, keepdim=True)
    p = q ** 2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)

    #print(manifold.mu)
    #if stage == 'local' and small_epoch == 0:
    #    Z = z

    _, _, _, D = manifold.compute_S_(z, small_epoch)

    if Y is not None:
        # y_pred = torch.argmax(D3, dim=1).data.cpu().numpy()
        # y_pred = kmeans.fit_predict(z.data.cpu().numpy())
        _, AK = acc(Y.cpu().numpy(), y_pred)
        y_pred1 = torch.argmax(p, dim=1).data.cpu().numpy()
        _, AD = acc(Y.cpu().numpy(), y_pred1)
        _, AC = acc(Y.cpu().numpy(), torch.argmax(y_c, dim=1).data.cpu().numpy())
        AK *= 100
        AD *= 100
        AC *= 100
        Nmi = nmi(Y.cpu().numpy(), y_pred1) * 100
        Ari = ari(Y.cpu().numpy(), y_pred1) * 100

    for batch_idx, (x, y) in enumerate(train_loader):
        if Data.net == 'CNN_VAE2':
            x = x.to(device).view(-1, Data.channel, Data.wide, Data.wide)
        elif Data.net == 'VAE':
            x = x.to(device).view(-1, Data.input_size)
        zbatch = Z[batch_idx * Data.cluster_batchsize: min((batch_idx + 1) * Data.cluster_batchsize, Data.total)]
        Dbatch = D[batch_idx * Data.cluster_batchsize: min((batch_idx + 1) * Data.cluster_batchsize, Data.total)]
        pbatch = p[batch_idx * Data.cluster_batchsize: min((batch_idx + 1) * Data.cluster_batchsize, Data.total)]
        #hbatch = manifold.H[:, batch_idx * Data.cluster_batchsize: min((batch_idx + 1) * Data.cluster_batchsize, Data.total)]

        #Debug
        if(torch.isnan(pbatch.sum())):
            print('------------------ pbatch has nan -----------------------')
        if(torch.isnan(Dbatch.sum())):
            print('------------------ Dbatch has nan -----------------------')
        if(torch.isnan(zbatch.sum())):
            print('------------------ zbatch has nan -----------------------')


        if stage == 'local':
            optimizer = optim.Adam([{'params': lgc.Local.parameters()}], lr=Data.manifold_lr)
            z = lgc(x, local=True)
            zbatch = Variable(zbatch)
            keep_relation = manifold.Similarity_Mantain_Norm(zbatch, z, batch_idx, Data.cluster_batchsize, Data.total, epoch, Data.m, Data.divide, int(stage_num/3), decay=1.0)
            #loss = (1 - (epoch / Data.cluster_epoch) ** 1) * keep_relation
            loss = keep_relation
        elif stage == 'global1':
            #optimizer = optim.Adam([{'params': manifold.mu}, {'params': lgc.global_.parameters()}, {'params': manifold.W}, {'params': manifold.H}], lr = 0.001)
            #z = lgc(x, local=True)
            #z2 = lgc.cluster_indice(z)

            ''' Solution 1  Find mu by Dec loss ''' '''
            pbatch = Variable(pbatch)
            q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - manifold.mu) ** 2, dim=2) / 1)
            q = q / torch.sum(q, dim=1, keepdim=True)
            loss = manifold.loss(q, q, pbatch, pbatch, one=False, pur=True) + manifold.CrossEntropyLoss(z2, torch.argmax(Dbatch, dim=1)) + manifold.Lmc_Lmc_r(z) + manifold.Norm_U()
            '''

            ''' Solution 2  Find mu by Our loss ''''''
            optimizer = optim.Adam([{'params': manifold.mu}, {'params': lgc.global_.parameters()}], lr=0.001)
            z = lgc(x, local=True)
            z2 = lgc.cluster_indice(z)
            Dbatch = Variable(Dbatch)
            Sbatch = manifold.compute_batch(z, batch_idx, Data.cluster_batchsize, Data.total, epoch)
            loss = manifold.loss(Sbatch, Sbatch, Dbatch, Dbatch, one=False, pur=False) + manifold.CrossEntropyLoss(z2, torch.argmax(Dbatch, dim=1)) + manifold.Lmc_Lmc_r(z) + manifold.Norm_U()
            '''

            ''' Solution 3 Find mu by new loss ''' '''
            if small_epoch == 0:
                manifold.init_G(D, y_pred)
            alpha = 0.1
            rho = 0.5
            manifold._NMF(z, alpha, rho)
            total_loss = torch.norm(z.T - manifold.F @ manifold.G.T)
            break'''

            ''' Solution 4 Find mu by new loss and our loss
            optimizer = optim.Adam([{'params': manifold.mu}, {'params': lgc.global_.parameters()}], lr=0.001)
            z1 = lgc(x, local=True)
            z2 = lgc.cluster_indice(z1)
            alpha = 0.1
            rho = 0.5
            #Dbatch = Variable(Dbatch)
            #Sbatch = manifold.compute_batch(z, batch_idx, Data.cluster_batchsize, Data.total, epoch)
            loss = manifold._NMF(z, alpha, rho)#manifold.loss(Sbatch, Sbatch, Dbatch, Dbatch, one=False, pur=False) + manifold.CrossEntropyLoss(z2, torch.argmax(Dbatch, dim=1)) + manifold.Lmc_Lmc_r(z2) + manifold.Norm_U()
            total_loss += loss.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break
            '''

            ''' Solution 5 Find mu by new deep matrix factorization''' '''
            optimizer = optim.Adam([{'params': lgc.decoder.parameters()}, {'params': lgc.global_.parameters()}], lr=0.001)
            z1 = lgc(x, local=True)
            z1 = add_noise(z1, Data.noise)
            H = lgc.cluster_indice(z1)+manifold.eps #manifold.softmax(lgc.cluster_indice(z1)).T
            #H = (H - H.min(dim=1, keepdim=True)[0]) / (H.max(dim=1, keepdim=True)[0] - H.min(dim=1, keepdim=True)[0])
            #H = H.T / torch.sum(H.T, dim=0, keepdim=True)
            x1 = lgc.recon(z1)
            Reconstruction_loss = manifold.mseloss(x, x1)
            Reconstruction_loss2 = manifold.mseloss(x.T, x1.T@(H / torch.sum(H, dim=0, keepdim=True))@(H.T / (torch.sum(H.T, dim=0, keepdim=True) + manifold.eps)))
            if(torch.sum(H, dim=0, keepdim=True).min() == 0 or torch.sum(H.T, dim=0, keepdim=True).min() == 0):
                print(torch.sum(H, dim=0, keepdim=True).min(), torch.sum(H.T, dim=0, keepdim=True).min())
                print(batch_idx)
            loss = Reconstruction_loss + Reconstruction_loss2
            total_loss += loss.data
            '''''' '''

            ''' Solution 6 Find mu by my deep matrix factorization''''''
            optimizer = optim.Adam([{'params': lgc.decoder.parameters()}, {'params': lgc.global_.parameters()}],
                                   lr=0.001)
            z1 = lgc(x, local=True)
            #z1 = add_noise(z1, Data.noise)
            H = lgc.cluster_indice(z1) + manifold.eps  # manifold.softmax(lgc.cluster_indice(z1)).T
            # H = (H - H.min(dim=1, keepdim=True)[0]) / (H.max(dim=1, keepdim=True)[0] - H.min(dim=1, keepdim=True)[0])
            # H = H.T / torch.sum(H.T, dim=0, keepdim=True)
            #x1 = lgc.recon(z1)
            #Reconstruction_loss = manifold.mseloss(x, x1)
            Reconstruction_loss2 = manifold.mseloss(z1.T, z1.T @ (H / torch.sum(H, dim=0, keepdim=True)) @ (
                        H.T / (torch.sum(H.T, dim=0, keepdim=True) + manifold.eps)))
            if (torch.sum(H, dim=0, keepdim=True).min() == 0 or torch.sum(H.T, dim=0, keepdim=True).min() == 0):
                print(torch.sum(H, dim=0, keepdim=True).min(), torch.sum(H.T, dim=0, keepdim=True).min())
                print(batch_idx)
            loss = Reconstruction_loss2
            total_loss += loss.data
            '''''''''

            ''' Solution 7 Find mu by my deep kmeans matrix faction''''''
            optimizer = optim.Adam([{'params': lgc.decoder.parameters()}, {'params': lgc.global_.parameters()}],
                                   lr=0.001)
            z1 = lgc(x, local=True)
            # z1 = add_noise(z1, Data.noise)
            H = lgc.cluster_indice(z1) + manifold.eps  # manifold.softmax(lgc.cluster_indice(z1)).T
            # H = (H - H.min(dim=1, keepdim=True)[0]) / (H.max(dim=1, keepdim=True)[0] - H.min(dim=1, keepdim=True)[0])
            # H = H.T / torch.sum(H.T, dim=0, keepdim=True)
            # x1 = lgc.recon(z1)
            # Reconstruction_loss = manifold.mseloss(x, x1)
            H = H / torch.sum(H, dim=1, keepdim=True)
            center = (z1.T @ (H / torch.sum(H, dim=0, keepdim=True))).T
            Reconstruction_loss2 = torch.min((torch.sum((z1.unsqueeze(1) - center)**2, dim=2)), dim=1)[0]
            #Reconstruction_loss3 = manifold.mseloss(z1.T, z1.T @ (H / torch.sum(H, dim=0, keepdim=True)) @ (
                    H.T / (torch.sum(H.T, dim=0, keepdim=True) + manifold.eps)))
            if (torch.sum(H, dim=0, keepdim=True).min() == 0 or torch.sum(H.T, dim=0, keepdim=True).min() == 0):
                print(torch.sum(H, dim=0, keepdim=True).min(), torch.sum(H.T, dim=0, keepdim=True).min())
                print(batch_idx)
            loss = Reconstruction_loss2.sum() #+ 0.5 * Reconstruction_loss3
            total_loss += loss.data
            '''''''''

            ''' Solution 7 Find mu by my deep kmeans matrix faction'''
            optimizer = optim.Adam([{'params': lgc.global_.parameters()}], lr=Data.cluster_lr)
            z1 = lgc(x, local=True)
            Dbatch = Variable(Dbatch)
            # z1 = add_noise(z1, Data.noise)
            z2 = lgc.global_(z1)
            H = lgc.softmax(z2) #+ manifold.eps  # manifold.softmax(lgc.cluster_indice(z1)).T
            # H = (H - H.min(dim=1, keepdim=True)[0]) / (H.max(dim=1, keepdim=True)[0] - H.min(dim=1, keepdim=True)[0])
            # H = H.T / torch.sum(H.T, dim=0, keepdim=True)
            #x1 = lgc.recon(z1)
            #Reconstruction_loss = manifold.mseloss(x, x1)
            #Reconstruction_loss3 = manifold.mseloss(x.T, x1.T @ (H / torch.sum(H, dim=0, keepdim=True)) @ (H.T / (torch.sum(H.T, dim=0, keepdim=True) + manifold.eps)))

            H = H / torch.sum(H, dim=1, keepdim=True)
            center = (z1.T @ (H / torch.sum(H, dim=0, keepdim=True))).T
            Reconstruction_loss2 = torch.min((torch.sum((z1.unsqueeze(1) - center) ** 2, dim=2)), dim=1)[0]
            #Reconstruction_loss3 = manifold.mseloss(z1.T, z1.T @ (H / torch.sum(H, dim=0, keepdim=True)) @ (H.T / (torch.sum(H.T, dim=0, keepdim=True) + manifold.eps)))
            if (torch.sum(H, dim=0, keepdim=True).min() == 0 or torch.sum(H.T, dim=0, keepdim=True).min() == 0):
                print(torch.sum(H, dim=0, keepdim=True).min(), torch.sum(H.T, dim=0, keepdim=True).min())
                print(batch_idx)
            loss = Reconstruction_loss2.sum() + manifold.Lmc_Lmc_r(H) + manifold.CrossEntropyLoss(z2, torch.argmax(Dbatch, dim=1)) #Reconstruction_loss2.sum() #+ + 0.5 * Reconstruction_loss3#+ manifold.Lmc_Lmc_r(H) #+ manifold.Lmc_Lmc_r(H)+ manifold.CrossEntropyLoss(z2, torch.argmax(Dbatch, dim=1)) #+ 0.5 * Reconstruction_loss3#+ manifold.CrossEntropyLoss(z, torch.argmax(Dbatch, dim=1)) #+ 0.5 * Reconstruction_loss3#Reconstruction_loss + Reconstruction_loss3  # + 0.5 * Reconstruction_loss3
            #total_loss += loss.data
            ''''''

        elif stage == 'global2':
            #break
            '''
            optimizer = optim.Adam([{'params': lgc.encoder2.parameters()}], lr=Data.cluster_lr)
            z = lgc(x, local=True)
            z2 = lgc.cluster_indice(z)
            Dbatch = Variable(Dbatch)
            Sbatch = manifold.compute_batch(z, batch_idx, Data.cluster_batchsize, Data.total, small_epoch)
            loss = manifold.loss(Sbatch, Sbatch, Dbatch, Dbatch, one=False,
                                 pur=False)  # + manifold.CrossEntropyLoss(z, torch.argmax(Dbatch, dim=1)) + manifold.Lmc_Lmc_r(z)
            # print("Crossentropy = {:.2f}".format(manifold.CrossEntropyLoss(z, torch.argmax(Dbatch, dim=1)).data * 100))
            '''

            ''' New Method '''
            if Data.net == 'VAE':
                optimizer = optim.Adam([{'params': manifold.mu}, {'params': lgc.encoder1.parameters()}, {'params': lgc.encoder2.parameters()}], lr=Data.cluster_lr)
            elif Data.net == 'CNN_VAE2':
                optimizer = optim.Adam([{'params': manifold.mu}, {'params': lgc.encoder.parameters()}],  lr=0.0001)
            z = lgc(x, local=True)
            pbatch = Variable(pbatch)
            q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - manifold.mu) ** 2, dim=2) / 1)
            q = q / torch.sum(q, dim=1, keepdim=True)
            loss = manifold.loss(q, q, pbatch, pbatch, one=False, pur=True)

        total_loss += loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ### show the procedure fig
    if args.plot and small_epoch == 0 or (stage=='global2' and small_epoch == 100):
        mark = torch.ones(10)
        for center in range(10):
            mark[center] = center
        show_picture = torch.cat([p_z[0:1000], manifold.mu], dim=0)
        if device.__str__() == 'cuda':
            label = torch.cat([Y[0:1000].int(), mark.cuda().int()], dim=0)
        else:
            label = torch.cat([Y[0:1000], mark.int()], dim=0)
        # data_umap = umap.UMAP().fit_transform(show_picture.cpu().detach().numpy())
        data_tsne = TSNE().fit_transform(show_picture.cpu().detach().numpy())


        plt.rcParams['figure.figsize'] = (15, 15)  ## 显示的大小
        if stage == 'local':
            name = '../f_fig/' + str(args.times) + '/' + Data.name + '_' + stage + ' ' + str(stage_num) + ': ' + str(epoch - global1_end) \
                   + '_epoch:' + str(epoch + 1)
            #name = str(epoch + 1)
        elif stage == 'global1':
            name = '../f_fig/' + str(args.times) + '/' + Data.name + '_' + stage + ' ' + str(stage_num) + ': ' + str(epoch - local_end) \
                   + '_epoch:' + str(epoch + 1)
            #name = str(epoch + 1)
        elif stage == 'global2':
            name = '../f_fig/' + str(args.times) + '/' + Data.name + '_' + stage + ' ' + str(stage_num) + ': ' + str(epoch - global1_end) \
                   + '_epoch:' + str(epoch + 1)
            #name = str(epoch + 1)
        #print('save embedding npy:', name)
        fig = plot_embedding(data_tsne, label, 't-sne test', str(stage_num))
        show_picture = show_picture.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        np.save(name+' X', show_picture)
        np.save(name + ' Y', label)
        name = name + '_Ac_' + str(round(AD, 2)) + '_Nm_' + str(round(Nmi, 2)) + '_Ar_' \
                + str(round(Ari, 2)) + '_m_' + str(Data.m) + '_divid_' + str(Data.divide) + '_chang_' + str(Data.change) + '_clr_' \
                + str(Data.cluster_lr) + '_mlr_' + str(Data.manifold_lr) + '_btsize_' + str(Data.cluster_batchsize) + '_epoh_' \
                + str(Data.cluster_epoch) + '_Fr_' + pretrain_model.replace('.pt', '_') + '.jpg'
        print('save embedding jpg:', name)
        plt.savefig(name)
        plt.close()
        '''
        show_picture = show_picture.cpu().detach().numpy()
        label = label.cpu().numpy()
        np.save(name, [show_picture, label])
        if small_epoch == 0 or (stage=='global2' and small_epoch >=15):
            show_mu = manifold.mu.cpu().detach().numpy()
            show_z = p_z.cpu().detach().numpy()
            show_Y = Y.cpu().detach().numpy()
            np.save(name + '_Whole_', [show_picture, label, show_mu, show_z, show_Y])
        '''
        ### plt.savefig() 一定在前，不然将会保存空白的图像

    plot_AD.append(AD)
    plot_AC.append(AC)
    plot_Nmi.append(Nmi)
    plot_Ari.append(Ari)
    plot_stage.append(stage)
    plot_AK.append(AK)
    plot_stage_num.append(stage_num)
    plot_loss.append(total_loss.detach().cpu().numpy()/(batch_idx + 1))

    if stage == 'local':
        print('\033[0;30;46mEpoch: [{}\{}\{}\{}], Manifold_loss: {:.8f}\033[0m'.format(Data.cluster_epoch, stage_num, epoch - global1_end, epoch+1,
                                                                        total_loss/(batch_idx + 1)))
        print('************\t\tmu - old_mu = ', torch.sum((old_mu - manifold.mu).abs()).data, '\t\t***************')
        print('************\t\tmu - first_mu = ', torch.sum((first_mu - manifold.mu).abs()).data, '\t\t***************')
        print('************\t\tkmeans_mu - first_mu = ', torch.sum((first_mu - kmeans_mu).abs()).data,
              '\t\t***************')
        print('************\t\tkmeans_mu - mu = ', torch.sum((manifold.mu - kmeans_mu).abs()).data,
              '\t\t***************')
        record.append(total_loss, AK, AD, Nmi, Ari, 'manifold')
        local_end = epoch
        small_epoch = epoch - global1_end
        if small_epoch >= 5:
            stage = 'global1'
            stage_num += 1
            small_epoch = 0
    elif stage == 'global1':
        print('\033[0;30;46mEpoch: [{}\{}\{}\{}], Cluster_loss1: {:.8f}\033[0m'.format(Data.cluster_epoch, stage_num, epoch - local_end, epoch+1,
                                                                               total_loss/(batch_idx + 1)))
        '''
        # new_center = Real_Center(manifold.W, manifold.H, y_pred1, Data.n_cluster)
        manifold.mu.data.copy_(manifold.F.T)
        new_center = torch.zeros((Data.n_cluster, Data.n_cluster)).to(device)
        for i in range(Data.n_cluster):
            new_center[i, :] = torch.mean(manifold.F @ manifold.G.T[:, y_pred1 == i], 1).T
        manifold.mu.data.copy_(new_center)
        '''
        print('************\t\tmu - old_mu = ', torch.sum((old_mu - manifold.mu).abs()).data, '\t\t***************')
        print('************\t\tmu - first_mu = ', torch.sum((first_mu - manifold.mu).abs()).data, '\t\t***************')
        print('************\t\tkmeans_mu - first_mu = ', torch.sum((first_mu - kmeans_mu).abs()).data, '\t\t***************')
        print('************\t\tkmeans_mu - mu = ', torch.sum((manifold.mu - kmeans_mu).abs()).data, '\t\t***************')
        record.append(total_loss, AK, AD, Nmi, Ari, 'cluster')
        global1_end = epoch
        small_epoch = epoch - local_end

        if ((last_loss - total_loss).abs() / (batch_idx + 1)) < 0.1:
            if stage_num < 5:
                stage = 'local'
                stage_num += 1
                small_epoch = 0
            else:
                stage = 'global2'
                stage_num += 1
                small_epoch = 0
        last_loss.data.copy_(total_loss)
    elif stage == 'global2':
        print('\033[0;30;46mEpoch: [{}\{}\{}\{}], Cluster_loss2: {:.8f}\033[0m'.format(Data.cluster_epoch, stage_num, epoch - global1_end, epoch+1,
                                                                                total_loss/(batch_idx + 1) ))
        print('************\t\tmu - old_mu = ', torch.sum((old_mu - manifold.mu).abs()).data, '\t\t***************')
        print('************\t\tmu - first_mu = ', torch.sum((first_mu - manifold.mu).abs()).data, '\t\t***************')
        print('************\t\tkmeans_mu - first_mu = ', torch.sum((first_mu - kmeans_mu).abs()).data,
              '\t\t***************')
        print('************\t\tkmeans_mu - mu = ', torch.sum((manifold.mu - kmeans_mu).abs()).data,
              '\t\t***************')
        #print(manifold.mu.grad.sum(), torch.sum(old_mu - manifold.mu))
        record.append(total_loss, AK, AD, Nmi, Ari, 'cluster')
        global2_end = epoch
        small_epoch = epoch - global1_end
        if small_epoch > 70:
            break
            #stage = 'local'
            #stage_num += 1
            #small_epoch = 0

    print('AK = %.4f\tAD = %.4f\tAC = %.4f\tNmi = %.4f\tAri = %.4f\t\n' % (AK, AD, AC, Nmi, Ari))

if Data.net == 'VAE':
    final_dir = '../final_model/F_DEC_'
elif Data.net == 'CNN_VAE2':
    final_dir = '../final_CNN_model/FC_DEC_'
final_model = Data.name+'_Ac_'+str(round(AD,2))+'_Nm_'+str(round(Nmi,2))+'_Ar_'+str(round(Ari,2))+'_m_'\
              +str(Data.m)+'_divid_'+str(Data.divide)+'_chang_'+str(Data.change) + '_clr_'+str(Data.cluster_lr)\
              +'_mlr_' + str(Data.manifold_lr) +'_btsize_'+str(Data.cluster_batchsize)+'_epoh_'+str(Data.cluster_epoch) \
              +'_Fr_' + pretrain_model
path = final_dir + final_model
print('\nsaving model to... ', path)
torch.save(lgc.state_dict(), path)
print('\nsaving score to ...')
scipy.io.savemat('../f_fig/' + str(args.times) + '/' + Data.name + '.mat', {'AD':plot_AD, 'AC':plot_AC, 'Nmi':plot_Nmi, 'Ari':plot_Ari, 'stage': plot_stage, 'AK': plot_AK, 'stage_num': plot_stage_num, 'loss': plot_loss})
record.plot(path.replace('.pt', ''), args.plot)
print('\nOver!')