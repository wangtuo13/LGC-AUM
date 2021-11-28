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

def z_Y_AC_Nmi_Ari(lgc, train_loader, Data):
    small_batch = []
    small_batch_c = []
    Y = []
    for batch_idx, (x, y) in enumerate(train_loader):
        if Data.net == 'CNN_VAE2':
            x = x.to(device).view(-1, Data.channel, Data.wide, Data.wide)
        elif Data.net == 'VAE':
            x = x.to(device).view(-1, Data.input_size)
        z = lgc(x, local=True)
        z_c = lgc.cluster_indice(z) #+ manifold.eps
        Y.append(y.data)
        small_batch.append(z.data)
        small_batch_c.append(z_c.data)
    z = torch.cat(small_batch, dim=0)
    y_c = torch.cat(small_batch_c, dim=0)
    Y = torch.cat(Y, dim=0).to(device)

    y_pred1 = torch.argmax(y_c, dim=1).data.cpu().numpy()
    _, AC = acc(Y.cpu().numpy(), torch.argmax(y_c, dim=1).data.cpu().numpy())
    AC *= 100
    Nmi = nmi(Y.cpu().numpy(), y_pred1) * 100
    Ari = ari(Y.cpu().numpy(), y_pred1) * 100
    return z, Y, AC, Nmi, Ari

def plot_whole_embedding(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], '.',
                 color=plt.cm.Set1((label.cpu().numpy()[i] / 10)),  ## color=plt.cm.Set1(np.random.randint(10))
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    return fig

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
SetSeed(0)
train_loader = find_data(Data, train_mode=True)
#val_loader   = find_data(Data, train_mode=False)

if Data.net == 'CNN_VAE2':
    lgc = LGC_CNN(channel_in=Data.channel, z=Data.hidden_dim, cluster_num=Data.n_cluster, wide=Data.wide).to(device)
elif Data.net == 'VAE':
    lgc = LGC(hidden=Data.hidden_dim, input_size=Data.input_size, cluster=Data.n_cluster).to(device)
#manifold = lgc_loss(change=Data.change,  n_cluster=Data.n_cluster, hidden_dim=Data.hidden_dim, eps=0.00001, total=Data.total).to(device)


Z, Y, _, _, _ = z_Y_AC_Nmi_Ari(lgc, train_loader, Data)
data_tsne = TSNE().fit_transform(Z.cpu().detach().numpy())
plt.rcParams['figure.figsize'] = (15, 15)
fig = plot_whole_embedding(data_tsne, Y)
plt.savefig('../f_fig/' + Data.name + ' initialize')
plt.close()



if Data.net == 'CNN_VAE2':
    pretrain_dir = '../pretrain_cnn_model/pre_cnn_'
elif Data.net == 'VAE':
    pretrain_dir = '../pretrain_model/pre_vae_'
pretrain_model = Data.name+'_Acc_'+str(round(args.Acc, 2))+'_hidim_'+str(Data.hidden_dim)+'_'+str(Data.pretrain_optim)\
                 +'_lr_'+str(Data.pretrain_lr)+'_n_work_'+str(Data.num_workers)\
                 +'_noise_'+str(Data.noise)+'_btsize_'+str(Data.pretrain_batchsize)+'_epoh_'+str(Data.pretrain_epoch) + '.pt' #str(Data.num_workers)\
path = pretrain_dir + pretrain_model
print('\nloading pretrain weight from ', path, '\n')
lgc.load_model(path)

Z, Y, _, _, _ = z_Y_AC_Nmi_Ari(lgc, train_loader, Data)
data_tsne = TSNE().fit_transform(Z.cpu().detach().numpy())
plt.rcParams['figure.figsize'] = (15, 15)
fig = plot_whole_embedding(data_tsne, Y)
plt.savefig('../f_fig/' + Data.name + ' Reconstruction')
plt.close()



if Data.net == 'VAE':
    final_dir = '../final_model/'
elif Data.net == 'CNN_VAE2':
    final_dir = '../final_CNN_model/'
path = final_dir + 'FC_Fashion_Ac_67.66_Nm_69.52_Ar_55.7_m_10_divid_0.1_chang_2_clr_0.001_mlr_0.0001_btsize_1000_epoh_500_Fr_Fashion_Acc_65.05_hidim_10_Adam_lr_0.0001_n_work_0_noise_0.1_btsize_256_epoh_200.pt'
print('\nloading final weight from ', path, '\n')
lgc.load_model(path)

Z, Y, _, _, _ = z_Y_AC_Nmi_Ari(lgc, train_loader, Data)
data_tsne = TSNE().fit_transform(Z.cpu().detach().numpy())
plt.rcParams['figure.figsize'] = (15, 15)
fig = plot_whole_embedding(data_tsne, Y)
plt.savefig('../f_fig/' + Data.name + ' final')
plt.close()


print('\nOver!\n')