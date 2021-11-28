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
from CNN_VAE3 import *
from loss import *
from load_data import *
from utils import *
from sklearn.manifold import TSNE
import numpy as np
import contrastive_loss
#import umap
#import math
import math
import cv2
import scipy.io
to_pil_image = transforms.ToPILImage()

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

def train():
    # Definition

    for epoch in range(1):
        print('\nLocal Stage\n')
        # Intra-manifold preservation
        Inter_epoch = 0
        global_epoch = 0
        Intra_Max = args.Intra_Max#201
        Inter_Max = args.Inter_Max#101
        Global_Max = args.Global_Max#101
        optimizer_intra = optim.Adam([{'params': lgc.parameters()}], lr=args.Intra_lr)
        optimizer_inter = optim.Adam([{'params': lgc.encoder.inter.parameters()}, {'params': manifold.mu}],
                                     lr=args.Inter_lr)
        optimizer_global = optim.Adam([{'params': lgc.parameters()}], lr=args.global_lr)
        if args.reload == "False": # start from beginning
            args.Intra_epoch  = 1
            args.Inter_epoch  = 1
            args.global_epoch = 1
        else:
            if 0 < args.global_epoch < Global_Max:
                args.global_epoch += 1
                args.Inter_epoch = Inter_Max
                args.Intra_epoch = Intra_Max
                Intra_epoch = Intra_Max - 1
                Inter_epoch = Inter_Max - 1
                global_epoch = Global_Max - 1
                # optimizer_global.load_state_dict(checkpoint['optimizer'])
                optimizer_global = optim.Adam([{'params': lgc.parameters()}, {'params': manifold.mu}],
                                              lr=args.global_lr)
            elif 0 < args.Inter_epoch < Inter_Max:
                args.global_epoch = 1
                args.Inter_epoch += 1
                args.Intra_epoch = Intra_Max
                Intra_epoch = Intra_Max - 1
                Inter_epoch = Inter_Max - 1
                global_epoch = 0
                optimizer_inter.load_state_dict(checkpoint['optimizer'])
                optimizer_global = optim.Adam([{'params': lgc.parameters()}, {'params': manifold.mu}],
                                              lr=args.global_lr)
            else:
                args.global_epoch = 1
                args.Inter_epoch = 1
                args.Intra_epoch += 1
                Intra_epoch       = Intra_Max - 1
                Inter_epoch       = 0
                global_epoch      = 0
                optimizer_intra.load_state_dict(checkpoint['optimizer'])
                optimizer_inter = optim.Adam([{'params': lgc.encoder.inter.parameters()}, {'params': manifold.mu}], lr=args.Inter_lr)
                optimizer_global = optim.Adam([{'params': lgc.parameters()}], lr=args.global_lr)

        for Intra_epoch in range(args.Intra_epoch, Intra_Max):
            total_loss = 0
            z_pred = []
            y_ground = []
            h_pred = []
            for batch_idx, ((x, aug_samp0, aug_samp1), y) in enumerate(Intra_loader):
                if args.net == 'CNN_VAE3':
                    x = x.view(-1, args.channel, args.wide, args.wide).to(device)
                    aug_samp0 = aug_samp0.view(-1, args.channel, args.wide, args.wide).to(device)
                    aug_samp1 = aug_samp1.view(-1, args.channel, args.wide, args.wide).to(device)
                    z, h = lgc(x)
                    z = z.data
                    h = h.data
                    aug_feat0, aug_h0 = lgc(aug_samp0)
                    aug_feat1, aug_h1 = lgc(aug_samp1)

                    zbatch = z.data
                    zbatch0 = aug_feat0.data
                    zbatch1 = aug_feat1.data
                z_pred.append(z)
                y_ground.append(y)
                h_pred.append(h)
                L_intra = manifold.maintain_similarity(aug_feat0, aug_feat1, zbatch, zbatch0, zbatch1, Intra_epoch,
                                                       batch_idx, args.Intra_batchsize)
                loss = L_intra
                optimizer_intra.zero_grad()
                loss.backward()
                optimizer_intra.step()
                total_loss += loss.item()
            z_pred = torch.cat(z_pred, dim=0).data.cpu().numpy()
            y_ground = torch.cat(y_ground, dim=0).cpu().numpy()
            h_pred = torch.cat(h_pred, dim=0)

            AK, AC, Nmi, Ari = Evaluation(z_pred, h_pred, y_ground, kmeans1)

            if Intra_epoch % 25 == 0:
                save_model(args.Path, lgc, manifold, optimizer_intra, Intra_epoch, Inter_epoch, global_epoch)
                show_a_batch(args, z_pred, h_pred.data.cpu().numpy(), y_ground, manifold.mu.data.cpu().numpy(), AK, AC, Nmi, Ari, 0, Intra_epoch, Inter_epoch, global_epoch)  ### error eroor error error error error missing mu and epoch

            print("Intra_stage: %d, L_intra: %.4f, AK: %.4f, AC: %.4f, Nmi: %.4f, Ari: %.4f, AD: %.4f" % (
                Intra_epoch, total_loss / (batch_idx + 1), AK, AC, Nmi, Ari, 0))

        if args.Inter_epoch < Inter_Max:
            z_pred = []
            mid_z = []
            h_pred = []
            y_ground = []
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(Inter_loader):
                    if args.net == 'CNN_VAE3':
                        x = x.view(-1, args.channel, args.wide, args.wide).to(device)
                        z, h, midz = lgc.compute_mid_x(x)  # 每一次小batch，数据进入网络生成z
                        z_pred.append(z)
                        y_ground.append(y)
                        h_pred.append(h)
                        mid_z.append(midz)
                Z = torch.cat(z_pred, dim=0)
                z_pred = Z.data.cpu().numpy()
                mid_z = torch.cat(mid_z, dim=0)
                y_ground = torch.cat(y_ground, dim=0).cpu().numpy()
                h_pred = torch.cat(h_pred, dim=0)

            if args.Inter_epoch == 1:
                # Compute D for next training
                kmeans0.fit_predict(z_pred)
                manifold.mu.data.copy_(torch.Tensor(kmeans0.cluster_centers_))

            AK, AC, Nmi, Ari = Evaluation(z_pred, h_pred, y_ground, kmeans1)

            print('\n')

        # Inter-manifold Discrimination
        for Inter_epoch in range(args.Inter_epoch, Inter_Max):
            total_loss = 0
            active = 1

            if active:
                h_pred = []
            _, _, _, D = manifold.compute_S_(torch.Tensor(z_pred).cuda(), epoch)
            AD, Nmi, Ari = Evaluation_single(D, y_ground)
            # for batch_idx, (x, y) in enumerate(Inter_loader):
            for batch_idx in range(math.ceil(args.total/args.Inter_batchsize)):
                #x = x.view(-1, args.channel, args.wide, args.wide).to(device)
                Dbatch = D[batch_idx * args.Inter_batchsize: min((batch_idx + 1) * args.Inter_batchsize, args.total)]
                z      = Z[batch_idx * args.Inter_batchsize: min((batch_idx + 1) * args.Inter_batchsize, args.total)]
                mid_zbatch = mid_z[batch_idx * args.Inter_batchsize: min((batch_idx + 1) * args.Inter_batchsize, args.total)]

                if args.net == 'CNN_VAE3':
                    #x = x.view(-1, args.channel, args.wide, args.wide).to(device)
                    Dbatch = Dbatch.data
                    mid_zbatch = mid_zbatch.data
                    h = lgc.encoder.inter(mid_zbatch)  # 每一次小batch，数据进入网络生成z
                    #z, h = lgc(x)
                    # center = (z1.T @ (H / torch.sum(H, dim=0, keepdim=True))).T
                    # Reconstruction_loss2 = torch.min((torch.sum((z1.unsqueeze(1) - center) ** 2, dim=2)), dim=1)[0]

                    Reconstruction_loss2 = torch.min((torch.sum((z.unsqueeze(1) - manifold.mu) ** 2, dim=2)), dim=1)[0]

                    # Reconstruction_loss3 = manifold.mseloss(z1.T, z1.T @ (H / torch.sum(H, dim=0, keepdim=True)) @ (H.T / (torch.sum(H.T, dim=0, keepdim=True) + manifold.eps)))
                    if (torch.sum(h, dim=0, keepdim=True).min() == 0 or torch.sum(h.T, dim=0, keepdim=True).min() == 0):
                        print(torch.sum(h, dim=0, keepdim=True).min(), torch.sum(h.T, dim=0, keepdim=True).min())
                        print(batch_idx)


                    L_Inter = Reconstruction_loss2.sum() + manifold.CrossEntropyLoss(h, torch.argmax(Dbatch, dim=1)) + manifold.Lmc_Lmc_r(h, h)


                    # Reconstruction_loss2.sum() #+ + 0.5 * Reconstruction_loss3#+ manifold.Lmc_Lmc_r(H) #+ manifold.Lmc_Lmc_r(H)+ manifold.CrossEntropyLoss(z2, torch.argmax(Dbatch, dim=1)) #+ 0.5 * Reconstruction_loss3#+ manifold.CrossEntropyLoss(z, torch.argmax(Dbatch, dim=1)) #+ 0.5 * Reconstruction_loss3#Reconstruction_loss + Reconstruction_loss3  # + 0.5 * Reconstruction_loss3
                    #loss = Reconstruction_loss2.sum() + manifold.CrossEntropyLoss(h, torch.argmax( Dbatch, dim=1))
                    # total_loss += loss.data
                    h_pred.append(h)

                    optimizer_inter.zero_grad()
                    L_Inter.backward()
                    optimizer_inter.step()
                    total_loss += L_Inter.item()

            h_pred = torch.cat(h_pred, dim=0)
            if Inter_epoch % 25 == 0:
                save_model(args.Path, lgc, manifold, optimizer_inter, Intra_epoch, Inter_epoch, global_epoch)
                show_a_batch(args, z_pred, h_pred.data.cpu().numpy(), y_ground, manifold.mu.data.cpu().numpy(), AK, AC, Nmi, Ari, AD, Intra_epoch, Inter_epoch, global_epoch)  ### error eroor error error error error missing mu and epoch

            AC, _, _ = Evaluation_single(h_pred, y_ground)
            print("Inter_stage: %d, L_Inter: %.4f, AK: %.4f, AC: %.4f, Nmi: %.4f, Ari: %.4f, Learned AD: %.4f" % (
                Inter_epoch, total_loss / (batch_idx + 1), AK, AC, Nmi, Ari, AD))

    z_pred = []
    y_ground = []
    h_pred = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(global_loader):
            if args.net == 'CNN_VAE3':
                x = x.view(-1, args.channel, args.wide, args.wide).to(device)
                z, h = lgc(x)  # 每一次小batch，数据进入网络生成z
                z_pred.append(z)
                h_pred.append(h)
                y_ground.append(y)
        Z = torch.cat(z_pred, dim=0)
        z_pred = Z.data.cpu().numpy()
        h_pred = torch.cat(h_pred, dim=0)
        y_ground = torch.cat(y_ground, dim=0).cpu().numpy()
    manifold.mu.data.copy_((h_pred.t() @ torch.Tensor(z_pred).to(device)) / torch.sum(h_pred, dim=0, keepdim=True))
    AK, AC, Nmi, Ari = Evaluation(z_pred, h_pred, y_ground, kmeans1)
    _, _, _, D = manifold.compute_S_(torch.Tensor(z_pred).cuda(), global_epoch)
    #h_mu = (h_pred.t() @ h_pred) / torch.sum(h_pred, dim=0, keepdim=True)
    #h_mu = Variable(h_mu)
    print('\nGlobal Stage\n')
    # Global Stage
    for global_epoch in range(args.global_epoch, Global_Max):
            total_loss = 0
            active = 1

            #_, _, _, D = manifold.compute_S_(torch.Tensor(z_pred).to(device), global_epoch)
            #AD, _, _ = Evaluation_single(D, y_ground)



            q = 1.0 / (1.0 + torch.sum((torch.Tensor(z_pred).cuda().unsqueeze(1) - manifold.mu) ** 2, dim=2) / 1)
            q = q / torch.sum(q, dim=1, keepdim=True)
            p = q ** 2 / torch.sum(q, dim=0)
            p = p / torch.sum(p, dim=1, keepdim=True)
            AP,_,_ = Evaluation_single(p, y_ground)


            if active:
                z_pred = []
                h_pred = []

            for batch_idx, ((x), y) in enumerate(global_loader):
                #Dbatch = D[batch_idx * args.global_batchsize: min((batch_idx + 1) * args.global_batchsize, args.total)]
                pbatch = p[batch_idx * args.global_batchsize: min((batch_idx + 1) * args.global_batchsize, args.total)]
                pbatch = pbatch.data
                if args.net == 'CNN_VAE3':
                    x = x.view(-1, args.channel, args.wide, args.wide).to(device)
                    z, h = lgc(x)


                q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - manifold.mu) ** 2, dim=2) / 1)
                q = q / torch.sum(q, dim=1, keepdim=True)
                loss = manifold.loss(q, q, pbatch, pbatch, one=False, pur=True)


                #Dbatch = Dbatch.data
                #Sbatch = manifold.compute_batch(z, batch_idx, args.global_batchsize, args.total, global_epoch)
                #L_Global = manifold.loss(Sbatch, Sbatch, Dbatch, Dbatch, one=False, pur=False) #+ manifold.CrossEntropyLoss(Sbatch, torch.argmax(h, dim=1)) #+ manifold.Lmc_Lmc_r(h, h)

                z_pred.append(z)
                h_pred.append(h)
                optimizer_global.zero_grad()
                loss.backward()
                optimizer_global.step()
                total_loss += loss.item()
                #total_loss += loss.item()

            z_pred = torch.cat(z_pred, dim=0).data.cpu().numpy()
            h_pred = torch.cat(h_pred, dim=0)

            #if global_epoch % 25 == 0:
            #    save_model(args.Path, lgc, manifold, optimizer_global, Intra_epoch, Inter_epoch, global_epoch)
            #    show_a_batch(args, z_pred, h_pred.data.cpu().numpy(), y_ground, manifold.mu.data.cpu().numpy(), AK, AC, Nmi, Ari, AD, Intra_epoch, Inter_epoch, global_epoch)  ### error eroor error error error error missing mu and epoch

            AK, AC, _, _ = Evaluation1(z_pred, h_pred, y_ground, kmeans1, p)
            nmi, ari = Evaluation2(y_ground, p)

            print("Global_stage: %d, L_Global: %.4f, AK: %.4f, AC: %.4f, Nmi: %.4f, Ari: %.4f, Learned p: %.4f" % (
                global_epoch, total_loss / (batch_idx + 1), AK, AC, nmi, ari, AP))










































































    return AD, Nmi, Ari

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', default='None', type=str)
    parser.add_argument('--nick_name', default='None', type=str)
    parser.add_argument('--n_cluster', default=-1, type=int)
    parser.add_argument('--hidden_dim', default=-1, type=int)
    parser.add_argument('--num_workers', default=-1, type=int)
    parser.add_argument('--optim', default='None', type=str)
    parser.add_argument('--net', default='None', type=str)
    parser.add_argument('--channel', default=-1, type=int)
    parser.add_argument('--wide', default=-1, type=int)
    parser.add_argument('--m', default=-1, type=int)
    parser.add_argument('--divide', default=-1, type=float)
    parser.add_argument('--reload', default="True", type=str)
    parser.add_argument('--times', default=True, type=int)

    parser.add_argument('--Intra_batchsize', default=-1, type=int)
    parser.add_argument('--Intra_epoch', default=-1, type=int)
    parser.add_argument('--Intra_lr', default=-1, type=float)
    parser.add_argument('--Intra_Max', default=-1, type=int)

    parser.add_argument('--Inter_batchsize', default=-1, type=int)
    parser.add_argument('--Inter_epoch', default=-1, type=int)
    parser.add_argument('--Inter_lr', default=-1, type=float)
    parser.add_argument('--Inter_Max', default=-1, type=int)

    parser.add_argument('--global_batchsize', default=-1, type=int)
    parser.add_argument('--global_epoch', default=-1, type=int)
    parser.add_argument('--global_lr', default=-1, type=float)
    parser.add_argument('--Global_Max', default=-1, type=int)

    args = parser.parse_args()

    #args.Path = 'F:\cache/0_model/STL_10_OUR'# + args.name.upper()
    args.Path = '../0_model/' + args.name.upper() + '_'+args.nick_name
    if not os.path.isdir(args.Path):
        os.makedirs(args.Path)
    args.Path1 = '../1_fig/' + args.name.upper() + '_'+args.nick_name
    if not os.path.isdir(args.Path1):
        os.makedirs(args.Path1)
    show_para(args)

    print('DEC_NEW')

    SetSeed(0)
    Intra_loader, args.total  = find_data(args, train_mode=True, aug=True,  stage=0)
    Inter_loader, _  = find_data(args, train_mode=True, aug=False, stage=1)
    global_loader,_ = find_data(args, train_mode=True, aug=False, stage=2)
    # val_loader   = find_data(Data, train_mode=False)

    if args.net == 'CNN_VAE2':
        lgc = LGC_CNN(channel_in=args.channel, z=args.hidden_dim, cluster_num=args.n_cluster, wide=args.wide).to(device)
    elif args.net == 'VAE':
        lgc = LGC(hidden=args.hidden_dim, input_size=args.input_size, cluster=args.n_cluster).to(device)
    elif args.net == 'CNN_VAE3':
        lgc = LGC_CNN3(hidden=args.hidden_dim, cluster_num=args.n_cluster, net='resnet18').to(device)
    manifold = lgc_loss(n_cluster=args.n_cluster, hidden_dim=args.hidden_dim, eps=0.00001,
                        total=args.total, inter_batchsize=args.Inter_batchsize, intra_batchsize=args.Intra_batchsize).to(device)

    print("lgc.encoder.conv1: \n", lgc.encoder.conv1,
          "\nlgc.encoder.intra: \n", lgc.encoder.intra,
          "\nlgc.encoder.inter: \n", lgc.encoder.inter)

    fig_dir = '../f_fig/' + str(args.times)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    if args.reload == "True":
        load_path = args.Path + '/checkpoint_Global_' + str(args.global_epoch) + '.Inter_' + str(args.Inter_epoch) + '.Intra_' + str(args.Intra_epoch) + '.tar'
        print("Loading weights from %s"%(load_path))
        checkpoint = torch.load(load_path)
        lgc.load_state_dict(checkpoint['net'])
        manifold.load_model(checkpoint['lgc_loss'])

    if args.net == 'VAE':
        print('encoder1: ', lgc.encoder1, '\nmanifold: ', lgc.encoder2, '\nclustering: ', lgc.global_, '\n')
    elif args.net == 'CNN_VAE2':
        print('encoder: ', lgc.encoder, '\nclustering: ', lgc.global_, '\n')
    kmeans0 = KMeans(args.n_cluster, n_init=30)
    kmeans1 = KMeans(args.n_cluster, n_init=5)

    AD, Nmi, Ari = train()

    final_model = '/_Ac_' + str(round(AD, 2)) + '_Nm_' + str(round(Nmi, 2)) + '_Ar_' + str(
        round(Ari, 2)) + '_m' + str(args.m)
    path = args.Path + final_model
    print('\nsaving model to... ', path)
    torch.save(lgc.state_dict(), path)
    print('\nsaving score to ...')
    print('\nOver!')