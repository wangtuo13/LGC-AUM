import sys
#sys.path.append(r'/home/wangtuo/Deep_cluster/dataset')
#sys.path.append(r'E:\Data')
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision import transforms

#test = 'Palm'
#test = 'Mnist'
#test = 'Fashion'
#test = 'USPS'
#test = 'Coil_20'
#test = 'Yale'
#test = 'Cifar_10'

from model import *
from CNN_VAE2 import *
from load_data import *


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
parser.add_argument('--plot', action='store_true', help='Run or not')
args = parser.parse_args()

Data = Data_style(name=args.name, input_size=args.input_size, n_cluster=args.n_cluster, hidden_dim=args.hidden_dim, shuffle=args.shuffle,
                  pretrain=args.pretrain, pretrain_batchsize=args.pretrain_batchsize, pretrain_epoch=args.pretrain_epoch,
                  pretrain_lr=args.pretrain_lr, pretrain_optim=args.pretrain_optim, noise=args.noise, trans=args.trans,
                  num_workers=args.num_workers, net=args.net, channel=args.channel, wide=args.wide)



'''
if test=='Palm':
    Data = Data_style(name='Palm',      input_size=256, n_cluster=100, hidden_dim=100, shuffle=True, pretrain=True,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None)
elif test == 'Mnist':
    Data = Data_style(name='Mnist',     input_size=784, n_cluster=10, hidden_dim=10, shuffle=True, pretrain=True,
                      pretrain_batchsize=256, pretrain_epoch=800, pretrain_lr=0.01, noise=0.2, trans=None)
elif test == 'Yale':
    Data = Data_style(name='Yale',      input_size=1024, n_cluster=38, hidden_dim=38, shuffle=True, pretrain=True,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None)
elif test == 'Fashion':
    Data = Data_style(name='Fashion',   input_size=784, n_cluster=10, hidden_dim=10, shuffle=True, pretrain=True,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None)
elif test == 'USPS':
    Data = Data_style(name='USPS',      input_size=256, n_cluster=10, hidden_dim=10, shuffle=True, pretrain=True,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None)
elif test == 'Coil_20':
    Data = Data_style(name='Coil_20',   input_size=16384, n_cluster=20, hidden_dim=20, shuffle=True, pretrain=True,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None)
elif test == 'Cifar_10':
    Data = Data_style(name='Cifar_10',  input_size=3072, n_cluster=10, hidden_dim=10, shuffle=True, pretrain=True,
                      pretrain_batchsize=256, pretrain_epoch=100, pretrain_lr=0.01, noise=0.2, trans=None)
'''

Data.show_para()
SetSeed(0)

train_loader = find_data(Data, train_mode=True)
val_loader   = find_data(Data, train_mode=False)

if Data.net == 'CNN_VAE2':
    autoencoder = CNN_VAE2(channel_in=Data.channel, z=Data.hidden_dim, wide=Data.wide).to(device)
    print('\nencoder: ', autoencoder.encoder, '\ndecoder: ', autoencoder.decoder,'\n')
elif Data.net == 'VAE':
    autoencoder = AutoEncoder(hidden=Data.hidden_dim, input_size=Data.input_size).to(device)
    print('\nencoder1: ', autoencoder.encoder1, '\nencoder2: ', autoencoder.encoder2, '\ndecoder: ', autoencoder.decoder, '\n')

if Data.pretrain_optim == 'SGD':
    #bias_params = filter(lambda x: ('bias' in x[0]) and (x[1].requires_grad), autoencoder.named_parameters())
    #bias_params = list(map(lambda x: x[1], bias_params))
    #nonbias_params = filter(lambda x: ('bias' not in x[0]) and (x[1].requires_grad), autoencoder.named_parameters())
    #nonbias_params = list(map(lambda x: x[1], nonbias_params))
    optimizer = optim.SGD(autoencoder.parameters(), lr=Data.pretrain_lr, momentum=0.9)
elif Data.pretrain_optim == 'Adam':
    optimizer = optim.Adam(autoencoder.parameters(), lr=Data.pretrain_lr, weight_decay=0.)

#data = torch.Tensor(dataset.x).to(device)

kmeans = KMeans(Data.n_cluster, n_init=5)
record = Show(pretrain=True)

for epoch in range(Data.pretrain_epoch):
    total_loss = 0

    small_z = []
    Y = []
    for batch_idx, ((x, x1, _), y) in enumerate(train_loader):
        if Data.net == 'CNN_VAE2':
            x = x.to(device).view(-1, Data.channel, Data.wide, Data.wide)
            x_noise = add_noise(x, Data.noise)
            x_rec = autoencoder(x_noise, Train=True)

            x1 = x1.to(device).view(-1, Data.channel, Data.wide, Data.wide)
            #x_noise1 = add_noise(x1, Data.noise)
            x_rec1 = autoencoder(x1, Train=True)

        elif Data.net == 'VAE':
            x = x.to(device).view(-1, Data.input_size)
            x_noise = add_noise(x, Data.noise)
            x_rec = autoencoder(x_noise)

        loss = F.mse_loss(x_rec, x) + F.mse_loss(x_rec1, x) #+ F.mse_loss(x_rec1, x)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        for _, (x, y) in enumerate(val_loader):
            if Data.net == 'CNN_VAE2':
                x = x.to(device).view(-1, Data.channel, Data.wide, Data.wide)
                small_z.append(autoencoder.encode(x, Train=False).data)
                Y.append(y.data)
            elif Data.net == 'VAE':
                x = x.to(device).view(-1, Data.input_size)
                small_z.append(autoencoder.encode(x).data)
                Y.append(y.data)
        small_z = torch.cat(small_z, dim=0)
        Y = torch.cat(Y, dim=0)
        print("Current Device: ", small_z.device)
        y_pred = kmeans.fit_predict(small_z.data.cpu().numpy())
        _, Acc = acc(Y.cpu().numpy(), y_pred)

    record.append(total_loss / (batch_idx + 1), Acc*100)
    print('\033[0;30;46mEpoch: [{}/{}], MSE_loss: {:.8f}, Acc: {:.8f}\033[0m'.format(epoch + 1, Data.pretrain_epoch,
                                                                        total_loss / (batch_idx + 1), Acc*100))

    if epoch+1 == 50 or epoch+1 == 100 or epoch+1 == 150 or epoch+1 == 200 or epoch+1 == 250 or epoch+1 == 300 or epoch+1 == 350 or epoch+1 == 400 or epoch+1 == 450 or epoch+1 == 500:
        if Data.net == 'CNN_VAE2':
            pretrain_dir = '../pretrain_cnn_model/pre_cnn_'
        else:
            pretrain_dir = '../pretrain_model/pre_vae_'
        pretrain_model = Data.name + '_Acc_' + str(round(Acc * 100, 2)) + '_hidim_' + str(Data.hidden_dim) + '_' + str(Data.pretrain_optim) \
                         + '_lr_' + str(Data.pretrain_lr) + '_n_work_' + str(Data.num_workers) \
                         + '_noise_' + str(Data.noise) + '_btsize_' + str(Data.pretrain_batchsize) + '_epoh_' + str(epoch+1) + '.pt'
        path = pretrain_dir + pretrain_model
        torch.save(autoencoder.state_dict(), path)
        print('\nsaving pretrain weight to... ', path)
        record.plot(path, args.plot)

if Data.net == 'CNN_VAE2':
    pretrain_dir = '../pretrain_cnn_model/pre_cnn_'
else:
    pretrain_dir = '../pretrain_model/pre_vae_'
pretrain_model = Data.name+'_Acc_'+str(round(Acc*100, 2))+'_hidim_'+str(Data.hidden_dim)+'_'+str(Data.pretrain_optim)\
                 +'_lr_'+str(Data.pretrain_lr)+'_n_work_'+str(Data.num_workers)\
                 +'_noise_'+str(Data.noise)+'_btsize_'+str(Data.pretrain_batchsize)+'_epoh_'+str(Data.pretrain_epoch) + '.pt'
path = pretrain_dir + pretrain_model
torch.save(autoencoder.state_dict(), path)
print('\nsaving pretrain weight to... ', path)
record.plot(path, args.plot)
print('\nOver!')




