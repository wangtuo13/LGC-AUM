import sys
#sys.path.append(r'/home/wangtuo/Deep_cluster/dataset')
sys.path.append(r'E:\Data')
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
from load_data import *

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

Data.show_para()

train_loader = find_data(Data, train_mode=True)
val_loader   = find_data(Data, train_mode=False)

autoencoder = AutoEncoder(hidden=Data.hidden_dim, input_size=Data.input_size).to(device)
print('\nencoder1: ', autoencoder.encoder1, '\nencoder2: ', autoencoder.encoder2, '\ndecoder: ', autoencoder.decoder, '\n')

bias_params = filter(lambda x: ('bias' in x[0]) and (x[1].requires_grad), autoencoder.named_parameters())
bias_params = list(map(lambda x: x[1], bias_params))
nonbias_params = filter(lambda x: ('bias' not in x[0]) and (x[1].requires_grad), autoencoder.named_parameters())
nonbias_params = list(map(lambda x: x[1], nonbias_params))
optimizer = optim.SGD([{'params': bias_params, 'lr': Data.pretrain_lr}, {'params': nonbias_params}],
                              lr=Data.pretrain_lr, momentum=0.9, weight_decay=0.0, nesterov=True)

#data = torch.Tensor(dataset.x).to(device)

for epoch in range(Data.pretrain_epoch):
    total_loss = 0

    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device).view(-1, Data.input_size)
        x_noise = add_noise(x, Data.noise)
        x_rec = autoencoder(x_noise)
        loss = F.mse_loss(x_rec, x)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('\033[0;30;46mEpoch: [{}/{}], MSE_loss: {:.8f}\033[0m'.format(epoch + 1, Data.pretrain_epoch,
                                                                        total_loss / (batch_idx + 1)))

path = './pretrain_model/pretrain_'+Data.name+'_lr_'+str(Data.pretrain_lr)+'_hiddendim_'+str(Data.hidden_dim)+'_noise_'+str(Data.noise)+'_batchsize_'+str(Data.pretrain_batchsize)+'_epoch_'+str(Data.pretrain_epoch) + '.pt'
print('\nsaving pretrain weight to... ', path)
torch.save(autoencoder.state_dict(), path)
print('\nOver!')