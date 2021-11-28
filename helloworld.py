'''
import argparse
parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', default='None', type=str)
args = parser.parse_args()
print(args.name)
'''

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
from dataset import PALM, Yale, JAFFE, MSRA, MNIST, Fashion, Fashion2, USPS, USPS2, Coil_20, Cifar_10, Reuters_10K
SetSeed(0)
def try_shuffle(set, labels=False):
    per = np.random.permutation(set.data.shape[0])
    if labels:
        set.labels = set.labels[per]
        set.data = set.data[per, :, :, :]
    else:
        set.data = set.data[per, :, :]
        set.targets = np.array(set.targets)[per]

transform_train=transforms.Compose([
    transforms.RandomHorizontalFlip(),   #在小型数据集上，通过随机水平翻转来实现数据增强
    transforms.RandomGrayscale(),   #将图像以一定的概率转换为灰度图像
    transforms.ToTensor(),   #数据集加载时，默认的图片格式是 numpy，所以通过 transforms 转换成 Tensor,图像范围[0, 255] -> [0.0,1.0]
    #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])   #使用公式进行归一化channel=（channel-mean）/std，因为transforms.ToTensor()已经把数据处理成[0,1],那么(x-0.5)/0.5就是[-1.0, 1.0]

transform_test=transforms.Compose([
    transforms.RandomHorizontalFlip(),   #在小型数据集上，通过随机水平翻转来实现数据增强
    transforms.RandomGrayscale(),   #将图像以一定的概率转换为灰度图像
    transforms.ToTensor(),   #数据集加载时，默认的图片格式是 numpy，所以通过 transforms 转换成 Tensor,图像范围[0, 255] -> [0.0,1.0]
    #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])   #使用公式进行归一化channel=（channel-mean）/std，因为transforms.ToTensor()已经把数据处理成[0,1],那么(x-0.5)/0.5就是[-1.0, 1.0]
set1 = torchvision.datasets.MNIST('/home/ubuntu/Data', train=True, download=True,
												  transform=transform_train)  # Transforms(size=32, s=0.5))
set2 = torchvision.datasets.MNIST('/home/ubuntu/Data', train=False, download=True,
                                                transform=transform_train)  # Transforms(size=32, s=0.5))

#trainloader=torch.utils.data.DataLoader(set1,batch_size=256,
#                                        shuffle=True,
#                                        num_workers=0)   #加载数据的时候使用几个子进程
try_shuffle(set2)


testloader=torch.utils.data.DataLoader(set2, batch_size=256, shuffle=False, num_workers=0)
#classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')



#try_shuffle(set1)

for i, x in enumerate(testloader):
    print(i, x[1][0])


