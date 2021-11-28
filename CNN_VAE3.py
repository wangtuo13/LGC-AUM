import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1, conv3x3
from typing import Type, Any, Callable, Union, List, Optional

def conv_up3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, output_padding=(dilation, dilation))

def conv_up1x1(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, output_padding=(dilation, dilation))

class UpBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(UpBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers upsample the input when stride != 1
        self.conv1 = conv_up3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


#Encoder block
#Built for a 64x64x3 image and will result in a latent vector of size Z x 1 x 1 
#As the network is fully convolutional it will work for other larger images sized 2^n the latent
#feature map size will just no longer be 1 - aka Z x H x W
class Resdown(nn.Module):
    def __init__(self, block, layers, cluster_num=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, hidden_dim=128):
        super(Resdown, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.rep_dim = 512 * block.expansion

        self.intra = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
        )

        self.inter = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, cluster_num),
            nn.Softmax(dim=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        # x = 256 * 3 * 224 * 224
        x = self.conv1(x)
        # x = 256 * 64 * 112 * 112
        x = self.bn1(x)  # x = 256 * 64 * 112 * 112
        x = self.relu(x)
        x, indices = self.maxpool(x)
        # x = 256 * 64 * 56 * 56
        x = self.layer1(x)
        # x = 256 * 64 * 56 * 56
        x = self.layer2(x)
        # x = 256 * 128 * 28 * 28
        x = self.layer3(x)
        # x = 256 * 256 * 14 * 14
        x = self.layer4(x)
        # x = 256 * 512 * 7 * 7

        x = self.avgpool(x)
        # x = 256 * 512 * 1 * 1
        x = torch.flatten(x, 1)
        # x = 256 * 512

        return F.normalize(self.intra(x), dim=1), self.inter(x), x
    
#Decoder block
#Built to be a mirror of the encoder block
class Resup(nn.Module):
    def __init__(self, block1, block2, layers, cluster_num=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, hidden_dim=10):
        super(Resup, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3,
                               bias=False, output_padding=(1, 1))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)
        self.layer4 = self._make_layer(block1, block2, 256, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.layer3 = self._make_layer(block1, block2, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer2 = self._make_layer(block1, block2, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer1 = self._make_layer(block2, block2, 64, layers[0])

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.rep_dim = 512 * block1.expansion

        self.outra = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.outer = nn.Sequential(
            nn.Linear(cluster_num, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block1, block2, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes / block1.expansion:
            upsample = nn.Sequential(
                conv_up1x1(self.inplanes, int(planes / block1.expansion), stride),
                norm_layer(int(planes / block1.expansion)),
            )

        layers = []
        layers.append(block1(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = int(planes / block1.expansion)
        for _ in range(1, blocks):
            layers.append(block2(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, intra, indices):
        # See note [TorchScript super()]
        # x = 256 * 512
        x = (self.outra(intra)).unsqueeze(2).unsqueeze(3)

        # x = 256 * 512 * 1 * 1
        x = self.avgpool(x)   # x = 256 * 512 * 7 * 7

        x = self.layer4(x)   # x = 256 * 256 * 14 * 14
        x = self.layer3(x)   # x = 256 * 128 * 28 * 28
        x = self.layer2(x)   # x = 256 * 64 * 56 * 56
        x = self.layer1(x)   # x = 256 * 64 * 1 * 1

        x = self.maxpool(x, indices)
        x = self.bn1(x)
        x = self.conv1(x)

        return x

#VAE network, uses the above encoder and decoder blocks 
class CNN_VAE3(nn.Module):
    def __init__(self, net, hidden=128, cluster_num=10):
        super(CNN_VAE3, self).__init__()
        if net == "resnet18":
            Encoder = Resdown(block=BasicBlock, layers=[2, 2, 2, 2], hidden_dim=hidden, cluster_num=cluster_num)                    #[2, 2, 2, 2] #1, 1, 1, 1
            Decoder = Resup(block1=UpBlock, block2=BasicBlock, layers=[2, 2, 2, 2], hidden_dim=hidden, cluster_num=cluster_num)     #[2, 2, 2, 2]
        elif net == "resnet34":
            Encoder = Resdown(block=BasicBlock, layers=[3, 4, 6, 3])
            Decoder = Resup(block=BasicBlock, layers=[3, 6, 4, 3])
        elif net == "resnet50":
            Encoder = Resdown(block=Bottleneck, layers=[3, 4, 6, 3])
            Decoder = Resup(block=Bottleneck, layers=[3, 6, 4, 3])

        self.encoder = Encoder
        self.decoder = Decoder

    def encode(self, x, Train=True):
        encoding = self.encoder(x)
        #encoding = encoding.view(encoding.size(0), -1)
        return encoding

    def forward(self, x):
        intra, _, indices = self.encoder(x)
        recon = self.decoder(intra, indices)
        return recon#, mu, logvar

class LGC_CNN3(nn.Module):
    def __init__(self, hidden=128, cluster_num=10, net='resnet18'):
        super(LGC_CNN3, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation (for a 64x64 image this is the size of the latent vector)"""
        if net == "resnet18":
            Encoder = Resdown(block=BasicBlock, layers=[2, 2, 2, 2], hidden_dim=hidden, cluster_num=cluster_num)
        elif net == "resnet34":
            Encoder = Resdown(block=BasicBlock, layers=[3, 4, 6, 3])
        elif net == "resnet50":
            Encoder = Resdown(block=Bottleneck, layers=[3, 4, 6, 3])

        self.encoder = Encoder

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.net.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def cluster_indice(self, x):
        _, indice, _ = self.encoder(x)
        return indice

    def compute_mid_x(self, x):
        intra_encoding, inter_indice, x = self.encoder(x)
        return intra_encoding, inter_indice, x

    def forward(self, x):
        intra_encoding, inter_indice, _ = self.encoder(x)
        return intra_encoding, inter_indice