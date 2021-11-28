import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

#Residual down sampling block for the encoder
#Average pooling is used to perform the downsampling
class Res_down(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2):
        super(Res_down, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale,scale)
        
    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x

    
#Residual up sampling block for the decoder
#Nearest neighbour is used to perform the upsampling
class Res_up(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2):
        super(Res_up, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        
        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x
    
#Encoder block
#Built for a 64x64x3 image and will result in a latent vector of size Z x 1 x 1 
#As the network is fully convolutional it will work for other larger images sized 2^n the latent
#feature map size will just no longer be 1 - aka Z x H x W
class Encoder(nn.Module):
    def __init__(self, channels, ch=64, z=512, hidden_dim = 10, wide=28):
        super(Encoder, self).__init__()
        self.wide = wide
        self.conv1 = Res_down(channels, ch)#64
        self.conv2 = Res_down(ch, 2*ch)#32
        if wide == 64:
            self.conv3 = Res_down(2*ch, 4*ch)#16
            self.conv4 = Res_down(4*ch, 8*ch)#8
            self.conv5 = Res_down(8*ch, 8*ch)#4
            self.conv_mu = nn.Conv2d(8*ch, z, 2, 2)#2
            self.conv_logvar = nn.Conv2d(8*ch, z, 2, 2)#2
        elif wide == 32:
            self.conv3 = Res_down(2*ch, 4*ch)#8
            self.conv4 = Res_down(4*ch, 4*ch)#4
            self.conv_mu = nn.Conv2d(4*ch, z, 2, 2)#2
            self.conv_logvar = nn.Conv2d(4*ch, z, 2, 2)#2
        elif wide == 28:
            self.avg8 = nn.AdaptiveAvgPool2d((8, 8))
            self.conv3 = Res_down(2*ch, 4*ch)#8
            self.conv4 = Res_down(4*ch, 4*ch)#4
            self.conv_mu = nn.Conv2d(4*ch, z, 2, 2)#2
            self.conv_logvar = nn.Conv2d(4*ch, z, 2, 2)#2
        elif wide == 16:
            self.conv3 = Res_down(2*ch, 2*ch)#4
            self.conv_mu = nn.Conv2d(2*ch, z, 2, 2)#2
            self.conv_logvar = nn.Conv2d(2*ch, z, 2, 2)#2
        elif wide == 128:
            self.avg32 = nn.AdaptiveAvgPool2d((32, 32))
            self.conv3 = Res_down(2*ch, 4*ch)#16
            self.conv4 = Res_down(4*ch, 8*ch)#8
            self.conv5 = Res_down(8*ch, 8*ch)#4
            self.conv_mu = nn.Conv2d(8*ch, z, 2, 2)#2
            self.conv_logvar = nn.Conv2d(8*ch, z, 2, 2)#2
        self.middle = nn.Linear(z, hidden_dim)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x, Train = True):
        x = self.conv1(x)
        x = self.conv2(x)

        if self.wide == 28:
            x = self.avg8(x)
            x = self.conv3(x)
            x = self.conv4(x)
        elif self.wide == 32:
            x = self.conv3(x)
            x = self.conv4(x)
        elif self.wide == 16:
            x = self.conv3(x)
        elif self.wide == 128:
            x = self.avg32(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
        elif self.wide == 64:
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)

        if Train:
            x = self.conv_mu(x)
            #logvar = self.conv_logvar(x)
            #x = self.sample(mu, logvar)
        else:
            x = self.conv_mu(x)
            mu = None
            logvar = None
        return self.middle(x.view(x.size(0), -1))#, mu, logvar
    
#Decoder block
#Built to be a mirror of the encoder block
class Decoder(nn.Module):
    def __init__(self, channels, ch = 64, z = 512, hidden_dim=10, wide = 28):
        super(Decoder, self).__init__()
        self.wide = wide
        if wide == 28:
            self.conv1 = Res_up(z, ch*4)
            self.conv2 = Res_up(ch*4, ch*4)
            self.conv3 = Res_up(ch*4, ch*2)
            self.avg7 = nn.AdaptiveAvgPool2d((7, 7))
            self.conv4 = Res_up(ch*2, ch)
            self.conv5 = Res_up(ch, ch//2)
            self.conv6 = nn.Conv2d(ch//2, channels, 3, 1, 1)
        elif wide == 32:
            self.conv1 = Res_up(z, ch*4)
            self.conv2 = Res_up(ch*4, ch*4)
            self.conv3 = Res_up(ch*4, ch*2)
            self.conv4 = Res_up(ch*2, ch)
            self.conv5 = Res_up(ch, ch//2)
            self.conv6 = nn.Conv2d(ch//2, channels, 3, 1, 1)
        elif wide == 64:
            self.conv1 = Res_up(z, ch * 8)
            self.conv2 = Res_up(ch * 8, ch * 8)
            self.conv3 = Res_up(ch * 8, ch * 4)
            self.conv4 = Res_up(ch * 4, ch * 2)
            self.conv5 = Res_up(ch * 2, ch)
            self.conv6 = Res_up(ch, ch // 2)
            self.conv7 = nn.Conv2d(ch // 2, channels, 3, 1, 1)
        elif wide == 16:
            self.conv1 = Res_up(z, ch*2)
            self.conv2 = Res_up(ch*2, ch*2)
            self.conv3 = Res_up(ch*2, ch)
            self.conv4 = Res_up(ch, ch//2)
            self.conv5 = nn.Conv2d(ch//2, channels, 3, 1, 1)
        elif wide == 128:
            self.conv1 = Res_up(z, ch*8)
            self.conv2 = Res_up(ch*8, ch*8)
            self.conv3 = Res_up(ch*8, ch*4)
            self.conv4 = Res_up(ch*4, ch*2)
            self.avg64 = nn.AdaptiveAvgPool2d((64, 64))
            self.conv5 = Res_up(ch*2, ch)
            self.conv6 = Res_up(ch, ch//2)
            self.conv7 = nn.Conv2d(ch//2, channels, 3, 1, 1)
        self.middle = nn.Linear(hidden_dim, z)


    def forward(self, x):
        x = self.middle(x).view(x.size(0), -1, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.wide == 28:
            x = self.avg7(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
        elif self.wide == 32:
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
        elif self.wide == 64:
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
        elif self.wide == 16:
            x = self.conv4(x)
            x = self.conv5(x)
        elif self.wide == 128:
            x = self.conv4(x)
            x = self.avg64(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)

        return x 
    
#VAE network, uses the above encoder and decoder blocks 
class CNN_VAE2(nn.Module):
    def __init__(self, channel_in, z = 10, z_512 = 512, wide = 28):
        super(CNN_VAE2, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation (for a 64x64 image this is the size of the latent vector)"""
        
        self.encoder = Encoder(channel_in, z = z_512, hidden_dim = z, wide=wide)
        self.decoder = Decoder(channel_in, z = z_512, hidden_dim = z,wide=wide)

    def encode(self, x, Train=True):
        encoding = self.encoder(x, Train)
        encoding = encoding.view(encoding.size(0), -1)
        return encoding

    def forward(self, x, Train = True):
        encoding = self.encoder(x, Train)
        recon = self.decoder(encoding)
        return recon#, mu, logvar

class LGC_CNN(nn.Module):
    def __init__(self, channel_in, z=10, cluster_num=10, z_512=512, wide=28):
        super(LGC_CNN, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation (for a 64x64 image this is the size of the latent vector)"""

        self.encoder = Encoder(channel_in, z=z_512, hidden_dim=z, wide=wide)
        self.global_ = nn.Sequential(
            nn.Linear(z, z),
            nn.ReLU(True),
            nn.Linear(z, cluster_num),
        )
        self.Local = self.encoder
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(True)
        self.sigmod = nn.Sigmoid()
        # self.Global = nn.Sequential(self.encoder1, self.encoder2, self.global_, self.relu)
        self.Global = nn.Sequential(self.encoder, self.global_, self.softmax)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def cluster_indice(self, x):
        return self.softmax(self.global_(x))

    #def recon(self, z):
    #    return self.decoder(z)

    def forward(self, x, local=True):
        if (local):
            z = self.Local(x)
            return z.view(z.size(0), -1)
        else:
            return self.Global(x)