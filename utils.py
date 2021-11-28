import torch
import numpy as np
import os
import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, pair_confusion_matrix
nmi = normalized_mutual_info_score
#ari = adjusted_rand_score

import torchvision
#import cv2
import scipy.io as scio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ari(labels_true, labels_pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    tn = tn.astype(np.float)
    fp = fp.astype(np.float)
    fn = fn.astype(np.float)
    tp = tp.astype(np.float)

    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))

# Set seeds to ensure reproducibility
def SetSeed(seed):
    SEED = 0

    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Add Gaussian noise for denoising autoencoder
def add_noise(img, noise_level):

    noise = torch.randn(img.size()).to(device) * noise_level
    noisy_img = img + noise

    return noisy_img

def Real_Center(W, H, y, n_cluster):
    new_center = torch.zeros((n_cluster, n_cluster)).to(device)
    for i in range(n_cluster):
        new_center[i, :] = torch.mean(W @ H[:, y == i], 1).T
    return new_center


def positive(z):
    return (torch.abs(z) + z)/2

def negtive (z):
    return (torch.abs(z) - z)/2

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    '''
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    '''
    y_true = y_true.astype(np.int64)
    y_predicted = y_pred
    cluster_number = None
    if cluster_number is None:
        cluster_number = max(y_predicted.max(), y_true.max()) + 1  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy

def save_model(model_path, net1, lgc_loss, optimizer, Intra_epoch, Inter_epoch, global_epoch):
    out = os.path.join(model_path, "checkpoint_Global_{}.Inter_{}.Intra_{}.tar".format(global_epoch, Inter_epoch, Intra_epoch))
    state = {'net': net1.state_dict(), 'lgc_loss': lgc_loss, 'optimizer': optimizer.state_dict(), 'Intra_epoch': Intra_epoch,
             'Inter_epoch': Inter_epoch, 'global_epoch': global_epoch}
    torch.save(state, out)

def show_para(args):
    print("Data\t\t\t:\t",        args.name)
    print("n_cluster\t\t:\t",     args.n_cluster)
    print("hidden_dim\t\t:\t",    args.hidden_dim)
    print("num_workers\t\t:\t",   args.num_workers)
    print("net\t\t\t\t:\t",       args.net)
    print("channel\t\t\t:\t",     args.channel)
    print("wide\t\t\t:\t",        args.wide)
    print("reload\t\t\t:\t",      args.reload)
    print("m:\t\t\t\t:\t",        args.m)
    print("divide\t\t\t:\t",      args.divide)
    print("optim\t\t\t:\t",       args.optim)
    print("path\t\t\t:\t",        args.Path)

    print("Intra_batchsize\t:\t", args.Intra_batchsize, "\t\tInter_batchsize\t:\t", args.Inter_batchsize, "\t\t\tglobal_batchsize:\t", args.global_batchsize)
    print("Intra_epoch\t\t:\t",   args.Intra_epoch,     "\t\t\tInter_epoch\t\t:\t", args.Inter_epoch,     "\t\t\t\tglobal_epoch\t:\t", args.global_epoch)
    print("Intra_lr\t\t:\t",      args.Intra_lr,        "\tInter_lr\t\t:\t",        args.Inter_lr,        "\t\tglobal_lr\t\t:\t",      args.global_lr)
    print("Intra_max\t\t:\t",     args.Intra_Max,       "\t\t\tInter_max\t\t:\t",   args.Inter_Max,       "\t\t\t\tGlobal_Max\t:\t",   args.Global_Max)

class Show():
    def __init__(self, pretrain=True):
        super(Show, self).__init__()
        self.pretrain = pretrain
        self.loss = []
        self.AccK = []
        if not self.pretrain:
            self.AccD = []
            self.Nmi = []
            self.Ari = []

            self.loss_m = []
            self.AccK_m = []
            self.AccD_m = []
            self.Nmi_m = []
            self.Ari_m = []

            self.loss_c = []
            self.AccK_c = []
            self.AccD_c = []
            self.Nmi_c = []
            self.Ari_c = []

    def append(self,  loss=-1, AccK=-1, AccD=-1, Nmi=-1, Ari=-1, loss_type='manifold'):
        self.loss.append(loss)
        self.AccK.append(AccK)
        if not self.pretrain:
            self.AccD.append(AccD)
            self.Nmi.append(Nmi)
            self.Ari.append(Ari)
            if loss_type == 'manifold':
                self.loss_m.append(loss)
                self.AccK_m.append(AccK)
                self.AccD_m.append(AccD)
                self.Nmi_m.append(Nmi)
                self.Ari_m.append(Ari)
            elif loss_type == 'cluster':
                self.loss_c.append(loss)
                self.AccK_c.append(AccK)
                self.AccD_c.append(AccD)
                self.Nmi_c.append(Nmi)
                self.Ari_c.append(Ari)

    def plot(self, path='./', plot=False):
        x = range(len(self.loss))
        if self.pretrain:
            plt.subplot(121)
            plt.plot(x, self.loss, label='loss')
            plt.xlabel('Epoch')
            plt.ylabel('pretrain_loss')
            plt.title('pretrain_loss')
            plt.subplot(122)
            plt.plot(x, self.AccK, label='AccK')
            plt.xlabel('Epoch')
            plt.ylabel('AccK')
            plt.title('AccK')
        else:
            x_m = range(len(self.loss_m))
            x_c = range(len(self.loss_c))
            plt.subplot(351)
            plt.plot(x, self.loss, label='loss')
            plt.xlabel('Epoch')
            plt.ylabel('loss')
            plt.title('loss')
            plt.subplot(352)
            plt.plot(x, self.AccK, label='AccK')
            plt.xlabel('Epoch')
            plt.ylabel('AccK')
            plt.title('AccK')
            plt.subplot(353)
            plt.plot(x, self.AccD, label='AccD')
            plt.xlabel('Epoch')
            plt.ylabel('AccD')
            plt.title('AccD')
            plt.subplot(354)
            plt.plot(x, self.Nmi, label='Nmi')
            plt.xlabel('Epoch')
            plt.ylabel('Nmi')
            plt.title('Nmi')
            plt.subplot(355)
            plt.plot(x, self.Ari, label='Ari')
            plt.xlabel('Epoch')
            plt.ylabel('Ari')
            plt.title('Ari')
            plt.subplot(356)
            plt.plot(x_m, self.loss_m, label='loss_m')
            plt.xlabel('Epoch')
            plt.ylabel('loss_m')
            plt.title('loss_m')
            plt.subplot(357)
            plt.plot(x_m, self.AccK_m, label='AccK_m')
            plt.xlabel('Epoch')
            plt.ylabel('AccK_m')
            plt.title('AccK_m')
            plt.subplot(358)
            plt.plot(x_m, self.AccD_m, label='AccD_m')
            plt.xlabel('Epoch')
            plt.ylabel('AccD_m')
            plt.title('AccD_m')
            plt.subplot(359)
            plt.plot(x_m, self.Nmi_m, label='Nmi_m')
            plt.xlabel('Epoch')
            plt.ylabel('Nmi_m')
            plt.title('Nmi_m')
            plt.subplot(3, 5, 10)
            plt.plot(x_m, self.Ari_m, label='Ari_m')
            plt.xlabel('Epoch')
            plt.ylabel('Ari_m')
            plt.title('Ari_m')
            plt.subplot(3, 5, 11)
            plt.plot(x_c, self.loss_c, label='loss_c')
            plt.xlabel('Epoch')
            plt.ylabel('loss_c')
            plt.title('loss_c')
            plt.subplot(3, 5, 12)
            plt.plot(x_c, self.AccK_c, label='AccK_c')
            plt.xlabel('Epoch')
            plt.ylabel('AccK_c')
            plt.title('AccK_c')
            plt.subplot(3, 5, 13)
            plt.plot(x_c, self.AccD_c, label='AccD_c')
            plt.xlabel('Epoch')
            plt.ylabel('AccD_c')
            plt.title('AccD_c')
            plt.subplot(3, 5, 14)
            plt.plot(x_c, self.Nmi_c, label='Nmi_c')
            plt.xlabel('Epoch')
            plt.ylabel('Nmi_c')
            plt.title('Nmi_c')
            plt.subplot(3, 5, 15)
            plt.plot(x_c, self.Ari_c, label='Ari_c')
            plt.xlabel('Epoch')
            plt.ylabel('Ari_c')
            plt.title('Ari_c')
        plt.savefig(path+'.jpg', dpi=1500, bbox_inches='tight')
        if plot:
            plt.show()

        plt.close()

def plot_embedding(data, label, name):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0] - 10):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1((label[i] / 10)),  ## color=plt.cm.Set1(np.random.randint(10))
                 fontdict={'weight': 'bold', 'size': 9})
    for j in range(10):
        plt.text(data[-1 - j, 0], data[-1 - j, 1], '*' + 'C' + '*',
                 color=plt.cm.Set1((label[-1 - j] / 10)),  ## color=plt.cm.Set1(np.random.randint(10))
                 fontdict={'weight': 'bold', 'size': 20})
    plt.xticks([])
    plt.yticks([])
    plt.title(name)
    return fig


def show_a_batch(args, z, h, Y, mu, AK, AC, Nmi, Ari, AD, Intra_epoch, Inter_epoch, global_epoch):
    name = args.Path1 + '/Epoch ' + "Global_{}.Inter_{}.Intra_{}".format(global_epoch, Inter_epoch, Intra_epoch)
    scio.savemat(name + '.mat', {'Z': z, 'H': h, 'Y': Y, 'MU': mu, 'AK': AK, 'AC': AC, 'Nmi': Nmi, 'Ari': Ari, 'AD': AD})
    mark  = np.ones(10)
    for center in range(10):
        mark[center] = center
    show_picture = np.vstack((z[0:1000, :], mu))

    label = np.hstack([Y[0:1000], mark.astype(np.int64)])
    # data_umap = umap.UMAP().fit_transform(show_picture.cpu().detach().numpy())
    data_tsne = TSNE().fit_transform(show_picture)

    plt.rcParams['figure.figsize'] = (15, 15)  ## 显示的大小

    fig = plot_embedding(data_tsne, label, 'Epoch ' + "Global_{}.Inter_{}.Intra_{}".format(global_epoch, Inter_epoch, Intra_epoch))
    show_picture = show_picture
    label = label

    print('save embedding jpg:', name)
    plt.savefig(name+'.jpg')
    plt.close()

def Evaluation(z_pred, h_pred, y_ground, kmeans1):
    y_pred = kmeans1.fit_predict(z_pred)
    y_pred_c = torch.argmax(h_pred, dim=1).data.cpu().numpy()
    _, AC1 = acc(y_ground, y_pred);
    AC1 *= 100
    _, AC2 = acc(y_ground, y_pred_c);
    AC2 *= 100
    Nmi1 = nmi(y_ground, y_pred) * 100
    Ari1 = ari(y_ground, y_pred) * 100
    return AC1, AC2, Nmi1, Ari1

def Evaluation1(z_pred, h_pred, y_ground, kmeans1, AD):
    y_pred = kmeans1.fit_predict(z_pred)
    y_pred_c = torch.argmax(h_pred, dim=1).data.cpu().numpy()
    _, AC1 = acc(y_ground, y_pred);
    AC1 *= 100
    _, AC2 = acc(y_ground, y_pred_c);
    AC2 *= 100
    y_pred_d = torch.argmax(AD, dim=1).data.cpu().numpy()
    Nmi1 = nmi(y_ground, y_pred_c) * 100
    Ari1 = ari(y_ground, y_pred_c) * 100
    return AC1, AC2, Nmi1, Ari1

def Evaluation_single(h_pred, y_ground):
    y_pred_c = torch.argmax(h_pred, dim=1).data.cpu().numpy()
    _, AC2 = acc(y_ground, y_pred_c);
    AC2 *= 100
    Nmi2 = nmi(y_ground, y_pred_c) * 100
    Ari2 = ari(y_ground, y_pred_c) * 100
    return AC2, Nmi2, Ari2

def Evaluation2(y_ground, AD):
    y_pred_d = torch.argmax(AD, dim=1).data.cpu().numpy()
    Nmi1 = nmi(y_ground, y_pred_d) * 100
    Ari1 = ari(y_ground, y_pred_d) * 100
    return Nmi1, Ari1

