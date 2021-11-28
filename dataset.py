import torch
import torch.utils.data as data
import hashlib
from scipy.io import loadmat
import os
import numpy as np
import gzip
import pickle
from PIL import Image
from torchvision import transforms
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class JAFFE (data.Dataset):
    def __init__(self, train = True, transform = None):
        jaffe_path = 'D:\wangtuo\deep_clust\data\jaffe.mat'
        jaffe = loadmat(jaffe_path)
        self.transform = transform
        self.train = train
        
        X = jaffe['Fea'].T#torch.Tensor()
        Y = jaffe['gnd'].T#torch.Tensor()
        k = 0
        valx = []
        valy = []
        trainx = []
        trainy = []
        for i in range(len(Y)):
            if (i%2==0):
                valx.append(X[k])
                valy.append(Y[k])
            else:
                trainx.append(X[k])
                trainy.append(Y[k])
            k = k+1
        if train:
            self.X = torch.Tensor(trainx)
            self.Y = torch.Tensor(trainy)
        else:
            self.X = torch.Tensor(valx)
            self.Y = torch.Tensor(valy)

        #if self.use_cuda:
        #    self.X = self.X.to(device)
        #    self.Y = self.Y.to(device)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)

class Reuters_10K(data.Dataset):
    def __init__(self, root='./', train=True, transform=None):
        Reuters_10K_path = root + '/' + 'reuters-10k.npy'
        data = np.load(Reuters_10K_path, allow_pickle=True).item()
        self.transform = transform
        x = data['data']
        y = data['label']
        self.data = x.reshape((x.shape[0], -1)).astype(np.float32)
        self.targets = y.reshape((y.shape[0])).astype(np.int32)
        #self.data = self.data / self.data.max()
        #self.data = torch.Tensor(self.data)
        #self.targets = torch.Tensor(self.targets)

    def __getitem__(self, index):
        X = self.data[index]
        Y = self.targets[index]
        if self.transform is not None:
            X = self.transform(X)

        return X, Y

    def __len__(self):
        return len(self.targets)


class MSRA (data.Dataset):
    def __init__(self, train = True, transform = None):
        jaffe_path = 'D:\wangtuo\deep_clust\data\MSRA.mat'
        jaffe = loadmat(jaffe_path)
        self.transform = transform
        self.train = train
        
        X = jaffe['X']#torch.Tensor()
        Y = jaffe['Y']#torch.Tensor()
        k = 0
        valx = []
        valy = []
        trainx = []
        trainy = []
        for i in range(len(Y)):
            if (i%4==0):
                valx.append(X[k])
                valy.append(Y[k])
            else:
                trainx.append(X[k])
                trainy.append(Y[k])
            k = k+1
        if train:
            self.X = torch.Tensor(trainx)
            self.Y = torch.Tensor(trainy)
        else:
            self.X = torch.Tensor(valx)
            self.Y = torch.Tensor(valy)

        #if self.use_cuda:
        #    self.X = self.X.to(device)
        #    self.Y = self.Y.to(device)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)

'''
class PALM (data.Dataset):
    def __init__(self, train=True, transform=None):
        palm_path = 'D:\Tecent\python_code\TKDE\Data\matlab_data/PalmData.mat'
        palm = loadmat(palm_path)
        self.transform = transform
        self.train = train
        
        self.X = palm['X']
        self.Y = palm['Y'][:, 0]

        self.valx = []
        self.valy = []
        for i in range(len(self.Y)):
            if i % 10 == 0:
                self.valx.append(self.X[i])
                self.valy.append(self.Y[i])

    def __getitem__(self, index):
        if self.train:
            X = self.X[index]
            Y = self.Y[index]
        else:
            X = self.valx[index]
            Y = self.valy[index]

        X = transforms.ToPILImage()(X[:,np.newaxis].astype(np.int16))

        if self.transform is not None:
            X = self.transform(X)

        return X, self.Y[index]

    def __len__(self):
        if self.train:
            return len(self.Y)
        else:
            return len(self.valy)
'''

class PALM(data.Dataset):
    def __init__(self, transform=None):
        path = r'D:\Tecent\python_code\TKDE\Data\New_palm/'
        self.data = np.load(path + 'palmX.npy')
        self.targets = np.load(path + 'palmY.npy')
        #self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        x = Image.fromarray(np.uint8(x))
        y = self.targets[index]
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.targets)

class Coil_20 (data.Dataset):
    def __init__(self, train=True, transform=None):
        palm_path = r'/home/ubuntu/Data/coil_20.mat'
        palm = loadmat(palm_path)
        self.transform = transform
        self.train = train
        
        X = palm['fea']
        Y = palm['gnd'][0, :].T
        X = np.vstack(X).reshape(-1, 128, 128)
        self.X = X  # convert to HWC
        self.Y = Y

    def __getitem__(self, index):
        img, target = self.X[index], self.Y[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.transform is not None:
            img = self.transform(img)

        #if self.transform is not None:
        #    img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.Y)

class Coil_100 (data.Dataset):
    def __init__(self, train=True, transform=None):
        palm_path = r'D:\wangtuo\deep_clust\dataset\New_data\coil_100.mat'
        palm = loadmat(palm_path)
        self.transform = transform
        self.train = train
        
        X = palm['fea']
        Y = palm['gnd'][0, :].T
        valx = X[0:256]
        valy = Y[0:256]

        if train:
            self.X = torch.Tensor(X)
            self.Y = torch.Tensor(Y)
        else:
            self.X = torch.Tensor(valx)
            self.Y = torch.Tensor(valy)

        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)

class USPS (data.Dataset):
    def __init__(self, train=True, transform=None):
        palm_path = r'E:\Data\New_data\USPS.mat'
        palm = loadmat(palm_path)
        self.transform = transform
        self.train = train
        
        X = palm['fea']
        Y = palm['gnd'][:, 0]
        valx = X[0:256]
        valy = Y[0:256]

        if train:
            self.X = torch.Tensor(X)
            self.Y = torch.Tensor(Y)
        else:
            self.X = torch.Tensor(valx)
            self.Y = torch.Tensor(valy)

        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]#, torch.from_numpy(np.array(index))

    def __len__(self):
        return len(self.Y)

class USPS2 (data.Dataset):
    def __init__(self, train=True, transform=None):
        path='/home/ubuntu/Data/USPS/usps_resampled.mat'
        data = loadmat(path)
        
        x_train, y_train, x_test, y_test = data['train_patterns'].T, data['train_labels'].T, data['test_patterns'].T, data['test_labels'].T
    
        y_train = [np.argmax(l) for l in y_train]
        y_test = [np.argmax(l) for l in y_test]
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        self.Y = np.concatenate((y_train, y_test)).astype(np.int32)

        self.X = (x.reshape((x.shape[0], -1)) + 1.0) / 2.0
        self.X = torch.Tensor(self.X)
        self.Y = torch.Tensor(self.Y)
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]#, torch.from_numpy(np.array(index))

    def __len__(self):
        return len(self.Y)

class Yale (data.Dataset):
    def __init__(self, train = True, transform = None):
        yale_path = 'E:\Data\matlab_data\YaleB.mat'
        palm = loadmat(yale_path)
        self.transform = transform
        self.train = train
        
        X = palm['fea'].T
        Y = palm['gnd'].T[:, 0]
        X = X.reshape(X.shape[0],1,32,32)/255
        k = 0
        valx = []
        valy = []
        trainx = []
        trainy = []
        for i in range(len(Y)):
            if (i%4==0):
                valx.append(X[k])
                valy.append(Y[k])
            else:
                trainx.append(X[k])
                trainy.append(Y[k])
            k = k+1

        #self.X = jaffe['Fea'].T
        #self.Y = jaffe['gnd'].T
        if train:
            self.X = torch.Tensor(trainx)
            self.Y = torch.Tensor(trainy)
        else:
            self.X = torch.Tensor(valx)
            self.Y = torch.Tensor(valy)

        #if self.use_cuda:
        #   self.X = self.X.to(device)
        #    self.Y = self.Y.to(device)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)
'''
palm_path = 'D:\Study\Deep_cluster\data\PalmData.mat'
palm = loadmat(palm_path)
x = palm['X']
y = palm['Y']
print(len(x),len(x[0]),len(y),len(y[0]))
'''


class Fashion(data.Dataset):
    def __init__(self, root, train = True):
        with gzip.open(os.path.join(root, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
            y_train = np.frombuffer(f.read(), np.uint8, offset=8)

        with gzip.open(os.path.join(root, 'train-images-idx3-ubyte.gz'), 'rb') as f:
            x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

        with gzip.open(os.path.join(root, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
            y_test = np.frombuffer(f.read(), np.uint8, offset=8)

        with gzip.open(os.path.join(root, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
            x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

        x_train = np.r_[x_train, x_test].reshape(70000, 784) / 255
        x_test = x_test.reshape(10000,784) / 255
        y_train = np.r_[y_train, y_test]

        if train:
            self.X = torch.Tensor(x_train)
            self.Y = torch.Tensor(y_train)
        else:
            self.X = torch.Tensor(x_test)
            self.Y = torch.Tensor(y_test)

        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)

class Fashion2(data.Dataset):
    def __init__(self, train = True):
        path=r'/home/ubuntu/Data/Fashion-MNIST/'
        x = np.load(path + 'data.npy').astype(np.float32)
        y = np.load(path + 'labels.npy').astype(np.int32)
        x = x.reshape((x.shape[0], -1))     

        self.X = torch.Tensor(x)
        self.Y = torch.Tensor(y)

        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)

class Cifar_10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        #super(Cifar_10, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

'''

    def __init__(self, train=True, transform=None):
        data = []
        targets = []
        self.transform = transform
        if train:
            train_list = [
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
                ['test_batch', '40351d587109b95175f43aff81a1287e'],
            ]
        else:
            train_list = [['test_batch', '40351d587109b95175f43aff81a1287e'],]

        file = 'E:\Data\cifar\cifar-10-batches-py'
        for file_name, checksum in train_list:
            file_path = os.path.join(file, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])
        data = np.vstack(data).reshape(-1, 3, 32, 32)
        self.X = torch.Tensor(data.transpose((0, 2, 3, 1)))/255  # convert to HWC
        self.Y = torch.Tensor(targets)
        if not train:
            self.X = self.X[0:256]
            self.Y = self.Y[0:256]

    def __getitem__(self, index):
        img, target = self.X[index], self.Y[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        #if self.transform is not None:
        #    img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.Y)
'''


class Cifar_100 (data.Dataset):
    def __init__(self, train=True, transform=None):
        palm_path = r'D:\wangtuo\deep_clust\dataset\cifar\cifar_100.mat'
        palm = loadmat(palm_path)
        self.transform = transform
        self.train = train
        
        X = palm['X']
        Y = palm['Y'][:, 0]
        valx = X[0:256]
        valy = Y[0:256]

        if train:
            self.X = torch.Tensor(X)
            self.Y = torch.Tensor(Y)
        else:
            self.X = torch.Tensor(valx)
            self.Y = torch.Tensor(valy)

        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        

        trans = torch.zeros(28, 28).to(device)
        for j in range(28):
            trans[j, 27 - j] = 1
        self.trans = trans

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            self.test_data = self.test_data.view(self.test_data.size(0), -1).float() / 255
            # self.test_data = self.test_data.view(self.test_data.size(0), -1).float()/255
            self.test_labels = self.test_labels.int()
            self.train_data = self.train_data.view(self.train_data.size(0), -1).float()/ 255
            # self.train_data = self.train_data.view(self.train_data.size(0), -1).float()/255
            self.train_labels = self.train_labels.int()
            self.train_data = torch.cat([self.train_data, self.test_data], dim=0)
            self.train_labels = torch.cat([self.train_labels, self.test_labels], dim=0)
            
            self.train_data = self.train_data.to(device)
            self.train_labels = self.train_labels.to(device)
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            self.test_data = self.test_data.view(self.test_data.size(0), -1).float()/ 255
            # self.test_data = self.test_data.view(self.test_data.size(0), -1).float()/255
            self.test_labels = self.test_labels.int()
            
            self.test_data = self.test_data.to(device)
            self.test_labels = self.test_labels.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        #img = img.unsqueeze(0)
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)
