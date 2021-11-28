import torchvision
import cv2
import numpy as np


class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


class Transforms:
    def __init__(self, size=224, s=0.5, channels=3, mean=None, std=None, blur=False, name='Mnist_test', aug=True):
        self.aug = aug
        if name=='cifar_10' or name == 'STL_10':
            self.train_transform = [
                torchvision.transforms.RandomResizedCrop(size=size),#, scale=(0.64, 1)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.5),
            torchvision.transforms.RandomGrayscale(p=0.2),
            ]
            if blur:
                self.train_transform.append(GaussianBlur(kernel_size=23))
            self.train_transform.append(torchvision.transforms.ToTensor())
            #self.train_transform.append(torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            self.test_transform = [
                #torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        elif name=='Fashion' :
            self.train_transform = [
                torchvision.transforms.Grayscale(num_output_channels=channels),
                torchvision.transforms.RandomResizedCrop(size=size),#, scale=(0.64, 1)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.5),
                torchvision.transforms.RandomGrayscale(p=0.2),
            ]
            if blur:
                self.train_transform.append(GaussianBlur(kernel_size=23))
            self.train_transform.append(torchvision.transforms.ToTensor())
            # self.train_transform.append(torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            self.test_transform = [
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        elif name=='Mnist_test' or name == 'USPS':
            self.train_transform = [
                torchvision.transforms.Grayscale(num_output_channels=channels),
                torchvision.transforms.RandomResizedCrop(size=size, scale=(0.64, 1)),
                torchvision.transforms.RandomRotation(30),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                    p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
            ]
            if blur:
                self.train_transform.append(GaussianBlur(kernel_size=23))
            self.train_transform.append(torchvision.transforms.ToTensor())
            #self.train_transform.append(torchvision.transforms.Normalize((0.5,), (0.5,)))
            self.test_transform = [
                torchvision.transforms.Grayscale(num_output_channels=channels),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize((0.5,), (0.5,))
            ]
        elif name=='Palm':
            self.train_transform = [
                torchvision.transforms.Grayscale(num_output_channels=channels),
                #torchvision.transforms.RandomResizedCrop(size=size, scale=(0.64, 1)),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                    p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
            ]
            if blur:
                self.train_transform.append(GaussianBlur(kernel_size=23))
            self.train_transform.append(torchvision.transforms.ToTensor())
            #self.train_transform.append(torchvision.transforms.Normalize((0.5,), (0.5,)))
            self.test_transform = [
                torchvision.transforms.Grayscale(num_output_channels=channels),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize((0.5,), (0.5,))
            ]

        if mean and std:
            self.train_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        if self.aug:
            return self.test_transform(x), self.train_transform(x), self.train_transform(x)#, self.train_transform(x), self.train_transform(x)
        else:
            return self.test_transform(x)
