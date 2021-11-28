import torch.utils.data
import torchvision
from torchvision import transforms
from dataset import PALM, Yale, JAFFE, MSRA, MNIST, Fashion, Fashion2, USPS, USPS2, Coil_20, Cifar_10, Reuters_10K
import numpy as np
import transform

def shuffle_dataset(set, labels=False):
	per = np.random.permutation(set.data.shape[0])
	if labels:
		set.labels = set.labels[per]
		set.data = set.data[per, :, :, :]
	else:
		set.data = set.data[per, :, :]
		set.targets = np.array(set.targets)[per]

def find_data(args, train_mode=True, aug=True, stage=0):
	import transform
	if stage == 0:
		batchsize = args.Intra_batchsize
		shuffle   = True
	elif stage == 1:
		batchsize = args.Inter_batchsize
		shuffle   = False
	elif stage == 2:
		batchsize = args.global_batchsize
		shuffle   = False

	if (args.name == 'Palm'):
		set = PALM(transform=transform.Transforms(size=args.wide, channels=args.channel, name='Mnist_test', aug=aug))
		shuffle_dataset(set)
		return torch.utils.data.DataLoader(set, batch_size=batchsize, num_workers=args.num_workers), set.targets.size
	elif (args.name == 'Mnist'):
		set1 = torchvision.datasets.MNIST('../Data', train=True, download=True,
			   	transform=transform.Transforms(size=args.wide, channels=args.channel, name='Mnist_test', aug=aug))  # Transforms(size=32, s=0.5))
		set2 = torchvision.datasets.MNIST('../Data', train=False, download=True,
				transform=transform.Transforms(size=args.wide, channels=args.channel, name='Mnist_test', aug=aug))  # Transforms(size=32, s=0.5))
		shuffle_dataset(set1)
		shuffle_dataset(set2)
		set = torch.utils.data.ConcatDataset([set1, set2])
		return torch.utils.data.DataLoader(set, batch_size=batchsize, shuffle=shuffle, num_workers=args.num_workers), set1.data.size(0) + set2.data.size(0)
	elif (args.name == 'KMnist'):
		class transform_train:
			def __init__(self, size=10):
				self.transform_train = transforms.Compose([
					# transforms.RandomHorizontalFlip(),  # 水平翻转
					# transforms.RandomVerticalFlip(),      # 上下翻转
					transforms.RandomCrop(28, padding=2),
					transforms.RandomRotation(15),  # 旋转 -15° - 15°
					# transforms.ToTensor(),            # To Tensor
					# transforms.RandomRotation([90, 180, 270]),  # 随即旋转 90° 180° 270°
					# transforms.Normalize((0.1307,), (0.3081,))  # 均衡化
					# transforms.Resize([32, 32]),        # 转化到32*32
					# transforms.RandomCrop([28, 28])     # 先旋转了15°，转换成32*32，再取28*28部分
					# transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					# transforms.Normalize((0.1307,), (0.3081,))])  # 这一步可以不要
					# transforms.Normalize((0.5,), (0.5,))
				])  # 这一步可以不要
				self.transform_test = transforms.Compose([
					transforms.ToTensor(),
					# transforms.Normalize((0.1307,), (0.3081,))])
					# transforms.Normalize((0.5,), (0.5,))
				])  # 这一步可以不要

				def __call__(self, x):
					return self.transform_test(x), self.transform_train(x), self.transform_train(x)

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize((0.1307,), (0.3081,))])
			# transforms.Normalize((0.5,), (0.5,))
		])  # 这一步可以不要
		if train_mode:
			if args.trans:
				set1 = torchvision.datasets.KMNIST('../Data', train=True, download=True,
												   transform=transform_train())  # Transforms(size=32, s=0.5))
				set2 = torchvision.datasets.KMNIST('../Data', train=False, download=True,
												   transform=transform_train())  # Transforms(size=32, s=0.5))
			else:
				set1 = torchvision.datasets.KMNIST('../Data', train=True, download=True,
												   transform=transform_test)
				set2 = torchvision.datasets.KMNIST('../Data', train=False, download=True,
												   transform=transform_test)
		else:
			set1 = torchvision.datasets.KMNIST('../Data', train=True, download=True,
											   transform=transform_test)
			set2 = torchvision.datasets.KMNIST('../Data', train=False, download=True,
											   transform=transform_test)
		shuffle_dataset(set1)
		shuffle_dataset(set2)
		set = torch.utils.args.ConcatDataset([set1, set2])
		return torch.utils.args.DataLoader(set, batch_size=batchsize, shuffle=args.shuffle,
										   num_workers=args.num_workers)
	elif (args.name == 'QMnist'):
		transform_train = transforms.Compose([
			transforms.RandomHorizontalFlip(),  # 水平翻转
			# transforms.RandomVerticalFlip(),      # 上下翻转
			# transforms.RandomRotation(15),        # 旋转 -15° - 15°
			# transforms.ToTensor(),            # To Tensor
			# transforms.RandomRotation([90, 180, 270]),  # 随即旋转 90° 180° 270°
			# transforms.Normalize((0.1307,), (0.3081,))  # 均衡化
			# transforms.Resize([32, 32]),        # 转化到32*32
			# transforms.RandomCrop([28, 28])     # 先旋转了15°，转换成32*32，再取28*28部分
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# transforms.Normalize((0.1307,), (0.3081,))])  # 这一步可以不要
			# transforms.Normalize((0.5,), (0.5,))
		])  # 这一步可以不要

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize((0.1307,), (0.3081,))])
			# transforms.Normalize((0.5,), (0.5,))
		])  # 这一步可以不要
		if train_mode:
			if args.trans:
				set1 = torchvision.datasets.QMNIST('../Data', train=True, download=True,
												   transform=transform_train)  # Transforms(size=32, s=0.5))
				set2 = torchvision.datasets.QMNIST('../Data', train=False, download=True,
												   transform=transform_train)  # Transforms(size=32, s=0.5))
			else:
				set1 = torchvision.datasets.QMNIST('../Data', train=True, download=True,
												   transform=transform_test)
				set2 = torchvision.datasets.QMNIST('../Data', train=False, download=True,
												   transform=transform_test)
		else:
			set1 = torchvision.datasets.QMNIST('../Data', train=True, download=True,
											   transform=transform_test)
			set2 = torchvision.datasets.QMNIST('../Data', train=False, download=True,
											   transform=transform_test)
		shuffle_dataset(set1)
		shuffle_dataset(set2)
		set = torch.utils.args.ConcatDataset([set1, set2])
		return torch.utils.args.DataLoader(set, batch_size=batchsize, shuffle=args.shuffle,
										   num_workers=args.num_workers)
	elif (args.name == 'Fashion'):
		set1 = torchvision.datasets.FashionMNIST('../Data', train=True, download=True,
										  transform=transform.Transforms(size=args.wide, channels=args.channel,
																		 name='Fashion',
																		 aug=aug))  # Transforms(size=32, s=0.5))
		set2 = torchvision.datasets.FashionMNIST('../Data', train=False, download=True,
										  transform=transform.Transforms(size=args.wide, channels=args.channel,
																		 name='Fashion',
																		 aug=aug))  # Transforms(size=32, s=0.5))
		shuffle_dataset(set1)
		shuffle_dataset(set2)
		set = torch.utils.data.ConcatDataset([set1, set2])
		return torch.utils.data.DataLoader(set, batch_size=batchsize, shuffle=shuffle,
										   num_workers=args.num_workers), set1.data.size(0) + set2.data.size(0)
	elif (args.name == 'Yale'):
		return torch.utils.args.DataLoader(
			Yale(train=train_mode), batch_size=batchsize, shuffle=args.shuffle,
			num_workers=args.num_workers)
	elif (args.name == 'USPS'):
		set1 = torchvision.datasets.USPS('../Data', train=True, download=True,
										  transform=transform.Transforms(size=args.wide, channels=args.channel,
																		 name='USPS',
																		 aug=aug))  # Transforms(size=32, s=0.5))
		set2 = torchvision.datasets.USPS('../Data', train=False, download=True,
										  transform=transform.Transforms(size=args.wide, channels=args.channel,
																		 name='USPS',
																		 aug=aug))  # Transforms(size=32, s=0.5))
		shuffle_dataset(set1)
		shuffle_dataset(set2)
		set = torch.utils.data.ConcatDataset([set1, set2])
		return torch.utils.data.DataLoader(set, batch_size=batchsize, shuffle=shuffle,
										   num_workers=args.num_workers), set1.data.shape[0] + set2.data.shape[0]


	# elif (args.name == 'Coil_20'):
	#   return torch.utils.args.DataLoader(
	#        Coil_20(train=train_mode, transform=args.trans), batch_size=batchsize, shuffle=args.shuffle,
	#        num_workers=args.num_workers)
	elif (args.name == 'Cifar_10'):
		set1 = torchvision.datasets.CIFAR10(
			root='../Data/cifar',
			download=True,
			train=True,
			transform=transform.Transforms(size=224, name='cifar_10', aug=aug),
		)
		set2 = torchvision.datasets.CIFAR10(
			root='../Data/cifar',
			download=True,
			train=False,
			transform=transform.Transforms(size=224, name='cifar_10', aug=aug),
		)
		shuffle_dataset(set1)
		shuffle_dataset(set2)
		set = torch.utils.data.ConcatDataset([set1, set2])
		return torch.utils.data.DataLoader(set, batch_size=batchsize, shuffle=shuffle, num_workers=args.num_workers), set1.data.shape[0] + set2.data.shape[0]
	elif (args.name == 'Cifar_10_train'):
		set = torchvision.datasets.CIFAR10(
			root='../Data/cifar',
			download=True,
			train=True,
			transform=transform.Transforms(size=224, name='cifar_10', aug=aug),
		)
		shuffle_dataset(set)
		return torch.utils.data.DataLoader(set, batch_size=batchsize, shuffle=shuffle, num_workers=args.num_workers), set.data.shape[0]
	elif (args.name == 'Cifar_10_test'):
		set = torchvision.datasets.CIFAR10(
			root='../Data/cifar',
			download=True,
			train=False,
			transform=transform.Transforms(size=224, name='cifar_10', aug=aug),
		)
		shuffle_dataset(set)
		return torch.utils.data.DataLoader(set, batch_size=batchsize, shuffle=shuffle, num_workers=args.num_workers), set.data.shape[0]
	elif (args.name == 'USPS2'):
		return torch.utils.args.DataLoader(
			USPS2(train=train_mode), batch_size=batchsize, shuffle=args.shuffle,
			num_workers=args.num_workers)
	elif (args.name == 'Fashion2'):
		return torch.utils.args.DataLoader(
			Fashion2(train=train_mode), batch_size=batchsize, shuffle=args.shuffle,
			num_workers=args.num_workers)

	elif (args.name == 'STL_10'):
		set1 = torchvision.datasets.STL10(
			root='../Data/STL10',
			split='train',
			download=True,
			transform=transform.Transforms(size=224, name='STL_10', blur=True, aug=aug),
		)
		set2 = torchvision.datasets.STL10(
			root='../Data/STL10',
			split='test',
			download=True,
			transform=transform.Transforms(size=224, name='STL_10', blur=True, aug=aug),
		)
		shuffle_dataset(set1, labels=True)
		shuffle_dataset(set2, labels=True)
		set = torch.utils.data.ConcatDataset([set1, set2])
		return torch.utils.data.DataLoader(set, batch_size=batchsize, shuffle=shuffle, num_workers=args.num_workers), set1.data.shape[0] + set2.data.shape[0]

	elif (args.name == 'tiny'):
		'''
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
            self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        '''
		transform_train = transforms.Compose([
			# transforms.RandomCrop(64, padding=4),
			transforms.RandomHorizontalFlip(),
			torchvision.transforms.RandomApply(
				[torchvision.transforms.ColorJitter(0.8 * 0.5, 0.8 * 0.5, 0.8 * 0.5, 0.2 * 0.5)], p=0.8),
			# transforms.RandomRotation(20),
			transforms.ToTensor(),
			# transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])])  # 使用公式进行归一化channel=（channel-mean）/std，因为transforms.ToTensor()已经把数据处理成[0,1],那么(x-0.5)/0.5就是[-1.0, 1.0]
			# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])])
			# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])

		if train_mode:
			if args.trans:
				set = torchvision.datasets.ImageFolder(root='../Data/tiny-imagenet-200/train',
													   transform=transform_train)
			else:
				set = torchvision.datasets.ImageFolder(root='../Data/tiny-imagenet-200/train', transform=transform_test)
		else:
			set = torchvision.datasets.ImageFolder(root='../Data/tiny-imagenet-200/train', transform=transform_test)

		np.random.shuffle(set.samples)
		return torch.utils.args.DataLoader(set, batch_size=batchsize, shuffle=args.shuffle,
										   num_workers=args.num_workers)
	elif (args.name == 'Reuters_10K'):
		transform = transforms.Compose([
			# transforms.ToTensor(),
			# transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])])
			transforms.Normalize([0.5], [0.5])
		])
		set = Reuters_10K(root='../Data', train=True, transform=None)
		per = np.random.permutation(set.args.shape[0])
		set.data = set.data[per, :]
		set.targets = np.array(set.targets)[per]
		return torch.utils.args.DataLoader(set, batch_size=batchsize, shuffle=args.shuffle,
										   num_workers=args.num_workers)
	elif (args.name == 'Coil_20'):
		transform = transforms.Compose([
			transforms.ToPILImage(mode=None),
			transforms.Resize(size=64),
			transforms.ToTensor(),
			# transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])])
			# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])
		set = Coil_20(train=True, transform=transform)
		per = np.random.permutation(set.Y.shape[0])
		set.X = set.X[per, :]
		set.Y = np.array(set.Y)[per]
		return torch.utils.args.DataLoader(set, batch_size=batchsize, shuffle=args.shuffle,
										   num_workers=args.num_workers)
	elif (args.name == 'Mnist_test'):
		set = torchvision.datasets.MNIST('../Data', train=False, download=True,
			  transform=transform.Transforms(size=args.wide, channels=args.channel,	name='Mnist_test', aug=aug))
		shuffle_dataset(set)
		return torch.utils.data.DataLoader(set, batch_size=batchsize, shuffle=shuffle, num_workers=args.num_workers), set.data.size(0)
	elif (args.name == 'train_mnist'):
		set = torchvision.datasets.MNIST('../Data', train=True, download=True,
			  transform=transform.Transforms(size=args.wide, channels=args.channel, name='Mnist_test', aug=aug))
		shuffle_dataset(set)
		return torch.utils.data.DataLoader(set, batch_size=batchsize, shuffle=shuffle,
										   num_workers=args.num_workers), set.data.size(0)
	elif (args.name == 'test_mnist'):
		set = torchvision.datasets.MNIST('../Data', train=False, download=True,
			  transform=transform.Transforms(size=args.wide, channels=args.channel, name='Mnist_test', aug=aug))
		shuffle_dataset(set)
		return torch.utils.data.DataLoader(set, batch_size=batchsize, shuffle=shuffle,
										   num_workers=args.num_workers), set.data.size(0)
	elif (args.name == 'test_mnist'):
		transform_train = transforms.Compose([
			transforms.RandomHorizontalFlip(),  # 水平翻转
			# transforms.RandomVerticalFlip(),      # 上下翻转
			# transforms.RandomRotation(15),        # 旋转 -15° - 15°
			# transforms.ToTensor(),            # To Tensor
			# transforms.RandomRotation([90, 180, 270]),  # 随即旋转 90° 180° 270°
			# transforms.Normalize((0.1307,), (0.3081,))  # 均衡化
			# transforms.Resize([32, 32]),        # 转化到32*32
			# transforms.RandomCrop([28, 28])     # 先旋转了15°，转换成32*32，再取28*28部分
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# transforms.Normalize((0.1307,), (0.3081,))])  # 这一步可以不要
			# transforms.Normalize((0.5,), (0.5,))
		])  # 这一步可以不要

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize((0.1307,), (0.3081,))])
			# transforms.Normalize((0.5,), (0.5,))
		])  # 这一步可以不要
		if train_mode:
			if args.trans:
				set1 = torchvision.datasets.MNIST('../Data', train=True, download=True,
												  transform=transform_train)  # Transforms(size=32, s=0.5))
				set2 = torchvision.datasets.MNIST('../Data', train=False, download=True,
												  transform=transform_train)  # Transforms(size=32, s=0.5))
			else:
				set1 = torchvision.datasets.MNIST('../Data', train=True, download=True,
												  transform=transform_test)
				set2 = torchvision.datasets.MNIST('../Data', train=False, download=True,
												  transform=transform_test)
		else:
			set1 = torchvision.datasets.MNIST('../Data', train=True, download=True,
											  transform=transform_test)
			set2 = torchvision.datasets.MNIST('../Data', train=False, download=True,
											  transform=transform_test)
		shuffle_dataset(set1)
		shuffle_dataset(set2)
		set1.data = set1.data[50000:60000]
		set1.targets = set1.targets[50000:60000]
		set = torch.utils.args.ConcatDataset([set1, set2])
		return torch.utils.args.DataLoader(set, batch_size=batchsize, shuffle=args.shuffle,
										   num_workers=args.num_workers)

	'''
    elif (args.name == 'Fashion'):
      return torch.utils.args.DataLoader(
              Fashion('../Data/fashion_minist/', train=train_mode), batch_size=batchsize, shuffle=args.shuffle,
              num_workers=args.num_workers)
      '''