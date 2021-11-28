import torch
from torch import nn
from torch.nn import Parameter

class AutoEncoder(nn.Module):
	def __init__(self, hidden, input_size):
		super(AutoEncoder, self).__init__()
		self.encoder1 = nn.Sequential(
			nn.Linear(input_size, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True)
		)
		self.encoder2 = nn.Sequential(
			nn.Linear(500, 2000),
			nn.ReLU(True),
			nn.Linear(2000, hidden)
		)
		self.decoder = nn.Sequential(
			nn.Linear(hidden, 2000),
			nn.ReLU(True),
			nn.Linear(2000, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
			nn.Linear(500, input_size))
		self.model = nn.Sequential(self.encoder1, self.encoder2, self.decoder)

	def encode(self, x):
		return self.encoder2(self.encoder1(x))

	def decode(self, x):
		return self.decoder(x)

	def forward(self, x):
		return self.model(x)

class LGC(nn.Module):
	def __init__(self, hidden, input_size, cluster=10):
		super(LGC, self).__init__()
		self.encoder1 = nn.Sequential(
			nn.Linear(input_size, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True)
		)
		self.encoder2 = nn.Sequential(
			nn.Linear(500, 2000),
			nn.ReLU(True),
			nn.Linear(2000, hidden)
		)
		self.global_ = nn.Sequential(
			nn.Linear(hidden, hidden),
			nn.ReLU(True),
			nn.Linear(hidden, cluster),
		)
		self.decoder = nn.Sequential(
			nn.Linear(hidden, 2000),
			nn.ReLU(True),
			nn.Linear(2000, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
			nn.Linear(500, input_size))
		self.Local = nn.Sequential(self.encoder1, self.encoder2)
		self.softmax = nn.Softmax(dim=1)
		self.relu = nn.ReLU(True)
		self.sigmod = nn.Sigmoid()
		#self.Global = nn.Sequential(self.encoder1, self.encoder2, self.global_, self.relu)
		self.Global = nn.Sequential(self.encoder1, self.encoder2, self.global_, self.softmax)

	def load_model(self, path):
		pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
		model_dict = self.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		self.load_state_dict(model_dict)

	def cluster_indice(self, x):
		return self.softmax(self.global_(x))

	def recon(self, z):
		return self.decoder(z)

	def forward(self, x, local=True):
		if(local):
			return self.Local(x)
		else:
			return self.Global(x)