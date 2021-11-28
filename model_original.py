import torch
from torch import nn
from torch.nn import Parameter

class AutoEncoder(nn.Module):
	def __init__(self, hidden, input_size):
		super(AutoEncoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_size, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
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
		self.model = nn.Sequential(self.encoder, self.decoder)

	def encode(self, x):
		return self.encoder(x)

	def decode(self, x):
		return self.decoder(x)

	def forward(self, x):
		return self.model(x)

class LGC(nn.Module):
	def __init__(self, hidden, input_size):
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
			nn.Linear(500, 2000),
			nn.ReLU(True),
			nn.Linear(2000, hidden)
		)
		self.Local = nn.Sequential(self.encoder1, self.encoder2)
		self.Global = nn.Sequential(self.encoder1, self.global_)

	def encoder(self, x):
		return self.encoder1(x)

	def Local_(self, x):
		return self.encoder2(x)

	def Global_(self, x):
		return self.global_(x)

	def load_model(self, path):
		pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
		model_dict = self.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		self.load_state_dict(model_dict)

	def Global_share_para(self):
		#self.encoder[l].conv1.weight.data.copy_(umaplayer[l].encoder.conv1.weight.data)
		self.global_[0].weight.data.copy_(self.encoder2[0].weight.data)
		self.global_[0].bias.data.copy_(self.encoder2[0].bias.data)
		self.global_[2].weight.data.copy_(self.encoder2[2].weight.data)
		self.global_[2].bias.data.copy_(self.encoder2[2].bias.data)

	def forward(self, x, local=True):
		if(local):
			return self.Local(x)
		else:
			return self.Global(x)