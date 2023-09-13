from torch import nn
import torch

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		x=torch.flatten(x, start_dim=1)
		return x

class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head


	def forward(self, x):
		out1 =self.ext[0](x)
		out=self.ext[1:3](out1[-1])
		return self.head(out)


def extractor_from_layer2(net):
	layers = [net.model,net.model.avgpool,ViewFlatten(),net.fc]
	return nn.Sequential(*layers)


def head_on_layer2(config):
	head=[]
	head.append(nn.Linear(2048, 256))
	if config.rotation:
		head.append(nn.Linear(256, 4))
	else:
		head.append(nn.Sigmoid())
	return nn.Sequential(*head)
