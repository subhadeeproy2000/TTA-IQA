from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


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
		return self.head(self.ext(x))
		# out1 =self.ext(x)
		# out=out1['target_in_vec']
		# return self.head(out)


def extractor_from_layer2(net):
	# layers = [net.model,net.model.avgpool,ViewFlatten(),net.fc]
	layers=[net]
	return nn.Sequential(*layers)


def head_on_layer2(config):
	head=[]
	head.append(nn.Linear(1000, 128))
	# head.append(nn.ReLU())
	if config.rotation:
		head.append(nn.Linear(128, 4))
	else:
		head.append(nn.Sigmoid())
	return nn.Sequential(*head)
