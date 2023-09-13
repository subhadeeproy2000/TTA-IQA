from utils import *

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
		dim_modelt = 3840
		self.position_embedding = PositionEmbeddingSine(dim_modelt // 2, normalize=True)
		self.dim_modelt = dim_modelt
		self.avg7 = nn.AvgPool2d((7, 7))
		self.avg8 = nn.AvgPool2d((8, 8))
		self.avg4 = nn.AvgPool2d((4, 4))
		self.avg2 = nn.AvgPool2d((2, 2))
		self.drop2d = nn.Dropout(p=0.1)
		self.L2pooling_l1 = L2pooling(channels=256)
		self.L2pooling_l2 = L2pooling(channels=512)
		self.L2pooling_l3 = L2pooling(channels=1024)
		self.L2pooling_l4 = L2pooling(channels=2048)

	def forward(self, x):
		self.pos_enc_1 = self.position_embedding(torch.ones(1, self.dim_modelt, 7, 7).cuda())
		self.pos_enc = self.pos_enc_1.repeat(x.shape[0], 1, 1, 1).contiguous()

		out, layer1, layer2, layer3, layer4 = self.ext[0](x)

		layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1, dim=1, p=2))))
		layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2, dim=1, p=2))))
		layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3, dim=1, p=2))))
		layer4_t = self.drop2d(self.L2pooling_l4(F.normalize(layer4, dim=1, p=2)))
		layers = torch.cat((layer1_t, layer2_t, layer3_t, layer4_t), dim=1)

		out_t_c = self.ext[1](layers, self.pos_enc)

		out=torch.flatten(self.ext[2](out_t_c), start_dim=1)

		return self.head(out)


def extractor_from_layer2(net):
	layers = [net.model,net.transformer,net.avg7]
	return nn.Sequential(*layers)


def head_on_layer2(config):
	head=[]
	head.append(nn.Linear(3840, 256))
	if config.rotation:
		head.append(nn.Linear(256, 4))
	else:
		head.append(nn.Sigmoid())
	return nn.Sequential(*head)
