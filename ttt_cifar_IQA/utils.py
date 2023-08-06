import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import os
from transformers import Transformer
from posencode import PositionEmbeddingSine
from matplotlib import pyplot as plt
from matplotlib.lines import  Line2D
from PIL import Image
from torchvision import transforms
from PIL import ImageCms
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage

class GroupContrastiveLoss(nn.Module):
	def __init__(self, batch_size, temperature=0.5):
		super().__init__()
		self.batch_size = batch_size
		self.register_buffer("temperature", torch.tensor(temperature))
		self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
		self.register_buffer("positives_mask", (~torch.eye(batch_size * 1, batch_size * 1, dtype=bool)).float())

	def forward(self, emb_i, emb_j):
		"""
		emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
		z_i, z_j as per SimCLR paper
		"""

		self.negatives_mask[:len(emb_i), :len(emb_j)]=False
		self.negatives_mask[len(emb_i):, len(emb_j):] = False

		z_i = F.normalize(emb_i, dim=1)
		z_j = F.normalize(emb_j, dim=1)

		representations = torch.cat([z_i, z_j], dim=0)
		similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

		pos_similarity_matrix = similarity_matrix[:len(emb_i), :len(emb_j)]
		neg_similarity_matrix = similarity_matrix[len(emb_i):, len(emb_j):]

		pos_similarity_matrix = pos_similarity_matrix * self.positives_mask
		sim_ij=torch.sum(pos_similarity_matrix,dim=1)/(len(neg_similarity_matrix)-1)

		neg_similarity_matrix = neg_similarity_matrix * self.positives_mask
		sim_ji = torch.sum(neg_similarity_matrix, dim=1)/(len(neg_similarity_matrix)-1)

		positives = torch.cat([sim_ij, sim_ji], dim=0)

		nominator = torch.exp(positives / self.temperature)
		denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

		loss_partial = torch.sum(nominator / (nominator + torch.sum(denominator, dim=1)))/ (2 * self.batch_size)
		loss = -torch.log(loss_partial)

		return loss

class ContrastiveLoss(nn.Module):
	def __init__(self, batch_size, temperature=0.5):
		super().__init__()
		self.batch_size = batch_size
		self.register_buffer("temperature", torch.tensor(temperature))
		self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

	def forward(self, emb_i, emb_j):
		"""
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
		z_i = F.normalize(emb_i, dim=1)
		z_j = F.normalize(emb_j, dim=1)

		representations = torch.cat([z_i, z_j], dim=0)
		similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

		sim_ij = torch.diag(similarity_matrix, self.batch_size)
		sim_ji = torch.diag(similarity_matrix, -self.batch_size)
		positives = torch.cat([sim_ij, sim_ji], dim=0)

		nominator = torch.exp(positives / self.temperature)
		denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

		loss_partial = -torch.log(nominator / (nominator+torch.sum(denominator, dim=1)))
		loss = torch.sum(loss_partial) / (2 * self.batch_size)

		return loss

class L2pooling(nn.Module):
	def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
		super(L2pooling, self).__init__()
		self.padding = (filter_size - 2 )//2
		self.stride = stride
		self.channels = channels
		a = np.hanning(filter_size)[1:-1]
		g = torch.Tensor(a[:,None]*a[None,:])
		g = g/torch.sum(g)
		self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))
	def forward(self, input):
		input = input**2
		out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
		return (out+1e-12).sqrt()


class Net(nn.Module):
	def __init__(self, cfg, device):
		super(Net, self).__init__()

		self.device = device

		self.cfg = cfg
		self.L2pooling_l1 = L2pooling(channels=256)
		self.L2pooling_l2 = L2pooling(channels=512)
		self.L2pooling_l3 = L2pooling(channels=1024)
		self.L2pooling_l4 = L2pooling(channels=2048)

		if cfg.network == 'resnet50':
			from resnet_modify import resnet50 as resnet_modifyresnet
			# def gn_helper(planes):
			# 	return nn.GroupNorm(8, planes)
			# norm_layer = gn_helper
			dim_modelt = 3840
			modelpretrain = models.resnet50(pretrained=True)

		elif cfg.network == 'resnet34':
			from resnet_modify import resnet34 as resnet_modifyresnet
			modelpretrain = models.resnet34(pretrained=True)
			dim_modelt = 960
			self.L2pooling_l1 = L2pooling(channels=64)
			self.L2pooling_l2 = L2pooling(channels=128)
			self.L2pooling_l3 = L2pooling(channels=256)
			self.L2pooling_l4 = L2pooling(channels=512)
		elif cfg.network == 'resnet18':
			from resnet_modify import resnet18 as resnet_modifyresnet
			modelpretrain = models.resnet18(pretrained=True)
			dim_modelt = 960
			self.L2pooling_l1 = L2pooling(channels=64)
			self.L2pooling_l2 = L2pooling(chFannels=128)
			self.L2pooling_l3 = L2pooling(channels=256)
			self.L2pooling_l4 = L2pooling(channels=512)

		torch.save(modelpretrain.state_dict(), 'modelpretrain')

		self.model = resnet_modifyresnet()

		self.dim_modelt = dim_modelt

		os.remove("modelpretrain")

		nheadt = cfg.nheadt
		num_encoder_layerst = cfg.num_encoder_layerst
		dim_feedforwardt = cfg.dim_feedforwardt
		ddropout = 0.5
		normalize = True

		self.transformer = Transformer(d_model=dim_modelt, nhead=nheadt,
									   num_encoder_layers=num_encoder_layerst,
									   dim_feedforward=dim_feedforwardt,
									   normalize_before=normalize,
									   dropout=ddropout)

		self.position_embedding = PositionEmbeddingSine(dim_modelt // 2, normalize=True)

		self.fc2 = nn.Linear(dim_modelt, self.model.fc.in_features)
		self.fc = nn.Linear(self.model.fc.in_features * 2, 1)
		self.ReLU = nn.ReLU()
		self.avg7 = nn.AvgPool2d((7, 7))
		self.avg8 = nn.AvgPool2d((8, 8))
		self.avg4 = nn.AvgPool2d((4, 4))
		self.avg2 = nn.AvgPool2d((2, 2))
		self.drop2d = nn.Dropout(p=0.1)
		self.consistency = nn.L1Loss()

	def forward(self, x):
		self.pos_enc_1 = self.position_embedding(torch.ones(1, self.dim_modelt, 7, 7).to(self.device))
		self.pos_enc = self.pos_enc_1.repeat(x.shape[0], 1, 1, 1).contiguous()

		out, layer1, layer2, layer3, layer4 = self.model(x)

		layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1, dim=1, p=2))))
		layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2, dim=1, p=2))))
		layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3, dim=1, p=2))))
		layer4_t = self.drop2d(self.L2pooling_l4(F.normalize(layer4, dim=1, p=2)))
		layers = torch.cat((layer1_t, layer2_t, layer3_t, layer4_t), dim=1)

		out_t_c = self.transformer(layers, self.pos_enc)
		out_t_o = torch.flatten(self.avg7(out_t_c), start_dim=1)
		out_t_o = self.fc2(out_t_o)
		layer4_o = self.avg7(layer4)
		layer4_o = torch.flatten(layer4_o, start_dim=1)
		predictionQA = self.fc(torch.flatten(torch.cat((out_t_o, layer4_o), dim=1), start_dim=1))

		# =============================================================================
		# =============================================================================

		fout, flayer1, flayer2, flayer3, flayer4 = self.model(torch.flip(x, [3]))
		flayer1_t = self.avg8(self.L2pooling_l1(F.normalize(flayer1, dim=1, p=2)))
		flayer2_t = self.avg4(self.L2pooling_l2(F.normalize(flayer2, dim=1, p=2)))
		flayer3_t = self.avg2(self.L2pooling_l3(F.normalize(flayer3, dim=1, p=2)))
		flayer4_t = self.L2pooling_l4(F.normalize(flayer4, dim=1, p=2))
		flayers = torch.cat((flayer1_t, flayer2_t, flayer3_t, flayer4_t), dim=1)
		fout_t_c = self.transformer(flayers, self.pos_enc)
		fout_t_o = torch.flatten(self.avg7(fout_t_c), start_dim=1)
		fout_t_o = (self.fc2(fout_t_o))
		flayer4_o = self.avg7(flayer4)
		flayer4_o = torch.flatten(flayer4_o, start_dim=1)
		fpredictionQA = (self.fc(torch.flatten(torch.cat((fout_t_o, flayer4_o), dim=1), start_dim=1)))

		consistloss1 = self.consistency(out_t_c, fout_t_c.detach())
		consistloss2 = self.consistency(layer4, flayer4.detach())
		consistloss = 1 * (consistloss1 + consistloss2)

		return predictionQA, consistloss

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

def ResizeCrop(image, sz, div_factor):
    image_size = image.size
    image = transforms.Resize([image_size[1] // div_factor, \
                               image_size[0] // div_factor])(image)

    if image.size[1] < sz[0] or image.size[0] < sz[1]:
        # image size smaller than crop size, zero pad to have same size
        image = transforms.CenterCrop(sz)(image)
    else:
        image = transforms.RandomCrop(sz)(image)

    return image


def compute_MS_transform(image, window, extend_mode='reflect'):
    h, w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    scipy.ndimage.correlate1d(image, window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, window, 1, mu_image, mode=extend_mode)
    return image - mu_image


def MS_transform(image):
    #   MS Transform
    image = np.array(image).astype(np.float32)
    window = gen_gauss_window(3, 7 / 6)
    image[:, :, 0] = compute_MS_transform(image[:, :, 0], window)
    image[:, :, 0] = (image[:, :, 0] - np.min(image[:, :, 0])) / (np.ptp(image[:, :, 0]) + 1e-3)
    image[:, :, 1] = compute_MS_transform(image[:, :, 1], window)
    image[:, :, 1] = (image[:, :, 1] - np.min(image[:, :, 1])) / (np.ptp(image[:, :, 1]) + 1e-3)
    image[:, :, 2] = compute_MS_transform(image[:, :, 2], window)
    image[:, :, 2] = (image[:, :, 2] - np.min(image[:, :, 2])) / (np.ptp(image[:, :, 2]) + 1e-3)

    image = Image.fromarray((image * 255).astype(np.uint8))
    return

def colorspaces(im, val):
    if val == 0:
        im = transforms.RandomGrayscale(p=1.0)(im)
    elif val == 1:
        srgb_p = ImageCms.createProfile("sRGB")
        lab_p  = ImageCms.createProfile("LAB")

        rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        im = ImageCms.applyTransform(im, rgb2lab)
    elif val == 2:
         im = im.convert('HSV')
    elif val == 3:
         im = MS_transform(im)
    return im


def Sort_Tuple(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup.sort(key=lambda x: x[1],reverse=True)
    return tup