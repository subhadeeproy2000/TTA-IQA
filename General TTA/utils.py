import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import  Line2D
from PIL import Image
from torchvision import transforms
from PIL import ImageCms
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage
import pandas as pd
import torch.utils.data as data
import os.path
import scipy.io
import csv
import cv2
from pandas_ods_reader import read_ods
from skimage.util import random_noise
import scipy.ndimage


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def noisy(path,transform):

    sigma1 = 0.00005+ np.random.random() * 0.000001
    sigma2 = 0.00001+ np.random.random() * 0.000001   #low noise

    ab = cv2.imread(path)
    ab = cv2.cvtColor(ab, cv2.COLOR_BGR2RGB)

    noise=random_noise(ab, mode='gaussian',var=sigma1)
    image1 = Image.fromarray((noise * 255).astype('uint8'))
    image1 = transform(image1)

    noise=random_noise(ab, mode='gaussian',var=sigma2)
    image2 = Image.fromarray((noise * 255).astype('uint8'))
    image2 = transform(image2)

    return image2,image1


def compress(image,transform,root):

    sigma1 = 40 + np.random.random() * 20  # 40-60
    sigma2 = 80 + np.random.random() * 10  # 80-90

    try:
        image.save(root+"/Compressed_" + '1.jpg', optimize=True, quality=int(sigma1))
        image1 = Image.open(root+'/Compressed_1.jpg')
    except:
        image.save(root + "/Compressed_" + '1.bmp', optimize=True, quality=int(sigma1))
        image1 = Image.open(root + '/Compressed_1.bmp')
    image1 = transform(image1)

    try:
        image.save(root + "/Compressed_" + '2.jpg', optimize=True, quality=int(sigma2))
        image2 = Image.open(root+'/Compressed_2.jpg')
    except:
        image.save(root + "/Compressed_" + '2.bmp', optimize=True, quality=int(sigma2))
        image2 = Image.open(root + '/Compressed_2.bmp')
    image2 = transform(image2)

    return image1,image2


def processing(data_dict,sample,transform,root,path,config):
    if config.rank:
        data_dict['comp_high'], data_dict['comp_low'] = compress(sample,transform, root)
        data_dict['nos_low'], data_dict['nos_high'] = noisy(path,transform)

    if config.comp:
        data_dict['comp_high'], data_dict['comp_low'] = compress(sample, transform, root)

    if config.nos:
        data_dict['nos_low'], data_dict['nos_high'] = noisy(path,transform)

    if config.contrastive:
        data_dict['image1'], data_dict['image2'] = transform(sample), transform(sample)

    # if config.online:
    #     data_dict['online'] = []
    #     for i in range(32):
    #         data_dict['online'].append(transform(sample))

    data_dict['img_name'] = path

    if config.contrique:

        image_orig = Image.open(path)

        image_size = image_orig.size

        if image_orig.mode == 'L':
            image_orig = np.array(image_orig)
            image_orig = np.repeat(image_orig[:, :, None], 3, axis=2)
            image_orig = Image.fromarray(image_orig)
        elif image_orig.mode != 'RGB':
            image_orig = image_orig.convert('RGB')

        # Data augmentations

        # scaling transform and random crop
        div_factor = np.random.choice([1, 2], 1)[0]
        image_2 = ResizeCrop(image_orig, image_size, div_factor)

        # change colorspace
        colorspace_choice = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        image_2 = colorspaces(image_2, colorspace_choice)
        try:
            image_2 = transform(image_2)
        except:
            image_2 = ResizeCrop(image_orig, image_size, div_factor)
            # change colorspace
            image_2 = colorspaces(image_2, 2)
            image_2 = transform(image_2)

        # scaling transform and random crop
        image = ResizeCrop(image_orig, image_size, 3 - div_factor)

        # change colorspace
        colorspace_choice = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        image = colorspaces(image, colorspace_choice)
        try:
            image = transform(image)
        except:
            image = ResizeCrop(image_orig, image_size, 3 - div_factor)
            image = colorspaces(image, 2)
            image = transform(image)

        data_dict['image1'], data_dict['image2'] = image, image_2

    return data_dict

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