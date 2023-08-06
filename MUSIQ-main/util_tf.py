import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy


def head_on_layer2():
	head=[]
	head.append(nn.Linear(192, 128))
	head.append(nn.Sigmoid())
	return nn.Sequential(*head)

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		x=torch.flatten(x, start_dim=1)
		return x


class ExtractorHead(nn.Module):
	def __init__(self, ext, head,model_transformer,mask_inputs):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head
		self.model_transformer=model_transformer
		self.mask_inputs=mask_inputs

	def forward(self, x):

		feat_dis_org = self.ext(x['d_img_org'].cuda())
		feat_dis_scale_1 = self.ext(x['d_img_scale_1'].cuda())
		feat_dis_scale_2 = self.ext(x['d_img_scale_2'].cuda())

		feat = self.model_transformer(self.mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2,feat=True)
		return self.head(feat)

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

		loss_partial = -torch.log(nominator / (nominator+torch.sum(denominator, dim=1)))
		loss = torch.sum(loss_partial) / (2 * self.batch_size)

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

		loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
		loss = torch.sum(loss_partial) / (2 * self.batch_size)
		return loss

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def copy_model_and_optimizer(model, optimizer):
	"""Copy the model and optimizer states for resetting after adaptation."""
	model_state = deepcopy(model.state_dict())
	try:
		model_anchor = deepcopy(model)
	except:
		model_anchor=model
	optimizer_state = deepcopy(optimizer.state_dict())
	try:
		ema_model = deepcopy(model)
	except:
		ema_model=model
	for param in ema_model.parameters():
		param.detach_()
	return model_state, optimizer_state, ema_model, model_anchor