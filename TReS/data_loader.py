import torch
import torchvision
import torchvision.transforms.functional as F
import folders



class DataLoader(object):
	"""
	Dataset class for IQA databases
	"""

	def __init__(self,config, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):

		self.batch_size = batch_size
		self.istrain = istrain
		self.config=config

		if (dataset == 'live') | (dataset == 'csiq') | (dataset == 'tid2013') | (dataset == 'clive')| (dataset == 'kadid10k') | (dataset == 'pipal') | (dataset == 'nnid') | (dataset == 'cidiq1'):
				transforms = torchvision.transforms.Compose([
					# torchvision.transforms.RandomHorizontalFlip(),
					# torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.CenterCrop(size=patch_size),
					# torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))
				])
		elif dataset == 'koniq' or dataset=='cidiq':
				transforms = torchvision.transforms.Compose([
					# torchvision.transforms.RandomHorizontalFlip(),
					# torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.Resize((512, 384)),
					# torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.CenterCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))
												 ])
		elif dataset == 'pipal':

			if istrain:
				transforms = torchvision.transforms.Compose([
					# torchvision.transforms.RandomHorizontalFlip(),
					# torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.CenterCrop(size=patch_size),
					# torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))
				])
		elif dataset == 'dslr':

			if istrain:
				transforms = torchvision.transforms.Compose([
					# torchvision.transforms.RandomHorizontalFlip(),
					# torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.CenterCrop(size=patch_size),
					# torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))
				])
		elif dataset == 'spaq':
				transforms = torchvision.transforms.Compose([
					# torchvision.transforms.RandomHorizontalFlip(),
					# torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.Resize((512, 384)),
					torchvision.transforms.CenterCrop(size=patch_size),
					# torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))
				])
		elif dataset == 'fblive':
			if istrain:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.Resize((512, 512)),
					# torchvision.transforms.RandomHorizontalFlip(),
					# torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.CenterCrop(size=patch_size),
					# torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))])
		if dataset == 'live':
			self.data = folders.LIVEFolder(
				self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		if dataset == 'dslr':
			self.data = folders.DSLRFolder(
				self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'pipal':
			self.data = folders.PIPALFolder(
				self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'nnid':
			self.data = folders.NNIDFolder(
				self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'cidiq':
			self.data = folders.CIDIQFolder(
				self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'clive':
			self.data = folders.LIVEChallengeFolder(
				self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'spaq':
			self.data = folders.SPAQFolder(
				self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'csiq':
			self.data = folders.CSIQFolder(
				self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'koniq':
			self.data = folders.Koniq_10kFolder(
				self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'fblive':
			self.data = folders.FBLIVEFolder(
				self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'tid2013':
			self.data = folders.TID2013Folder(
				self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'kadid10k':
			self.data = folders.Kadid10k(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)			

	def get_data(self):
		if self.istrain:
			dataloader = torch.utils.data.DataLoader(
				self.data, batch_size=self.batch_size, shuffle=False)
		else:
			dataloader = torch.utils.data.DataLoader(
				self.data, batch_size=1, shuffle=False)
		return dataloader