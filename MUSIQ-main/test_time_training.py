from torchvision import transforms
from torch.utils.data import DataLoader
import random
import os
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import copy
import torchvision.transforms as T
from rotation import *
import argparse


from option.config import Config
from model.model_main import IQARegression
from model.backbone import resnet50_backbone
from utils.util import RandHorizontalFlip, Normalize, ToTensor, RandShuffle
from util import *

""" validation """

def eval_epoch(config, model_transformer, model_backbone, criterion, test_loader):

	checkpoint = torch.load(config.checkpoint)
	model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
	model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])

	losses = []
	losses_old = []

	model_transformer.eval()
	model_backbone.eval()

	# value is not changed
	mask_inputs = torch.ones(config.batch_size, config.n_enc_seq + 1).to(config.device)

	# save data for one epoch
	pred_epoch = []
	pred_epoch_old = []

	labels_epoch = []

	steps=0

	old_net = copy.deepcopy(model_backbone)

	for dict in tqdm(test_loader):

		data=dict['image']

		if not config.online:

			model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])

		d_img_org = data['d_img_org'].to(config.device)
		d_img_scale_1 = data['d_img_scale_1'].to(config.device)
		d_img_scale_2 = data['d_img_scale_2'].to(config.device)

		labels = data['score']
		labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

		old_net.load_state_dict(checkpoint['model_backbone_state_dict'])

		if config.group_contrastive:
			if len(d_img_org ) > 3:
				loss_hist = adapt(model_backbone,dict, config,optimizer,old_net, model_transformer,mask_inputs)
			else:
				if config.rank or config.blur or config.comp or config.nos or config.contrastive or config.rotation:
					config.group_contrastive=False
					loss_hist = adapt(model_backbone,dict, config,optimizer,old_net, model_transformer,mask_inputs)
		elif config.rank or config.blur or config.comp or config.nos or config.contrastive or config.rotation:
			loss_hist = adapt(model_backbone, dict, config, optimizer,old_net, model_transformer,mask_inputs)

		old_net.load_state_dict(checkpoint['model_backbone_state_dict'])

		with torch.no_grad():

			old_pred=predict(old_net, model_transformer, d_img_org, d_img_scale_1, d_img_scale_2, mask_inputs)

		with torch.no_grad():

			pred = predict(model_backbone, model_transformer, d_img_org, d_img_scale_1, d_img_scale_2, mask_inputs)

		# compute loss
		loss_old = criterion(torch.squeeze(old_pred), labels)
		loss_val_old = loss_old.item()
		losses_old.append(loss_val_old)

		loss = criterion(torch.squeeze(pred), labels)
		loss_val = loss.item()
		losses.append(loss_val)

		# save results in one epoch
		pred_batch_numpy = pred.data.cpu().numpy()
		labels_batch_numpy = labels.data.cpu().numpy()
		pred_epoch = np.append(pred_epoch, pred_batch_numpy)
		labels_epoch = np.append(labels_epoch, labels_batch_numpy)

		pred_batch_numpy = old_pred.data.cpu().numpy()
		pred_epoch_old = np.append(pred_epoch_old, pred_batch_numpy)

		steps+=1

		if steps%20==0:

			rho_s_old, _ = spearmanr(np.squeeze(pred_epoch_old), np.squeeze(labels_epoch))
			rho_p_old, _ = pearsonr(np.squeeze(pred_epoch_old), np.squeeze(labels_epoch))
			rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
			rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

			print('After {} images test_srcc old : {}  new {} \n test_plcc old:{} new:{}'.format(steps, rho_s_old,rho_s,rho_p_old,rho_p))

	# compute correlation coefficient
	rho_s_old, _ = spearmanr(np.squeeze(pred_epoch_old), np.squeeze(labels_epoch))
	rho_p_old, _ = pearsonr(np.squeeze(pred_epoch_old), np.squeeze(labels_epoch))
	rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
	rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

	print(' loss:%f /SROCC:%4f / PLCC:%4f' % (loss.item(), rho_s, rho_p))
	print(' test_srcc old : {}  new {} \n test_plcc old:{} new:{}'.format( rho_s_old, rho_s, rho_p_old, rho_p))
	return np.mean(losses), rho_s, rho_p

def predict(model_backbone,model_transformer,d_img_org,d_img_scale_1,d_img_scale_2,mask_inputs):
	feat_dis_org = model_backbone(d_img_org)
	feat_dis_scale_1 = model_backbone(d_img_scale_1)
	feat_dis_scale_2 = model_backbone(d_img_scale_2)

	pred = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2)

	return pred


def adapt(model, data_dict, config, optimizer,old_net, model_transformer,mask_inputs):

		inputs = data_dict['image']['d_img_org']

		f_low = []
		f_high = []

		with torch.no_grad():
			pred0=predict(old_net, model_transformer, data_dict['image']['d_img_org'].cuda(), data_dict['image']['d_img_scale_1'].cuda(), data_dict['image']['d_img_scale_2'].cuda(), mask_inputs)

			if config.rank:

				sigma1 = 40 + np.random.random() * 20
				sigma2 = 5 + np.random.random() * 15

				data_dict['blur_low']={}
				data_dict['blur_high']={}

				data_dict['blur_high']['d_img_org'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(inputs).cuda()
				data_dict['blur_low']['d_img_org'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(inputs).cuda()

				id_dict = {0: data_dict['comp_high'], 1: data_dict['comp_low'], 2: data_dict['nos_high'],
						   3: data_dict['nos_low'], 4: data_dict['blur_high'], 5: data_dict['blur_low']}

				pred1 = predict(old_net, model_transformer, data_dict['comp_high']['d_img_org'].cuda(), data_dict['comp_high']['d_img_scale_1'].cuda(), data_dict['comp_high']['d_img_scale_2'].cuda(), mask_inputs)
				pred2= predict(old_net, model_transformer, data_dict['comp_low']['d_img_org'].cuda(), data_dict['comp_low']['d_img_scale_1'].cuda(), data_dict['comp_low']['d_img_scale_2'].cuda(), mask_inputs)
				comp = torch.abs(pred2 - pred1)


				pred3 = predict(old_net, model_transformer, data_dict['nos_high']['d_img_org'].cuda(), data_dict['nos_high']['d_img_scale_1'].cuda(), data_dict['nos_high']['d_img_scale_2'].cuda(), mask_inputs)
				pred4 = predict(old_net, model_transformer, data_dict['nos_low']['d_img_org'].cuda(), data_dict['nos_low']['d_img_scale_1'].cuda(), data_dict['nos_low']['d_img_scale_2'].cuda(), mask_inputs)
				nos = torch.abs(pred4 - pred3)

				pred5 = predict(old_net, model_transformer, data_dict['blur_high']['d_img_org'].cuda(), data_dict['image']['d_img_scale_1'].cuda(), data_dict['image']['d_img_scale_2'].cuda(), mask_inputs)
				pred6 = predict(old_net, model_transformer, data_dict['blur_low']['d_img_org'].cuda(), data_dict['image']['d_img_scale_1'].cuda(), data_dict['image']['d_img_scale_2'].cuda(), mask_inputs)
				blur = torch.abs(pred6 - pred5)

				all_diff = torch.cat([comp, nos, blur], dim=1)

				for p in range(len(pred0)):
					if all_diff[p].argmax().item() == 0:
						f_low.append(id_dict[1]['d_img_org'][p].cuda())
						f_high.append(id_dict[0]['d_img_org'][p].cuda())
					if all_diff[p].argmax().item() == 1:
						f_low.append(id_dict[3]['d_img_org'][p].cuda())
						f_high.append(id_dict[2]['d_img_org'][p].cuda())
					if all_diff[p].argmax().item() == 2:
						f_low.append(id_dict[5]['d_img_org'][p].cuda())
						f_high.append(id_dict[4]['d_img_org'][p].cuda())

				f_low = torch.squeeze(torch.stack(f_low), dim=1)
				f_high = torch.squeeze(torch.stack(f_high), dim=1)

		if config.comp:
			f_low = data_dict['comp_low']['d_img_org'].cuda()
			f_high = data_dict['comp_high']['d_img_org'].cuda()
		if config.nos:
			f_low = data_dict['nos_low']['d_img_org'].cuda()
			f_high = data_dict['nos_high']['d_img_org'].cuda()
		if config.blur:
			sigma1 = 40 + np.random.random() * 20
			sigma2 = 5 + np.random.random() * 15

			f_low = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(inputs).cuda()
			f_high = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(inputs).cuda()

		if config.contrastive:
			f_low = data_dict['image1']['d_img_org'].cuda()
			f_high = data_dict['image2']['d_img_org'].cuda()

		m = nn.Sigmoid()

		for param in model.parameters():
			param.requires_grad = False

		for layer in model.modules():
			if isinstance(layer, nn.BatchNorm2d):
				layer.requires_grad_(True)

		loss_hist=[]

		head = head_on_layer2()
		ssh = ExtractorHead(model, head).cuda()

		ssh.train()

		for iteration in range(config.niter):

			target = torch.ones(inputs.shape[0]).cuda()

			if config.rank or config.blur or config.comp or config.nos:

				f_neg_feat = ssh(f_low)
				f_pos_feat = ssh(f_high)
				f_actual = ssh(inputs.cuda())

				dist_high = torch.nn.PairwiseDistance(p=2)(f_pos_feat, f_actual)
				dist_low = torch.nn.PairwiseDistance(p=2)(f_neg_feat, f_actual)

				loss = nn.BCELoss()(m(dist_high - dist_low), target)

			if config.contrastive:
				f_neg_feat = ssh(f_low)
				f_pos_feat = ssh(f_high)

				loss_fn = ContrastiveLoss(f_pos_feat.shape[0], 1).cuda()

				loss = loss_fn(f_neg_feat, f_pos_feat)

			if config.group_contrastive:

				idx=np.argsort(pred0.cpu(),axis=0)

				f_feat=ssh(inputs.cuda())

				f_pos_feat=[]
				f_neg_feat=[]

				for n in range(len(f_feat)//4):
					f_pos_feat.append(f_feat[idx[n]])
					f_neg_feat.append(f_feat[idx[-n-1]])

				try:
					f_pos_feat=torch.squeeze(torch.stack(f_pos_feat),dim=1)
					f_neg_feat=torch.squeeze(torch.stack(f_neg_feat),dim=1)
				except:
					print('Batch size is too less')
					continue

				loss_fn = GroupContrastiveLoss(f_pos_feat.shape[0], 0.1).cuda()


				if config.rank or config.blur or config.comp or config.nos:
					loss+=loss_fn(f_neg_feat,f_pos_feat)*config.wth
				else:
					loss=loss_fn(f_neg_feat,f_pos_feat)


			if config.rotation:
				inputs_ssh, labels_ssh = rotate_batch(inputs.cuda(), 'rand')
				outputs_ssh = ssh(inputs_ssh.float())
				loss = nn.CrossEntropyLoss()(outputs_ssh, labels_ssh.cuda())

			loss.backward()
			optimizer.step()
			loss_hist.append(loss.detach().cpu())

		# print(loss_hist)
		return loss_hist



parser = argparse.ArgumentParser()
parser.add_argument('--database',default='cid')
parser.add_argument('--datapath',default='CID2013')
parser.add_argument('--weight', type=float, default=1,help='Weight decay')
parser.add_argument('--con', action='store_true')
parser.add_argument('--gc', action='store_true')
parser.add_argument('--rank', action='store_true')
parser.add_argument('--comp', action='store_true')
parser.add_argument('--nos', action='store_true')
parser.add_argument('--blur', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--online', action='store_true')
parser.add_argument('--seed', dest='seed', type=int, default=2021,
                        help='for reproducing the results')
parser.add_argument('--run', dest='run', type=int, default=1,
                        help='for running at multiple seeds')
args = parser.parse_args()

# config file
config = Config({
    # device
    'gpu_id': "0",                          # specify GPU number to use
    'num_workers': 8,
	'wth':args.weight,
    # data
    'db_name':args.database, #'KonIQ-10k', #'PIPAL', # 'cid', #                                    # database type
    'db_path': '/media/user/New Volume/Subhadeep/datasets/'+args.datapath,      # root path of database
    'txt_file_name': 'txt_file_name.txt',                # list of images in the database
    'train_size': 1,                                          # train/vaildation separation ratio
    'scenes': 'all',                                            # using all scenes
    'scale_1': 384,
    'scale_2': 224,
    'batch_size': 8,
    'patch_size': 7,

    # ViT structure
    'n_enc_seq': 7*7 + 12*12+ 7*5,        # input feature map dimension (N = H*W) from backbone
    'n_layer': 14,                          # number of encoder layers
    'd_hidn': 192,                          # input channel of encoder (input: C x N)
    'i_pad': 0,
    'd_ff': 192,                            # feed forward hidden layer dimension
    'd_MLP_head': 1152,                     # hidden layer of final MLP
    'n_head': 6,                            # number of head (in multi-head attention)
    'd_head': 384,                          # channel of each head -> same as d_hidn
    'dropout': 0.1,                         # dropout ratio
    'emb_dropout': 0.1,                     # dropout ratio of input embedding
    'layer_norm_epsilon': 1e-12,
    'n_output': 1,                          # dimension of output
    'Grid': 10,                             # grid of 2D spatial embedding

    # optimization & training parameters
    'n_epoch': 1,                         # total training epochs
    'learning_rate': 1e-4,                  # initial learning rate
    'weight_decay': 0,                      # L2 regularization weight
    'momentum': 0.9,                        # SGD momentum
    'T_max': 3e4,                           # period (iteration) of cosine learning rate decay
    'eta_min': 0,                           # minimum learning rate
    'save_freq': 10,                        # save checkpoint frequency (epoch)
    'val_freq': 5,                          # validation frequency (epoch)


    # load & save checkpoint
    'snap_path': './weights',               # directory for saving checkpoint
    'checkpoint': './weights/fblive1_epoch34.pth',                     # load checkpoint
	'niter':3,
	'comp':args.comp,
	'nos':args.nos,
	'blur':args.blur,
	'contrastive':args.con,
	'rank':args.rank,
	'group_contrastive':args.gc,
	'rotation':args.rot,
	'online':args.online
})


# device setting
config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU %s' % config.gpu_id)
else:
    print('Using CPU')

rho_s_list, rho_p_list=[],[]

for mul in range(args.run):

	# fix the seed if needed for reproducibility
	if args.seed == 0:
		pass
	else:
		if mul!=0:
			args.seed=args.seed+np.random.randint(1000)
		print('we are using the seed = {}'.format(args.seed))
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)
		np.random.seed(args.seed)
		random.seed(args.seed)


	# data selection
	if config.db_name == 'KonIQ-10k':
		from data.koniq import IQADataset
	if config.db_name == 'livefb':
		from data.livefb import IQADataset
	if config.db_name == 'PIPAL':
		from data.pipal import IQADataset
	if config.db_name == 'cid':
		from data.cidiq import IQADataset
	if config.db_name == 'LIVE':
		from data.live import IQADataset
	if config.db_name == 'tid2013':
		from data.tid import IQADataset
	if config.db_name == 'dslr':
		from data.dslr import IQADataset

	# dataset separation (8:2)
	train_scene_list, test_scene_list = RandShuffle(config)
	print('number of train scenes: %d' % len(train_scene_list))
	print('number of test scenes: %d' % len(test_scene_list))

	# data load
	train_dataset = IQADataset(
		config=config,
		db_path=config.db_path,
		txt_file_name=config.txt_file_name,
		scale_1=config.scale_1,
		scale_2=config.scale_2,
		transform=transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), RandHorizontalFlip(), ToTensor()]),
		train_mode=True,
		scene_list=train_scene_list,
		train_size=config.train_size
	)

	train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)


	# create model
	model_backbone = resnet50_backbone().to(config.device)
	model_transformer = IQARegression(config).to(config.device)


	# loss function & optimization
	criterion = torch.nn.L1Loss()
	params = list(model_backbone.parameters()) + list(model_transformer.parameters())
	optimizer = torch.optim.Adam(model_backbone.parameters(), lr=config.learning_rate,weight_decay=config.weight_decay)

	# load weights & optimizer
	if config.checkpoint is not None:
		checkpoint = torch.load(config.checkpoint)
		model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
		model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])

	# make directory for saving weights
	if not os.path.exists(config.snap_path):
		os.mkdir(config.snap_path)

	# train & validation

	loss, rho_s, rho_p = eval_epoch(config, model_transformer, model_backbone, criterion, train_loader)


	rho_s_list.append(rho_s)
	rho_p_list.append(rho_p)

final_rho_s=np.mean(np.array(rho_s_list))
final_rho_p=np.mean(np.array(rho_p_list))

print(' final_srcc new {} \n final_plcc new:{}'.format(final_rho_s,final_rho_p))


