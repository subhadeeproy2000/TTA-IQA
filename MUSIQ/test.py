from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr
import argparse
from option.config import Config
from model.model_main import IQARegression
from model.backbone import resnet50_backbone
from utils.util import RandHorizontalFlip, Normalize, ToTensor, RandShuffle

""" validation """


def eval_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader):

    checkpoint = torch.load(config.checkpoint)


    model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
    model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])

    losses = []

    model_transformer.eval()
    model_backbone.eval()

    # value is not changed
    mask_inputs = torch.ones(config.batch_size, config.n_enc_seq + 1).to(config.device)

    # save data for one epoch
    pred_epoch = []

    labels_epoch = []

    steps=0



    for data in tqdm(test_loader):

        data = data['image']

        d_img_org = data['d_img_org'].to(config.device)
        d_img_scale_1 = data['d_img_scale_1'].to(config.device)
        d_img_scale_2 = data['d_img_scale_2'].to(config.device)

        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)


        with torch.no_grad():

            pred = predict(model_backbone, model_transformer, d_img_org, d_img_scale_1, d_img_scale_2, mask_inputs)

        # compute loss
        loss = criterion(torch.squeeze(pred), labels)
        loss_val = loss.item()
        losses.append(loss_val)

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        steps+=1

        if steps%50==0:
            rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

            print('After {} images test_srcc : {}  \n test_plcc:{}'.format(steps,rho_s,rho_p))

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    print('test epoch:%d / loss:%f /SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p))
    return np.mean(losses), rho_s, rho_p

def predict(model_backbone,model_transformer,d_img_org,d_img_scale_1,d_img_scale_2,mask_inputs):
	feat_dis_org = model_backbone(d_img_org)
	feat_dis_scale_1 = model_backbone(d_img_scale_1)
	feat_dis_scale_2 = model_backbone(d_img_scale_2)

	pred = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2)

	return pred

parser = argparse.ArgumentParser()
parser.add_argument('--database',default='cid')
parser.add_argument('--datapath',default='CID2013')
parser.add_argument('--con', action='store_true')
parser.add_argument('--gc', action='store_true')
parser.add_argument('--rank', action='store_true')
parser.add_argument('--comp', action='store_true')
parser.add_argument('--nos', action='store_true')
parser.add_argument('--blur', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--online', action='store_true')
args = parser.parse_args()

# config file
config = Config({
    # device
    'gpu_id': "0",                          # specify GPU number to use
    'num_workers': 8,

    # data
    'db_name':args.database, #'KonIQ-10k', #'PIPAL', # 'cid', #                                    # database type
    'db_path': '/media/user/New Volume/Subhadeep/datasets/'+args.datapath,
    'txt_file_name': './IQA_list/koniq-10k.txt',                # list of images in the database
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
    'comp': args.comp,
    'nos': args.nos,
    'blur': args.blur,
    'contrastive': args.con,
    'rank': args.rank,
    'group_contrastive': args.gc,
    'rotation': args.rot,
    'online': args.online
})

# device setting
config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU %s' % config.gpu_id)
else:
    print('Using CPU')


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

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)


# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
    model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']


# train & validation

loss, rho_s, rho_p = eval_epoch(config, 0, model_transformer, model_backbone, criterion, train_loader)


