from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from utils.rotation import rotate_batch
from SSHead import head_on_layer2,ExtractorHead,extractor_from_layer2
import data_loader
import random
import json
from tqdm import tqdm
from scipy import stats
from models import TReS, Net



parser = argparse.ArgumentParser()
parser.add_argument('--datapath', dest='datapath', type=str,
                        default='provid the path to the dataset',
                        help='path to dataset')
parser.add_argument('--dataset', dest='dataset', type=str, default='csiq',
                    help='Support datasets: clive|koniq|fblive|live|csiq|tid2013')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--group_norm', default=0, type=int)
########################################################################
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--nepoch', default=5, type=int)
parser.add_argument('--milestone_1', default=50, type=int)
parser.add_argument('--milestone_2', default=65, type=int)
parser.add_argument('--rotation_type', default='rand')
########################################################################
parser.add_argument('--outf', default='.')
parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=50,
                        help='Number of sample patches from training image')
parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=50,
                    help='Number of sample patches from testing image')
parser.add_argument('--seed', dest='seed', type=int, default=2021,
                        help='for reproducing the results')
parser.add_argument('--vesion', dest='vesion', type=int, default=1,
                    help='vesion number for saving')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                    help='Crop size for training & testing image patches')
parser.add_argument('--svpath', dest='svpath', type=str,default='path to save the results',
                        help='the path to save the info')
parser.add_argument('--droplr', dest='droplr', type=int, default=5,
                        help='drop lr by every x iteration')
parser.add_argument('--gpunum', dest='gpunum', type=str, default='0',
                    help='the id for the gpu that will be used')
parser.add_argument('--network', dest='network', type=str, default='resnet50',
                        help='the resnet backbone to use')
parser.add_argument('--nheadt', dest='nheadt', type=int, default=16,
                        help='nheadt in the transformer')
parser.add_argument('--num_encoder_layerst', dest='num_encoder_layerst', type=int, default=2,
                        help='num encoder layers in the transformer')
parser.add_argument('--dim_feedforwardt', dest='dim_feedforwardt', type=int, default=64,
                    help='dim feedforward in the transformer')
parser.add_argument('--epochs', dest='epochs', type=int, default=2,
                        help='Epochs for training')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4,
                        help='Weight decay')
parser.add_argument('--auxilary_task', dest='auxilary_task', type=str, default='rotation',
                        help='specify the auxilary task')
parser.add_argument('--auxilary_task_loss', dest='auxilary_task_loss', type=str, default='cls',
                        help='specify the auxilary task loss function')
#--------------------------------------------------------------------------------------------
parser.add_argument('--fix_bn', action='store_true')
parser.add_argument('--fix_ssh', action='store_true')



args = parser.parse_args()
import os


my_makedir(args.outf)

if torch.cuda.is_available():
    if len(args.gpunum) == 1:
        device = torch.device("cuda", index=int(args.gpunum))
    else:
        device = torch.device("cpu")

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpunum

folder_path = {
    'live': args.datapath,
    'csiq': args.datapath,
    'tid2013': args.datapath,
    'kadid10k': args.datapath,
    'clive': args.datapath,
    'koniq': args.datapath,
    'fblive': args.datapath,
    'pipal': args.datapath,
'nnid':args.datapath
}

img_num = {
    'live': list(range(0, 29)),
    'csiq': list(range(0, 30)),
    'kadid10k': list(range(0, 80)),
    'tid2013': list(range(0, 25)),
    'clive': list(range(0, 1162)),
    'koniq': list(range(0, 10073)),
    'fblive': list(range(0, 39810)),
    'pipal': list(range(0, 11600)),
    'nnid': list(range(0, 450))
}

print('Training and Testing on {} dataset...'.format(args.dataset))

SavePath = args.svpath
svPath = SavePath + args.dataset + '_' + str(args.vesion) + '_' + str(args.seed) + '/' + 'sv'
os.makedirs(svPath, exist_ok=True)

# fix the seed if needed for reproducibility
if args.seed == 0:
    pass
else:
    print('we are using the seed = {}'.format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

total_num_images = img_num[args.dataset]

# Randomly select 80% images for training and the rest for testing
random.shuffle(total_num_images)
train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]

imgsTrainPath = svPath + '/' + 'train_index_' + str(args.vesion) + '_' + str(args.seed) + '.json'
imgsTestPath = svPath + '/' + 'test_index_' + str(args.vesion) + '_' + str(args.seed) + '.json'

with open(imgsTrainPath, 'w') as json_file2:
    json.dump(train_index, json_file2)

with open(imgsTestPath, 'w') as json_file2:
    json.dump(test_index, json_file2)

solver = TReS(args, device, svPath, folder_path[args.dataset], train_index, test_index, Net)
# solver.train_ssh(args.seed, svPath,args)
srcc_computed, plcc_computed = solver.train(args.seed, svPath,args)
