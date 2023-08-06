from __future__ import print_function
import argparse
import random
import data_loader
import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='KONIQ')
parser.add_argument('--datapath', default='..')
########################################################################
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--ng', default=2, type=int)
parser.add_argument('--fix_ssh', action='store_true')
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=3, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument('--weight', type=float, default=1,help='Weight for rank plus GC')
parser.add_argument('--p', type=float, default=0.25,help='p for GC loss')
########################################################################
parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1,
                    help='Number of sample patches from testing image')
parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=1,
                    help='Number of sample patches from training image')
parser.add_argument('--seed', dest='seed', type=int, default=2021,
                        help='for reproducing the results')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                    help='Crop size for training & testing image patches')
parser.add_argument('--svpath', dest='svpath', type=str,default='/home/user/Subhadeep/ttt_cifar_IQA/Save_RES/',
                        help='the path to save the info')
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
parser.add_argument('--train_data', dest='train_data', type=str, default='fblive',
                        help='give the train dataset name')
parser.add_argument('--contrastive', action='store_true')
parser.add_argument('--group_contrastive', action='store_true')
parser.add_argument('--rank', action='store_true')
parser.add_argument('--comp', action='store_true')
parser.add_argument('--contrique', action='store_true')
parser.add_argument('--nos', action='store_true')
parser.add_argument('--blur', action='store_true')
parser.add_argument('--sr', action='store_true')
parser.add_argument('--rotation', action='store_true')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--online_mse', action='store_true')
parser.add_argument('--tta_tf', action='store_true')
parser.add_argument('--bn', action='store_true')
parser.add_argument('--ln', action='store_true')
parser.add_argument('--run', dest='run', type=int, default=1,
                        help='for running at multiple seeds')




args = parser.parse_args()
args.threshold += 0.001		# to correct for numeric errors
args.datapath='/media/user/New Volume/Subhadeep/datasets/'+args.datapath

from models import TReS, Net
if args.ng>2:
    from models_multiple_group_GC_loss import TReS, Net

if torch.cuda.is_available():
    if len(args.gpunum) == 1:
        device = torch.device("cuda", index=int(args.gpunum))
else:
    device = torch.device("cpu")

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

#----------------------------------------
folder_path = {
    'live': args.datapath,
    'csiq': args.datapath,
    'tid2013': args.datapath,
    'kadid10k': args.datapath,
    'clive': args.datapath,
    'koniq': args.datapath,
    'fblive': args.datapath,
    'pipal': args.datapath,
'cidiq': args.datapath,
    'nnid':args.datapath,
    'spaq': args.datapath,
    'dslr': args.datapath
}

img_num = {
    'live': list(range(0, 29)),
    'csiq': list(range(0, 30)),
    'kadid10k': list(range(0, 80)),
    'tid2013': list(range(0, 25)),
    'clive': list(range(0, 1162)),
    'koniq': list(range(0, 10073)),
    'fblive': list(range(0, 39810)),
    'pipal': list(range(0, 23200)),
'cidiq': list(range(0, 475)),
    'nnid':list(range(0,450)),
    'spaq': list(range(0, 11125)),
    'dslr': list(range(0, 1035))
}

print('Testing on {} dataset...'.format(args.dataset))

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

    sel_num=img_num[args.dataset]

    # Randomly select 80% images for training and the rest for testing
    random.shuffle(sel_num)
    train_index = sel_num[0:int(round(0.80 * len(sel_num)))]
    # test_index = sel_num[int(round(0.80 * len(sel_num))):]
    test_index = sel_num


    test_loader = data_loader.DataLoader(args,args.dataset, folder_path[args.dataset],
                                         test_index, args.patch_size,
                                         args.test_patch_num,batch_size=args.batch_size,istrain=True)
    test_data = test_loader.get_data()

    solver = TReS(args, device, Net)

    if args.test_only:
        srcc_computed, plcc_computed,_,_= solver.test(test_data,pretrained=1)
        print('srcc_computed_test {}, plcc_computed_test {}'.format(srcc_computed, plcc_computed))
    else:
        test_srcc_old,test_plcc_old,srcc_computed, plcc_computed =solver.new_ttt(test_data,args)
        print('srcc_computed {}, plcc_computed {}'.format(srcc_computed, plcc_computed))

    rho_s_list.append(srcc_computed)
    rho_p_list.append(plcc_computed)

final_rho_s=np.mean(np.array(rho_s_list))
final_rho_p=np.mean(np.array(rho_p_list))

if not args.test_only:
    print('final_srcc old {}, final_plcc old {}'.format(test_srcc_old, test_plcc_old))
print(' final_srcc new {}, final_plcc new:{}'.format(final_rho_s,final_rho_p))
