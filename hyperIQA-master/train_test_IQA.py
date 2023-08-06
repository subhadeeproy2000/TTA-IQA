import os
import argparse
import random
import numpy as np
import torch


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(config):
    folder_path = {
        'live': config.datapath,
        'csiq': config.datapath,
        'tid2013': config.datapath,
        'kadid10k': config.datapath,
        'clive': config.datapath,
        'koniq': config.datapath,
        'fblive': config.datapath,
        'pipal': config.datapath,
        'cidiq': config.datapath,
        'dslr': config.datapath
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013': list(range(0, 25)),
        'clive': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
        'fblive': list(range(0, 39810)),
        'pipal': list(range(0, 5800)),
        'cidiq': list(range(0, 475)),
        'dslr': list(range(0, 1035))
    }

    from HyerIQASolver import HyperIQASolver

    svPath = config.svpath
    os.makedirs(svPath, exist_ok=True)

    rho_s_list, rho_p_list = [], []

    for mul in range(config.run):

        # fix the seed if needed for reproducibility
        if config.seed == 0:
            pass
        else:
            if mul != 0:
                config.seed = config.seed + np.random.randint(1000)
            print('we are using the seed = {}'.format(config.seed))
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)

        sel_num = img_num[config.dataset]

        # Randomly select 80% images for training and the rest for testing
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        # test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        test_index = sel_num


        solver = HyperIQASolver(config, folder_path[config.dataset], train_index, test_index)


        if config.test:
            srcc_computed1, plcc_computed1= solver.test(config.svpath,config.seed,pretrained=True)
            print('srcc_computed_test {}, plcc_computed_test {}'.format(srcc_computed1, plcc_computed1))
            continue
        else:
            test_srcc_old, test_plcc_old, srcc_computed, plcc_computed = solver.new_ttt(config.svpath,config)
            print('srcc_computed {}, plcc_computed {}'.format(srcc_computed, plcc_computed))
            print('srcc_computed_old {}, plcc_computed_old {}'.format(test_srcc_old, test_plcc_old))
            rho_s_list.append(srcc_computed)
            rho_p_list.append(plcc_computed)

    if not config.test:
        final_rho_s = np.mean(np.array(rho_s_list))
        final_rho_p = np.mean(np.array(rho_p_list))

        print(' final_srcc new {} \n final_plcc new:{}'.format(final_rho_s, final_rho_p))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='koniq', help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=1, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight', dest='weight', type=float, default=1, help='Weight')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--seed', dest='seed', type=int, default=2021,help='for reproducing the results')
    parser.add_argument('--svpath', dest='svpath', type=str, default='path to save the results',help='the path to save the info')
    parser.add_argument('--datapath', default='..')
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--group_contrastive', action='store_true')
    parser.add_argument('--rank', action='store_true')
    parser.add_argument('--comp', action='store_true')
    parser.add_argument('--nos', action='store_true')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--rotation', action='store_true')
    parser.add_argument('--contrique', action='store_true')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--fix_ssh', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--niter', default=3, type=int)
    parser.add_argument('--run', dest='run', type=int, default=1,help='for running at multiple seeds')

    config = parser.parse_args()
    config.datapath = '/media/user/New Volume/Subhadeep/datasets/' + config.datapath
    main(config)
