from __future__ import print_function
import argparse
import random
import torchvision
import torchvision.transforms as T
from scipy import stats
from tqdm import tqdm
import copy
import torch.utils.data as data
import os.path
from utils import *
from rotation import *
from models import Net


class DataLoader(object):
    """
    Dataset class for IQA databases
    """

    def __init__(self,config, path, img_indx, patch_size, patch_num, batch_size=1):

        self.batch_size = batch_size
        self.config=config

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(size=patch_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        self.data = Folder(self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)


    def get_data(self):
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        return dataloader

class Folder(data.Dataset):
    def __init__(self, config, root, index, transform, patch_num):

        csv_path = os.path.join(root, 'MOS.csv')
        df = pd.read_csv(csv_path)

        df['0'] = df['0'].apply(lambda x: root + '/' + x)
        dataset = df['0'].tolist()
        labels = df['1'].tolist()

        # df['image_name'] = df['image_name'].apply(lambda x: root + '/' + x)
        # dataset = df['image_name'].tolist()
        # labels = df['MOS'].tolist()
        sample = []
        self.root = root
        self.config = config
        for item, i in enumerate(index):
            for aug in range(patch_num):
                sample.append((dataset[i], labels[i]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        data_dict = {}

        if self.transform is not None:
            data_dict['image'] = self.transform(sample)

        path, target = self.samples[index]
        sample = pil_loader(path)
        data_dict = {}

        if self.transform is not None:
            data_dict['image'] = self.transform(sample)

        data_dict = processing(data_dict, sample, self.transform, self.root, path, self.config)

        return data_dict, target

    def __len__(self):
        length = len(self.samples)
        return length

class Model(object):

    def __init__(self, config, device, Net):
        super(Model, self).__init__()

        if not config.tta_tf:   #if we want to adapt transformer also
            from SSHead import head_on_layer2, ExtractorHead, extractor_from_layer2
        else:
            from SSHead_tf import head_on_layer2, ExtractorHead, extractor_from_layer2

        self.device = device
        self.test_patch_num = config.test_patch_num
        self.l1_loss = torch.nn.L1Loss()
        self.lr = config.lr
        self.rank_loss = nn.BCELoss()

        self.net = Net(config, device).to(device)
        self.head = head_on_layer2(config)
        self.ext = extractor_from_layer2(self.net)
        self.ssh = ExtractorHead(self.ext, self.head).cuda()

        self.config = config
        self.clsloss = nn.CrossEntropyLoss()

        self.optimizer_ssh = torch.optim.Adam(self.ext.parameters(), lr=self.lr)
        if not config.fix_ssh:
            self.optimizer_ssh = torch.optim.Adam(self.ssh.parameters(), lr=self.lr)


    def test(self, data, pretrained=0):
        if pretrained:
            self.net.load_state_dict(torch.load(self.config.svpath))

        self.net.eval()

        pred_scores = []
        gt_scores = []

        srcc=np.zeros(len(data))
        plcc=np.zeros(len(data))


        with torch.no_grad():
            steps2 = 0

            for data_dict, label in tqdm(data, leave=False):

                img = data_dict['image']
                img = torch.as_tensor(img.to(self.device))
                label = torch.as_tensor(label.to(self.device))
                pred, _ = self.net(img)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                steps2 += 1

                try:
                    pred_scores4 = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
                    gt_scores4 = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
                except:
                    pred_scores4 = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
                    gt_scores4 = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)

                test_srcc4, _ = stats.spearmanr(pred_scores4, gt_scores4)
                test_plcc4, _ = stats.pearsonr(pred_scores4, gt_scores4)

                srcc[steps2 - 1]=test_srcc4
                plcc[steps2 - 1]=test_plcc4


                if steps2%50==0:

                    print('After {} images test_srcc : {} \n test_plcc:{}'.format(steps2, test_srcc4, test_plcc4))

        try:
            pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
            gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        except:
            pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
            gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        return test_srcc, test_plcc,srcc,plcc

    def adapt(self, data_dict, config, old_net):

        inputs = data_dict['image']

        f_low = []
        f_high = []

        with torch.no_grad():
            pred0, _ = old_net(data_dict['image'].cuda())

            if config.rank:

                sigma1 = 40 + np.random.random() * 20
                sigma2 = 5 + np.random.random() * 15

                data_dict['blur_high'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(inputs).cuda()
                data_dict['blur_low'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(inputs).cuda()

                id_dict = {0: data_dict['comp_high'], 1: data_dict['comp_low'], 2: data_dict['nos_high'],
                           3: data_dict['nos_low'], 4: data_dict['blur_high'], 5: data_dict['blur_low']}

                pred1, _ = old_net(data_dict['comp_high'].cuda())
                pred2, _ = old_net(data_dict['comp_low'].cuda())

                pred3, _ = old_net(data_dict['nos_high'].cuda())
                pred4, _ = old_net(data_dict['nos_low'].cuda())

                pred5, _ = old_net(data_dict['blur_high'].cuda())
                pred6, _ = old_net(data_dict['blur_low'].cuda())

                try:
                    comp = torch.unsqueeze(torch.abs(pred2 - pred1), dim=1)
                except:
                    comp = (torch.ones(1, 1) * (torch.abs(pred2 - pred1)).item()).cuda()

                try:
                    nos = torch.unsqueeze(torch.abs(pred4 - pred3), dim=1)
                except:
                    nos = (torch.ones(1, 1) * (torch.abs(pred4 - pred3)).item()).cuda()

                try:
                    blur = torch.unsqueeze(torch.abs(pred6 - pred5), dim=1)
                except:
                    blur = (torch.ones(1, 1) * (torch.abs(pred6 - pred5)).item()).cuda()


                all_diff = torch.cat([comp, nos, blur], dim=1)

                for p in range(len(pred0)):
                    if all_diff[p].argmax().item() == 0:
                        f_low.append(id_dict[1][p].cuda())
                        f_high.append(id_dict[0][p].cuda())
                        # print('comp', end=" ")
                    if all_diff[p].argmax().item() == 1:
                        f_low.append(id_dict[3][p].cuda())
                        f_high.append(id_dict[2][p].cuda())
                        # print('nos', end=" ")
                    if all_diff[p].argmax().item() == 2:
                        f_low.append(id_dict[5][p].cuda())
                        f_high.append(id_dict[4][p].cuda())
                        # print('blur', end=" ")

                f_low = torch.squeeze(torch.stack(f_low), dim=1)
                f_high = torch.squeeze(torch.stack(f_high), dim=1)

        if config.comp:
            f_low = data_dict['comp_low'].cuda()
            f_high = data_dict['comp_high'].cuda()
        if config.nos:
            f_low = data_dict['nos_low'].cuda()
            f_high = data_dict['nos_high'].cuda()
        if config.blur:

            sigma2 = 40 + np.random.random() * 20
            sigma1 = 5 + np.random.random() * 15

            f_low = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(inputs).cuda()
            f_high = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(inputs).cuda()


        if config.contrastive or  config.contrique:
            f_low = data_dict['image1'].cuda()
            f_high = data_dict['image2'].cuda()

        m = nn.Sigmoid()

        for param in self.ssh.parameters():
            param.requires_grad = False

        for layer in self.ssh.ext.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.requires_grad_(True)
            if config.tta_tf:
                if not config.bn:
                    if isinstance(layer, nn.BatchNorm2d):
                        layer.requires_grad_(False)
            if config.ln:
                if isinstance(layer, nn.LayerNorm):
                    layer.requires_grad_(True)

        if config.fix_ssh:
            self.ssh.eval()
            self.ssh.ext.train()
        else:
            self.ssh.train()

        loss_hist = []

        for iteration in range(config.niter):

            target = torch.ones(inputs.shape[0]).cuda()

            if config.rank or config.blur or config.comp or config.nos:
                f_neg_feat = self.ssh(f_low)
                f_pos_feat = self.ssh(f_high)
                f_actual = self.ssh(inputs.cuda())

                dist_high = torch.nn.PairwiseDistance(p=2)(f_pos_feat, f_actual)
                dist_low = torch.nn.PairwiseDistance(p=2)(f_neg_feat, f_actual)

                loss = self.rank_loss(m(dist_high - dist_low), target)

            if config.contrastive or config.contrique:
                f_neg_feat = self.ssh(f_low)
                f_pos_feat = self.ssh(f_high)

                loss_fn = ContrastiveLoss(f_pos_feat.shape[0], 0.1).cuda()

                loss = loss_fn(f_neg_feat, f_pos_feat)

            if config.group_contrastive:

                idx = np.argsort(pred0.cpu(), axis=0)

                f_feat = self.ssh(inputs.cuda())

                f_pos_feat = []
                f_neg_feat = []

                for n in range(max(2,int(config.batch_size*config.p))):
                    try:
                        f_pos_feat.append(f_feat[idx[n]])
                        f_neg_feat.append(f_feat[idx[-n - 1]])
                    except:
                        continue

                f_pos_feat = torch.squeeze(torch.stack(f_pos_feat), dim=1)
                f_neg_feat = torch.squeeze(torch.stack(f_neg_feat), dim=1)

                loss_fn = GroupContrastiveLoss(f_pos_feat.shape[0], 0.1).cuda()

                if config.rank or config.blur or config.comp or config.nos:
                    loss += loss_fn(f_neg_feat, f_pos_feat) * config.weight
                else:
                    loss = loss_fn(f_neg_feat, f_pos_feat)

            if config.rotation:
                inputs_ssh, labels1_ssh = rotate_batch(inputs.cuda(), 'rand')
                outputs_ssh = self.ssh(inputs_ssh.float())
                loss = nn.CrossEntropyLoss()(outputs_ssh, labels1_ssh.cuda())

            loss.backward()
            self.optimizer_ssh.step()
            loss_hist.append(loss.detach().cpu())

        # print(loss_hist)

        return loss_hist

    def new_ttt(self, data, config):

        if config.online:
            self.net.load_state_dict(torch.load(self.config.svpath ))

        old_net = copy.deepcopy(self.net)
        old_net.load_state_dict(torch.load(self.config.svpath))

        steps = 0

        pred_scores = []
        pred_scores_old = []

        gt_scores = []
        mse_all = []
        mse_all_old = []


        for data_dict, label in tqdm(data, leave=False):
            img = data_dict['image']

            if not config.online:
                self.net.load_state_dict(torch.load(self.config.svpath ))

            label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

            old_net.load_state_dict(torch.load(self.config.svpath ))

            if config.group_contrastive:
                if len(img) > 3:
                    loss_hist = self.adapt(data_dict, config, old_net)
                else:
                    if config.rank or config.blur or config.comp or config.nos or config.contrastive or config.rotation or config.contrique:
                        config.group_contrastive = False
                        loss_hist = self.adapt(data_dict, config, old_net)
            elif config.rank or config.blur or config.comp or config.nos or config.contrastive or config.rotation or config.contrique:
                loss_hist = self.adapt(data_dict, config, old_net)

            # if config.rank:
            #     print('done')

            mse, pred = self.test_single_iqa(self.net, img, label)

            old_net.load_state_dict(torch.load(self.config.svpath ))

            mse_old, pred_old = self.test_single_iqa(old_net, img, label)

            pred_scores = pred_scores + pred.cpu().tolist()
            pred_scores_old = pred_scores_old + pred_old.cpu().tolist()
            gt_scores = gt_scores + label.cpu().tolist()

            mse_all.append(mse.cpu())
            mse_all_old.append(mse_old.cpu())

            steps += 1

            if steps % 10 == 0:
                pred_scores1 = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
                pred_scores_old1 = np.mean(np.reshape(np.array(pred_scores_old), (-1, self.test_patch_num)), axis=1)
                gt_scores1 = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

                test_srcc_old, _ = stats.spearmanr(pred_scores_old1, gt_scores1)
                test_plcc_old, _ = stats.pearsonr(pred_scores_old1, gt_scores1)

                test_srcc, _ = stats.spearmanr(pred_scores1, gt_scores1)
                test_plcc, _ = stats.pearsonr(pred_scores1, gt_scores1)
                print(
                    'After {} images test_srcc old : {}  new {} \n test_plcc old:{} new:{}'.format(steps, test_srcc_old,
                                                                                                   test_srcc,
                                                                                                   test_plcc_old,
                                                                                                   test_plcc))

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        pred_scores_old = np.mean(np.reshape(np.array(pred_scores_old), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

        test_srcc_old, _ = stats.spearmanr(pred_scores_old, gt_scores)
        test_plcc_old, _ = stats.pearsonr(pred_scores_old, gt_scores)

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        return test_srcc_old, test_plcc_old, test_srcc, test_plcc

    def test_single_iqa(self, model, image, label):
        model.eval()
        with torch.no_grad():
            pred, _ = model(image.cuda())
            mse_loss = self.l1_loss(label, pred)
        return mse_loss, pred

parser = argparse.ArgumentParser()
parser.add_argument('--img_num',type=int, default='1000')
parser.add_argument('--datapath', default='..')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--ng', default=2, type=int)
parser.add_argument('--fix_ssh', action='store_true')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=3, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--weight', type=float, default=1,help='Weight for rank plus GC')
parser.add_argument('--p', type=float, default=0.25,help='p for GC loss')
parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1,
                    help='Number of sample patches from testing image')
parser.add_argument('--seed', dest='seed', type=int, default=2021,
                        help='for reproducing the results')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                    help='Crop size for training & testing image patches')
parser.add_argument('--svpath', dest='svpath', type=str,default='/home/user/Subhadeep/ttt_cifar_IQA/Save_RES/',
                        help='the path to save the info')
parser.add_argument('--gpunum', dest='gpunum', type=str, default='0',
                    help='the id for the gpu that will be used')
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
args.datapath='/media/user/New Volume/Subhadeep/datasets/'+args.datapath

if torch.cuda.is_available():
    if len(args.gpunum) == 1:
        device = torch.device("cuda", index=int(args.gpunum))
else:
    device = torch.device("cpu")

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

#----------------------------------------
folder_path = args.datapath
sel_num = list(range(0, args.img_num))


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

    # Randomly select 80% images for training and the rest for testing
    random.shuffle(sel_num)
    test_index = sel_num

    test_loader = DataLoader(args, folder_path,
                                         test_index, args.patch_size,
                                         args.test_patch_num,batch_size=args.batch_size)
    test_data = test_loader.get_data()

    solver = Model(args, device,Net)

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
