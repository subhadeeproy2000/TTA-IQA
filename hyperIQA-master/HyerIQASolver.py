from scipy import stats
import models
from tqdm import tqdm
from SSHead import head_on_layer2,ExtractorHead
import torchvision.transforms as T

import copy

import data_loader
from rotation import  *
from util import *

class HyperIQASolver(object):
    """Solver for training and testing hyperIQA"""
    def __init__(self, config, path, train_idx, test_idx):

        self.test_patch_num = config.test_patch_num
        self.config=config

        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_hyper.train(True)

        self.head = head_on_layer2(config)
        self.ssh = ExtractorHead(self.model_hyper.res, self.head).cuda()

        self.l1_loss = torch.nn.L1Loss().cuda()

        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())

        self.lr = config.lr
        self.optimizer_ssh = torch.optim.Adam(self.ssh.ext.parameters(), lr=self.lr)

        if not config.fix_ssh:
            self.optimizer_ssh = torch.optim.Adam(self.ssh.parameters(), lr=self.lr)

        train_loader = data_loader.DataLoader(config, config.dataset, path,
                                              train_idx, config.patch_size,
                                              config.train_patch_num,
                                              batch_size=config.batch_size, istrain=True)

        test_loader = data_loader.DataLoader(config, config.dataset, path,
                                             test_idx, config.patch_size,
                                             config.test_patch_num, batch_size=config.batch_size, istrain=True)

        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        self.rank_loss = nn.BCELoss()

    def test(self,svPath,seed,pretrained=False):
        """Testing"""
        self.model_hyper.train(False)

        if pretrained:
            self.model_hyper.load_state_dict(torch.load(svPath + '/model_livefb_17'))

        pred_scores = []
        gt_scores = []

        with torch.no_grad():

            steps2 = 0

            for img, label in tqdm(self.test_data):

                img=img['image']

                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())

                paras = self.model_hyper(img)
                model_target = models.TargetNet(paras).cuda()
                model_target.train(False)
                pred = model_target(paras['target_in_vec'])

                try:
                    pred_scores = pred_scores + pred.cpu().tolist()
                except:
                    pred_scores = pred_scores + [pred.cpu()]

                try:
                    gt_scores = gt_scores + label.cpu().tolist()
                except:
                    gt_scores = gt_scores + [label.cpu()]

                steps2 += 1

                if steps2 % 50 == 0:
                    try:
                        pred_scores4 = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
                        gt_scores4 = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
                    except:
                        pred_scores4 = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
                        gt_scores4 = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)

                    test_srcc4, _ = stats.spearmanr(pred_scores4, gt_scores4)
                    test_plcc4, _ = stats.pearsonr(pred_scores4, gt_scores4)

                    print('After {} images test_srcc : {} \n test_plcc:{}'.format(steps2, test_srcc4, test_plcc4))

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_hyper.train(True)
        return test_srcc, test_plcc

    def adapt(self, data_dict, config, old_net):

        inputs = data_dict['image']

        f_low = []
        f_high = []

        old_net.eval()

        with torch.no_grad():

            paras = old_net(data_dict['image'].cuda())
            model_target = models.TargetNet(paras).cuda()
            pred0 = model_target(paras['target_in_vec'])

            if config.rank:

                sigma1 = 40 + np.random.random() * 20
                sigma2 = 5 + np.random.random() * 15

                data_dict['blur_high'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(inputs).cuda()
                data_dict['blur_low'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(inputs).cuda()

                id_dict = {0: data_dict['comp_high'], 1: data_dict['comp_low'], 2: data_dict['nos_high'],
                           3: data_dict['nos_low'], 4: data_dict['blur_high'], 5: data_dict['blur_low']}

                paras = old_net(data_dict['comp_high'].cuda())
                model_target = models.TargetNet(paras).cuda()
                pred1 = model_target(paras['target_in_vec'])


                paras = old_net(data_dict['comp_low'].cuda())
                model_target = models.TargetNet(paras).cuda()
                pred2 = model_target(paras['target_in_vec'])

                paras = old_net(data_dict['nos_high'].cuda())
                model_target = models.TargetNet(paras).cuda()
                pred3 = model_target(paras['target_in_vec'])

                paras = old_net(data_dict['nos_low'].cuda())
                model_target = models.TargetNet(paras).cuda()
                pred4 = model_target(paras['target_in_vec'])

                paras = old_net(data_dict['blur_high'].cuda())
                model_target = models.TargetNet(paras).cuda()
                pred5 = model_target(paras['target_in_vec'])

                paras = old_net(data_dict['blur_low'].cuda())
                model_target = models.TargetNet(paras).cuda()
                pred6 = model_target(paras['target_in_vec'])

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

                try:
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
                except:
                    p=0
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
                    print('h')

                f_low = torch.squeeze(torch.stack(f_low), dim=1)
                f_high = torch.squeeze(torch.stack(f_high), dim=1)

        if config.comp:
            f_low = data_dict['comp_low'].cuda()
            f_high = data_dict['comp_high'].cuda()
        if config.nos:
            f_low = data_dict['nos_low'].cuda()
            f_high = data_dict['nos_high'].cuda()
        if config.blur:

            sigma2 = 40 + np.random.random() * 40  #40-80
            sigma1 = 1 + np.random.random() * 19  #1-20

            f_low = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(inputs).cuda()
            f_high = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(inputs).cuda()

        if config.contrastive:
            f_low = data_dict['image1'].cuda()
            f_high = data_dict['image2'].cuda()

        m = nn.Sigmoid()

        for param in self.ssh.parameters():
            param.requires_grad = False

        for layer in self.ssh.ext.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.requires_grad_(True)

        if config.fix_ssh:
            self.ssh.eval()
            self.ssh.ext.train()

        loss_hist = []

        for iteration in range(config.niter):

            target = torch.ones(inputs.shape[0]).cuda()

            if config.contrastive:
                f_neg_feat = self.ssh(f_low)
                f_pos_feat = self.ssh(f_high)

                loss_fn = ContrastiveLoss(f_pos_feat.shape[0], 1).cuda()

                loss = loss_fn(f_neg_feat, f_pos_feat)

            # self.ssh.ext.eval()

            if config.group_contrastive:

                idx = np.argsort(pred0.cpu(), axis=0)

                f_feat = self.ssh(inputs.cuda())

                f_pos_feat = []
                f_neg_feat = []

                for n in range(config.batch_size // 4):
                    f_pos_feat.append(f_feat[idx[n]])
                    f_neg_feat.append(f_feat[idx[-n - 1]])

                f_pos_feat = torch.squeeze(torch.stack(f_pos_feat), dim=1)
                f_neg_feat = torch.squeeze(torch.stack(f_neg_feat), dim=1)


                loss_fn = GroupContrastiveLoss(f_pos_feat.shape[0], 1).cuda()

                loss = loss_fn(f_neg_feat, f_pos_feat)

                # print(self.ssh.training)

            # self.ssh.ext.train()

            # self.ssh.ext.eval()


            if config.rank or config.blur or config.comp or config.nos:
                f_neg_feat = self.ssh(f_low)
                f_pos_feat = self.ssh(f_high)
                f_actual = self.ssh(inputs.cuda())

                dist_high = torch.nn.PairwiseDistance(p=2)(f_pos_feat, f_actual)
                dist_low = torch.nn.PairwiseDistance(p=2)(f_neg_feat, f_actual)

                if config.group_contrastive:
           	        loss = self.rank_loss(m(dist_high - dist_low), target) * self.config.weight+loss
                else:
                    loss=self.rank_loss(m(dist_high-dist_low),target)

            # self.ssh.ext.train()

            if config.rotation:
                inputs_ssh, labels_ssh = rotate_batch(inputs.cuda(), 'rand')
                outputs_ssh = self.ssh(inputs_ssh.float())
                loss = nn.CrossEntropyLoss()(outputs_ssh, labels_ssh.cuda())

            loss.backward()
            self.optimizer_ssh.step()
            loss_hist.append(loss.detach().cpu())

        # print(loss_hist)
        return loss_hist

    def new_ttt(self, svPath, config):

        if config.online:
            self.model_hyper.load_state_dict(torch.load(svPath + '/model_livefb_17'))

        steps = 0

        pred_scores_old = []
        pred_scores = []

        gt_scores = []

        mse_all = []
        mse_all_old = []

        for data_dict, label in tqdm(self.test_data, leave=False):

            img = data_dict['image']

            if not config.online:
                self.model_hyper.load_state_dict(torch.load(svPath + '/model_livefb_17'))

            label = torch.as_tensor(label.cuda()).requires_grad_(False)

            old_net = copy.deepcopy(self.model_hyper)
            old_net.load_state_dict(torch.load(svPath + '/model_livefb_17'))


            if config.group_contrastive:
                if len(img) > 3:
                    loss_hist = self.adapt(data_dict, config, old_net)
                elif config.rank or config.blur or config.comp or config.nos or config.contrastive or config.rotation:
                    config.group_contrastive = False
                    loss_hist = self.adapt(data_dict, config, old_net)
            elif config.rank or config.blur or config.comp or config.nos or config.contrastive or config.rotation:
                loss_hist = self.adapt(data_dict, config, old_net)

            old_net.load_state_dict(torch.load(svPath + '/model_livefb_17'))

            old_net.eval()
            self.model_hyper.eval()

            mse, pred = self.test_single_iqa(self.model_hyper.cuda(), img.cuda(), label.cuda())
            mse_old, pred_old = self.test_single_iqa(old_net.cuda(), img.cuda(), label.cuda())

            self.model_hyper.train()

            try:
                pred_scores = pred_scores + pred.cpu().tolist()
            except:
                pred_scores = pred_scores + [pred.cpu()]
            try:
                pred_scores_old = pred_scores_old + pred_old.cpu().tolist()
            except:
                pred_scores_old = pred_scores_old + [pred_old.cpu()]
            try:
                gt_scores = gt_scores + label.cpu().tolist()
            except:
                gt_scores = gt_scores + [label.cpu()]

            mse_all.append(mse.cpu())
            mse_all_old.append(mse_old.cpu())

            steps += 1

            if steps % 20 == 0:

                pred_scores1 = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
                pred_scores_old1 = np.mean(np.reshape(np.array(pred_scores_old), (-1, self.test_patch_num)), axis=1)
                gt_scores1 = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

                test_srcc_old, _ = stats.spearmanr(pred_scores_old1, gt_scores1)
                test_plcc_old, _ = stats.pearsonr(pred_scores_old1, gt_scores1)

                test_srcc, _ = stats.spearmanr(pred_scores1, gt_scores1)
                test_plcc, _ = stats.pearsonr(pred_scores1, gt_scores1)

                print('After {} images test_srcc old : {}  new {} \n test_plcc old:{} new:{}'.format(steps, test_srcc_old,test_srcc,test_plcc_old,test_plcc))

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

            paras = model(image)
            model_target = models.TargetNet(paras).cuda()
            pred = model_target(paras['target_in_vec'])
            mse_loss = self.l1_loss(label, pred)

        model.train()

        return mse_loss, pred