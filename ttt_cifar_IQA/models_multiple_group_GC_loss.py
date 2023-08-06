import torchvision.transforms as T
from rotation import *
from scipy import stats
from tqdm import tqdm
import copy
from utils import *

class TReS(object):

    def __init__(self, config, device, Net):
        super(TReS, self).__init__()

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
            self.net.load_state_dict(torch.load(self.config.svpath + '/{}_TReS'.format(str(self.config.train_data))))

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
                f_mid_feat = []

                if config.ng == 4:
                    if len(f_feat) == config.batch_size:
                        f_mid1_feat = []
                        f_mid2_feat = []
                        for n in range(config.batch_size // 4):
                            f_pos_feat.append(f_feat[idx[n]])
                            f_neg_feat.append(f_feat[idx[-n - 1]])
                            f_mid1_feat.append(f_feat[idx[n + 2]])
                            f_mid2_feat.append(f_feat[idx[n + 4]])

                elif config.ng == 3:
                    if len(f_feat) >= 3 * config.batch_size // 4:
                        for n in range(config.batch_size // 4):
                            f_pos_feat.append(f_feat[idx[n]])
                            f_neg_feat.append(f_feat[idx[-n - 1]])
                            if len(f_feat) > 3 * config.batch_size // 4:
                                f_mid_feat.append(f_feat[idx[n + int(config.batch_size * 0.375)]])
                            else:
                                f_mid_feat.append(f_feat[idx[n + int(len(f_feat) * 0.375)]])

                    elif len(f_feat) >= 4:
                        print('batch size is 4 or 5')
                        for n in range(config.batch_size // 4):
                            f_pos_feat.append(f_feat[idx[n]])
                            f_neg_feat.append(f_feat[idx[-n - 1]])

                else:
                    for n in range(config.batch_size // 4):
                        f_pos_feat.append(f_feat[idx[n]])
                        f_neg_feat.append(f_feat[idx[-n - 1]])

                f_pos_feat = torch.squeeze(torch.stack(f_pos_feat), dim=1)
                f_neg_feat = torch.squeeze(torch.stack(f_neg_feat), dim=1)
                if len(f_feat) >= 3 * config.batch_size // 4 and config.ng == 3:
                    f_mid_feat = torch.squeeze(torch.stack(f_mid_feat), dim=1)
                if config.ng == 4 and len(f_feat) == config.batch_size:
                    f_mid1_feat = torch.squeeze(torch.stack(f_mid1_feat), dim=1)
                    f_mid2_feat = torch.squeeze(torch.stack(f_mid2_feat), dim=1)

                loss_fn = GroupContrastiveLoss(f_pos_feat.shape[0], 1).cuda()
                loss_fn1 = GroupContrastiveLoss(f_pos_feat.shape[0], 1).cuda()
                loss_fn2 = GroupContrastiveLoss(f_pos_feat.shape[0], 1).cuda()


                if len(f_feat) >= 3 * config.batch_size // 4 and config.ng == 3:
                    loss = loss_fn(f_neg_feat, f_pos_feat) + loss_fn1(f_neg_feat, f_mid_feat) + loss_fn2(f_pos_feat,f_mid_feat)
                elif config.ng == 4 and len(f_feat) == config.batch_size:
                    loss = loss_fn(f_neg_feat, f_pos_feat) + loss_fn1(f_neg_feat, f_mid1_feat) + loss_fn2(f_pos_feat,f_mid2_feat)
                else:
                    loss = loss_fn(f_neg_feat, f_pos_feat)

            if config.rotation:
                inputs_ssh, labels1_ssh = rotate_batch(inputs.cuda(), 'rand')
                outputs_ssh = self.ssh(inputs_ssh.float())
                loss = nn.CrossEntropyLoss()(outputs_ssh, labels1_ssh.cuda())

            loss.backward()
            self.optimizer_ssh.step()
            loss_hist.append(loss.detach().cpu())

        return loss_hist

    def new_ttt(self, data, config):

        if config.online:
            self.net.load_state_dict(torch.load(self.config.svpath + '/{}_TReS'.format(str(self.config.train_data))))

        old_net = copy.deepcopy(self.net)
        old_net.load_state_dict(torch.load(self.config.svpath + '/{}_TReS'.format(str(self.config.train_data))))

        steps = 0

        pred_scores = []
        pred_scores_old = []

        gt_scores = []
        mse_all = []
        mse_all_old = []


        for data_dict, label in tqdm(data, leave=False):

            img = data_dict['image']

            if not config.online:
                self.net.load_state_dict(torch.load(self.config.svpath + '/{}_TReS'.format(str(self.config.train_data))))

            label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

            old_net.load_state_dict(torch.load(self.config.svpath + '/{}_TReS'.format(str(self.config.train_data))))

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

            old_net.load_state_dict(torch.load(self.config.svpath + '/{}_TReS'.format(str(self.config.train_data))))

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


if __name__ == '__main__':
    import os
    import argparse
    import random
    import numpy as np
    from args import *

