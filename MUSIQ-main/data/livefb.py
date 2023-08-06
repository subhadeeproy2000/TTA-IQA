import os
import torch
import numpy as np
import cv2
from skimage.util import random_noise
from PIL import Image
import csv


def noisy(path,transform,root):
    sigma1 = 0.1 + np.random.random() * 0.0001
    sigma2 = 0.0005 + np.random.random() * 0.0001

    ab = cv2.imread(path)

    # gauss = np.random.normal(0, 1, ab.size).reshape(ab.shape).astype('uint8')
    # noise = ab + ab * gauss
    noise=random_noise(ab, mode='gaussian',var=sigma1)
    image1 = Image.fromarray((noise * 255).astype('uint8'))
    # cv2.imwrite(root + "/noisy1." +path.split('.')[-1],noise)

    # image1 = Image.fromarray(noise.astype('uint8'), 'RGB')

    # image1 = Image.open(root + '/noisy1.'+path.split('.')[-1])
    image1 = transform(image1)

    # gauss = np.random.normal(0, 0.2, ab.size).reshape(ab.shape).astype('uint8')
    # noise = ab + ab * gauss
    noise=random_noise(ab, mode='gaussian',var=sigma2)
    image2 = Image.fromarray((noise * 255).astype('uint8'))

    # image2 = Image.fromarray(noise.astype('uint8'), 'RGB')
    image2 = transform(image2)

    return image1,image2

def compress(sample,transform,root):
    image1={}
    image2={}
    for key,image in sample.items():
        image = Image.fromarray((image*255).astype(np.uint8)).convert('RGB')

        sigma1 = 60 + np.random.random()*3
        sigma2 = 90+ np.random.random()*3

        image.save(root+"/Compressed_" + '1.jpg', optimize=True, quality=int(sigma1))
        image1[key]= cv2.imread(root+'/Compressed_1.jpg')
        if image1[key] is None:
            print( 'k' )

        image.save(root + "/Compressed_" + '2.jpg', optimize=True, quality=int(sigma2))
        image2[key] = cv2.imread(root+'/Compressed_2.jpg')
        if image1[key] is None:
            print( 'k' )

    image1 = transform(image1)
    image2 = transform(image2)

    return image1,image2

class IQADataset(torch.utils.data.Dataset):
    def __init__(self, config,db_path, txt_file_name, scale_1, scale_2, transform, train_mode, scene_list, train_size=0.8):
        super(IQADataset, self).__init__()

        self.db_path = db_path
        self.config=config
        self.txt_file_name = txt_file_name
        self.scale_1 = scale_1
        self.scale_2 = scale_2
        self.transform = transform
        self.train_mode = train_mode
        self.scene_list = scene_list
        self.train_size = train_size

        self.data_dict = IQADatalist(
            txt_file_name = self.txt_file_name,
            train_mode = self.train_mode,
            scene_list = self.scene_list,
            train_size = self.train_size
        ).load_data_dict()

        self.n_images = len(self.data_dict['d_img_list'])
    
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        # d_img_org: H x W x C
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img_org = cv2.imread(os.path.join((self.db_path), d_img_name), cv2.IMREAD_COLOR)
        d_img_org = cv2.cvtColor(d_img_org, cv2.COLOR_BGR2RGB)
        d_img_org= cv2.resize(d_img_org, dsize=(224,224),interpolation=cv2.INTER_CUBIC)
        d_img_org = np.array(d_img_org).astype('float32') / 255

        h, w, c = d_img_org.shape
        d_img_scale_1 = cv2.resize(d_img_org, dsize=(self.scale_1, int(h * (self.scale_1 / w))),
                                   interpolation=cv2.INTER_CUBIC)
        d_img_scale_2 = cv2.resize(d_img_org, dsize=(self.scale_2, int(h * (self.scale_2 / w))),
                                   interpolation=cv2.INTER_CUBIC)
        d_img_scale_2 = d_img_scale_2[:160, :, :]

        score = self.data_dict['score_list'][idx]

        sample = {'d_img_org': d_img_org, 'd_img_scale_1': d_img_scale_1, 'd_img_scale_2': d_img_scale_2,
                  'score': score}

        # image1, image2 = compress(sample, self.transform, self.db_path)
        # image1,image2=noisy(path,self.transform,self.root)
        # image1, image2 = self.transform(sample), self.transform(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

class IQADatalist():
        def __init__(self, txt_file_name, train_mode, scene_list, train_size=0.8):
            self.txt_file_name = txt_file_name
            self.train_mode = train_mode
            self.train_size = train_size
            self.scene_list = scene_list
            self.db_path='/home/user/Subhadeep/TRes/Dataset/LIVE_FB'

        def load_data_dict(self):
            scn_idx_list, d_img_list, score_list = [], [], []

            csv_file = os.path.join(self.db_path, 'labels_image.csv')
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dis=row['name'].split('/')[1]
                    score = float(row['mos'])
                    d_img_list.append(dis)
                    score_list.append(score)

            # reshape score_list (1xn -> nx1)
            score_list = np.array(score_list)
            score_list = score_list.astype('float').reshape(-1, 1)

            data_dict = {'d_img_list': d_img_list, 'score_list': score_list}

            return data_dict
