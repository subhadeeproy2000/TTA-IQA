import pandas as pd
import torch.utils.data as data
import os.path
import scipy.io
import csv
import cv2
from pandas_ods_reader import read_ods
from skimage.util import random_noise
import scipy.ndimage
from util import *


def processing(data_dict,sample,transform,root,path,config):
    if config.rank:
        data_dict['comp_high'], data_dict['comp_low'] = compress(sample,transform, root)
        data_dict['nos_low'], data_dict['nos_high'] = noisy(path,transform)

    if config.comp:
        data_dict['comp_high'], data_dict['comp_low'] = compress(sample, transform, root)

    if config.nos:
        data_dict['nos_low'], data_dict['nos_high'] = noisy(path,transform)

    if config.contrastive:
        data_dict['image1'], data_dict['image2'] = transform(sample), transform(sample)

    # if config.online:
    #     data_dict['online'] = []
    #     for i in range(32):
    #         data_dict['online'].append(transform(sample))

    data_dict['img_name'] = path

    if config.contrique:

        image_orig = Image.open(path)

        image_size = image_orig.size

        if image_orig.mode == 'L':
            image_orig = np.array(image_orig)
            image_orig = np.repeat(image_orig[:, :, None], 3, axis=2)
            image_orig = Image.fromarray(image_orig)
        elif image_orig.mode != 'RGB':
            image_orig = image_orig.convert('RGB')

        # Data augmentations

        # scaling transform and random crop
        div_factor = np.random.choice([1, 2], 1)[0]
        image_2 = ResizeCrop(image_orig, image_size, div_factor)

        # change colorspace
        colorspace_choice = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        image_2 = colorspaces(image_2, colorspace_choice)
        try:
            image_2 = transform(image_2)
        except:
            image_2 = ResizeCrop(image_orig, image_size, div_factor)
            # change colorspace
            image_2 = colorspaces(image_2, 2)
            image_2 = transform(image_2)

        # scaling transform and random crop
        image = ResizeCrop(image_orig, image_size, 3 - div_factor)

        # change colorspace
        colorspace_choice = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        image = colorspaces(image, colorspace_choice)
        try:
            image = transform(image)
        except:
            image = ResizeCrop(image_orig, image_size, 3 - div_factor)
            image = colorspaces(image, 2)
            image = transform(image)

        data_dict['image1'], data_dict['image2'] = image, image_2

    return data_dict


class LIVEFolder(data.Dataset):

    def __init__(self, config,root, index, transform, patch_num):

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'].astype(np.float32)

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']


        refname.sort()
        sample = []
        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                                sample.append((imgpath[item], labels[0][item]))
        self.samples = sample
        self.transform = transform
        self.root=root
        self.config=config
        self.image_size = (224,224)

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

        if self.config.rank or self.config.blur or self.config.comp or self.config.nos or self.config.contrastive or self.config.rotation or self.config.contrique:
            data_dict=processing(data_dict,sample,self.transform,self.root,path,self.config)

        return data_dict, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename

class NNIDFolder(data.Dataset):

    def __init__(self, config,root, index, transform, patch_num):

        txt_file = os.path.join(root, 'mos512_with_names.txt')
        sample = []
        self.root=root
        self.config=config

        with open(txt_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split("\t")
                for aug in range(patch_num):
                    sample.append((root+'/sub512/'+line_split[1],float(line_split[0])))

        txt_file = os.path.join(root, 'mos1024_with_names.txt')

        with open(txt_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split("\t")
                for aug in range(patch_num):
                    sample.append((root + '/sub1024/' + line_split[1], float(line_split[0])))

        txt_file = os.path.join(root, 'mos2048_with_names.txt')

        with open(txt_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split("\t")
                for aug in range(patch_num):
                    sample.append((root + '/Sub2048/' + line_split[1], float(line_split[0])))

        self.transform = transform
        self.samples=sample

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

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename

class SPAQFolder(data.Dataset):

    def __init__(self,config, root, index, transform, patch_num):

        df = pd.read_csv(root + '/Annotations/MOS and Image attribute scores.csv')
        df['Image name'] = df['Image name'].apply(lambda x: root +'/TestImage/'+ x)
        dataset=df['Image name'].tolist()
        labels=df['MOS'].tolist()
        sample = []
        self.root=root
        self.config=config
        self.image_size = (224,224)

        for i in range(len(df)):
            for aug in range(patch_num):
                if os.path.isfile(dataset[i]):
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

class PIPALFolder(data.Dataset):

    def __init__(self,config, root, index, transform, patch_num):

        csv_path = os.path.join(root, 'mos_data.csv')
        df=pd.read_csv(csv_path)
        df['id'] = df['filename'].apply(lambda x: int(x.split('/')[0][-1]))
        df=df[df['id']<5]  #decide how manhy folder you want to test
        df['filename'] = df['filename'].apply(lambda x: root +'/'+ x)
        dataset=df['filename'].tolist()
        labels=df['mos'].tolist()
        sample = []
        self.root=root
        self.config=config
        self.image_size = (224,224)
        # for item, i in enumerate(index):
        for i in range(len(df)):
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

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename

class CIDIQFolder(data.Dataset):

    def __init__(self,config, root, index, transform, patch_num):

        df = read_ods(root + '/cid.ods')
        dic = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
        df['Source_ID'] = df['Source_ID'].apply(lambda x: root+'/IS'+str(dic[x.split('_')[1]])+'/co'+x.split('_')[2][-1]+'/'+x+'.jpg')
        dataset = df['Source_ID'].tolist()
        labels = df['Image set specific  MOS'].tolist()
        sample = []

        for i in range(len(df)):
            for aug in range(patch_num):
                   sample.append((dataset[i], labels[i]))

        self.samples = sample
        self.transform = transform
        self.root=root
        self.config=config
        self.image_size = (224,224)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        data_dict={}

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

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class LIVEChallengeFolder(data.Dataset):

    def __init__(self,config, root, index, transform, patch_num):

        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]

        self.root=root
        self.config=config

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'Images', imgpath[item][0][0]), labels[item]))

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


class CSIQFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath,'.png')
        txtpath = os.path.join(root, 'csiq_label.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[0]))
            target.append(words[1])
            ref_temp = words[0].split(".")
            refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []
        refname.sort(reverse=True)
        # refnames_all.sort()

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'dst_imgs_all', imgnames[item]), labels[item]))
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
        sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class Koniq_10kFolder(data.Dataset):

    def __init__(self,config, root, index, transform, patch_num):
        self.root=root
        self.image_size = (224,224)
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                    sample.append((os.path.join(root, '1024x768', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform
        self.config=config

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

        data_dict = processing(data_dict, sample, self.transform, self.root, path, self.config)

        return data_dict, target

    def __len__(self):
        length = len(self.samples)
        return length

class DSLRFolder(data.Dataset):
    def __init__(self, config, root, index, transform, patch_num):

        csv_path = os.path.join(root, 'MOS.csv')
        df = pd.read_csv(csv_path)
        df['0'] = df['0'].apply(lambda x: root + '/' + x)
        dataset = df['0'].tolist()
        labels = df['1'].tolist()
        sample = []
        self.root = root
        self.config = config
        self.image_size = (224, 224)
        for item, i in enumerate(index):
        # for i in range(len(df)):
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


def noisy(path,transform):

    sigma1 = 0.00005+ np.random.random() * 0.000001
    sigma2 = 0.00001+ np.random.random() * 0.000001   #low noise

    ab = cv2.imread(path)
    ab = cv2.cvtColor(ab, cv2.COLOR_BGR2RGB)

    noise=random_noise(ab, mode='gaussian',var=sigma1)
    image1 = Image.fromarray((noise * 255).astype('uint8'))
    image1 = transform(image1)

    noise=random_noise(ab, mode='gaussian',var=sigma2)
    image2 = Image.fromarray((noise * 255).astype('uint8'))
    image2 = transform(image2)

    return image2,image1


def compress(image,transform,root):

    sigma1 = 40 + np.random.random() * 20  # 40-60
    sigma2 = 80 + np.random.random() * 10  # 80-90

    try:
        image.save(root+"/Compressed_" + '1.jpg', optimize=True, quality=int(sigma1))
        image1 = Image.open(root+'/Compressed_1.jpg')
    except:
        image.save(root + "/Compressed_" + '1.bmp', optimize=True, quality=int(sigma1))
        image1 = Image.open(root + '/Compressed_1.bmp')
    image1 = transform(image1)

    try:
        image.save(root + "/Compressed_" + '2.jpg', optimize=True, quality=int(sigma2))
        image2 = Image.open(root+'/Compressed_2.jpg')
    except:
        image.save(root + "/Compressed_" + '2.bmp', optimize=True, quality=int(sigma2))
        image2 = Image.open(root + '/Compressed_2.bmp')
    image2 = transform(image2)

    return image1,image2


class FBLIVEFolder(data.Dataset):

    def __init__(self,config, root, index, transform, patch_num):

        self.root=root
        self.config=config
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'labels_image.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['name'].split('/')[1])
                mos = np.array(float(row['mos'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        data_dict={}
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
     
        

class TID2013Folder(data.Dataset):

    def __init__(self, config,root, index, transform, patch_num):

        self.root=root
        self.config=config
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath,'.bmp.BMP')
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        refname.sort()
        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
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

        data_dict = {}

        if self.transform is not None:
            data_dict['image'] = self.transform(sample)

        data_dict = processing(data_dict, sample, self.transform, self.root, path, self.config)

        return data_dict, target

    def __len__(self):
        length = len(self.samples)
        return length


class Kadid10k(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath,'.png.PNG')
        # txtpath = os.path.join(root, 'dmos.txt')
        # fh = open(txtpath, 'r')



        imgnames = []
        target = []
        refnames_all = []

        csv_file = os.path.join(root, 'dmos.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgnames.append(row['dist_img'])
                refnames_all.append(row['ref_img'][1:3])

                mos = np.array(float(row['dmos'])).astype(np.float32)
                target.append(mos)




        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)


        refname.sort()
        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
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
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')