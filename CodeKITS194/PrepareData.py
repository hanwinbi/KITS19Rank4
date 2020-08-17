import os
import cv2 as cv
import torch
import numpy as np
import config
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

dir = config.abs_data_root

# 载入测试集[验证集、测试集]案例
def load_data(path):
    with open(path, 'r') as load_f:
        load_dict = json.load(load_f)
    data = []
    for i in range(len(load_dict)):
        data.append(load_dict[i])
    return data

trainData = load_data(dir+'trainData.json')
testData = load_data(dir+'testData.json')
validationData = load_data(dir+'validationData.json')

# 载入一个样例中的全部切片
def load_case_data(case):
    print(case)
    caseData = []
    for i in range(config.depth):
        start = int(case[-8:-4])  # 起始图片的序号
        slice_path = case[0:-8] + str("%04d" % (start+i)) + '.bmp'  # 切片数是32，连续的32张
        caseData.append(slice_path)
    return caseData

# caseData = load_case_data(dir+'trainData.json', 0)

class DataSets(Dataset):
    def __init__(self, casedata):
        self.transform = transforms.Compose(
            [transforms.Normalize(mean=(0.485,), std=(0.229,))]
        )
        self.GT = casedata
        print('case data', casedata)
        print('len GT', len(self.GT))
        self.slice = config.depth  # 一个病人取32张切片

    def __len__(self):
        return len(self.GT)

    def __getitem__(self, idx):
        # print('idx:', idx)
        # gt = self.getGroundTruth()
        # print(gt)
        # img = self.getOriginImage()
        # print('img', img)

        # slice_name = str(gt)
        # gt = cv.imread(gt, 0)
        # gt = cv.resize(gt, (160, 160))
        # gt = gt/127
        # gt = np.array(gt, dtype='int64')
        # gt = self.n_class(gt)
        # gt = gt.transpose(2, 0, 1)
        # gt = torch.FloatTensor(gt)

        # imgs = []  # 原始图片的序列list
        # for i in img:
        #     pic = cv.imread(i, 0)
        #     pic = cv.resize(pic, (160, 160))
        #     imgs.append(pic)
        slice_name = self.GT
        gts = self.getGroundTruth()
        imgs = self.getOriginImage()

        gts = np.array(gts, dtype='int64')
        print('gts', gts.shape)
        gts = torch.from_numpy(gts)

        imgs = np.array(imgs, dtype='float32')
        print('imgs', imgs.shape)
        imgs = torch.from_numpy(imgs)
        imgs = self.transform(imgs)

        return imgs, (gts, slice_name)

    def getOriginImage(self):
        casedata = self.GT
        imgs = []
        for i in range(config.depth):
            origin_image = casedata[i].replace('GT', 'Images')
            pic = cv.imread(origin_image, 0)
            pic = cv.resize(pic, (160, 160))
            imgs.append(pic)
        return imgs

    def getGroundTruth(self):
        casedata = self.GT
        gts = []
        for i in range(config.depth):
            gt = cv.imread(casedata[i], 0)
            gt = cv.resize(gt, (160, 160))
            gt = gt / 127
            gts.append(gt)
        return gts

    def n_class(self, data, n=3):
        # one-hot
        buf = np.zeros(data.shape + (n,))
        buf = buf.transpose((2, 0, 1))
        buf[0] = np.where(data == 0, 1, 0)
        buf[1] = np.where(data == 1, 1, 0)
        buf[2] = np.where(data == 2, 1, 0)
        buf = buf.transpose((2, 1, 0))
        buf = buf.transpose((1, 0, 2))
        return buf

# exam = DataSets(caseData)
# loader = DataLoader(exam, batch_size=1)
# for item in loader:
#     print('=======================')
#     print(item)