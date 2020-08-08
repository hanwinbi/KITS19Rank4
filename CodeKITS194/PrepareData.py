import os
import cv2 as cv
import torch
import numpy as np
import config
import json
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

dir = config.abs_data_root

# 获取到某个训练集的所有案例的路径
'''def load_all_data(file_path):
    data = []
    with open(file_path, 'r') as load_f:
        load_dict = json.load(load_f)
    for idx in range(len(load_dict)):
        data.append(load_dict[idx]['path'])
    print(data)
    return data
trainData = load_all_data(dir+'/trainData.json')
'''
# 这个部分可以和上面写在一起，因为我生成的Data.json文件既有目录也有图片起始位置
def load_case_data(case, idx):
    print(case)
    with open(case, 'r') as load_f:
        load_dict = json.load(load_f)
    caseData = []
    for i in range(config.depth):
        start = int(load_dict[idx]['start pos'].strip('.bmp'))
        slice_path = load_dict[idx]['path'] + str(start + i) + '.bmp'
        caseData.append(slice_path)
    print('case data', caseData)
    return caseData

# caseData = load_case_data(dir+'trainData.json', 0)

class DataSets(Dataset):
    def __init__(self, casedata):
        self.transform = transforms.Compose(
            [transforms.Normalize(mean=(0.485,), std=(0.229,))]
        )
        self.GT = casedata
        self.slice = config.depth  # 一个病人取32张切片

    def __len__(self):
        return len(self.GT)

    def __getitem__(self, idx):
        gt = self.GT[idx]
        img = self.getOriginImage(gt)

        slice_name = str(gt)
        gt = cv.imread(gt, 0)
        gt = cv.resize(gt, (160, 160))
        gt = gt/127
        gt = np.array(gt, dtype='int64')
        gt = self.n_class(gt)
        gt = gt.transpose(2, 0, 1)
        gt = torch.FloatTensor(gt)

        imgs = []  # 原始图片的序列list
        for i in img:
            pic = cv.imread(i, 0)
            pic = cv.resize(pic, (160, 160))
            imgs.append(pic)

        imgs = np.array(imgs, dtype='float32')
        imgs = torch.from_numpy(imgs)
        imgs = self.transform(imgs)

        return imgs, (gt, slice_name)

    def getOriginImage(self, gt_path):
        imgs = []
        for i in range(config.depth):
            origin_image = gt_path.replace('GT', 'Images')
            imgs.append(origin_image)
        return imgs

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