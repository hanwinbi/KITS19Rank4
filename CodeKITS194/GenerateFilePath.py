import os
import cv2
import json
import torch
import random
import config
from random import shuffle
from collections import OrderedDict

path = config.abs_path + '/'
dir = config.abs_data_root + '/'
json_dir = config.abs_data_root

json_dict = OrderedDict()

# 从seeds.json中按6：2：2比例得到训练集、验证集、测试集
def randomDiv(seeds_path, size):
    with open(seeds_path, 'r') as load_f:
        load_dict = json.load(load_f)
    seeds = load_dict['case']
    print(seeds)

    lenofsets = len(seeds)
    trainsize = int(size[0] * lenofsets)
    validationsize = int(size[1] * lenofsets)

    # 生成随机数作为seeds字典的idx
    idx = list(range(0, lenofsets))
    trainDataset = random.sample(idx, trainsize)
    restDataset = set(idx) - set(trainDataset)
    validationDataset = random.sample(restDataset, validationsize)
    testDataset = set(restDataset) - set(validationDataset)

    json_dict['train case'] = get_slice_include_aug(trainDataset, seeds)
    json_dict['test case'] = get_origin_slice(testDataset, seeds)
    json_dict['validation case'] = get_origin_slice(validationDataset, seeds)

    trainData_path = os.path.join(json_dir, 'trainData.json')
    testData_path = os.path.join(json_dir, 'testData.json')
    validationData_path = os.path.join(json_dir, 'validationData.json')

    with open(trainData_path, 'w') as f:
        traincase = json_dict["train case"]
        json.dump(traincase, f, indent=4)
    with open(testData_path, 'w') as f:
        testcase = json_dict['test case']
        json.dump(testcase, f, indent=4)
    with open(validationData_path, 'w') as f:
        validationcase = json_dict['validation case']
        json.dump(validationcase, f, indent=4)
    print('train data path', trainData_path)
    return trainData_path, testData_path, testData_path

# 训练集中包括数据增强部分
def get_slice_include_aug(random_seed, seeds):
    case_list = list()
    loop_time = int(seeds[0]['Total Image Num']/seeds[0]['Slice num'])  # 一个案例中遍历的次数，数据增强为四次
    # 遍历得到的随机种子，生成对应的list
    for idx in random_seed:
        file_list = sorted(os.listdir(seeds[idx]['GT']))
        slice_num = seeds[idx]['Slice num']  # 每个案例的切片数目不一样，获取案例的切片数
        start_pos = int((seeds[idx]['Slice num'] - config.depth) / 2)  # 得到此案例的中间切片位置
        for i in range(loop_time):
            start = file_list[start_pos + slice_num * i]
            slice_path = seeds[idx]['GT'] + start
            case_list.append(slice_path)
    shuffle(case_list)
    print('case list', case_list)
    return case_list

# 测试集和验证集中不包括数据增强
def get_origin_slice(random_seed, seeds):
    case_list = list()
    for idx in random_seed:
        file_list = sorted(os.listdir(seeds[idx]['GT']))
        start_pos = int((seeds[idx]['Slice num'] - config.depth) / 2)  # 得到此案例的中间切片位置
        start = file_list[start_pos]
        slice_path = seeds[idx]['GT'] + start
        case_list.append(slice_path)
    print(case_list)
    return case_list

trainData, validationData, testData = randomDiv(json_dir+'dataset.json', (0.6, 0.2, 0.2))
