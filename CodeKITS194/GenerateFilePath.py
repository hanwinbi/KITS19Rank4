import os
from collections import OrderedDict
import cv2
import random
import config
import json

path = config.abs_path + '/'
dir = config.abs_data_root + '/'
json_dir = config.abs_data_root

json_dict = OrderedDict()

# TODO 将所有案例作为种子
# TODO 选取训练、测试和验证集

#  生成seeds.json
def get_all_case(data_info_json_path):
    dir = os.path.join(data_info_json_path, 'dataset.json')  # 获取所有数据的相关信息
    with open(dir, 'r') as load_f:
        load_dict = json.load(load_f)
    length_of_data = load_dict['case num']  # case的数量，共11个
    loop_times = 4  # 因为只用了四次增强，可以用load_dict['case'][0]['Total Image Num']/load_dict['case'][0]['Slice num']计算
    json_dict['all cases'] = list()

    for i in range(length_of_data):
        slice_num = load_dict['case'][i]['Slice num']
        start_pos = int((load_dict['case'][i]['Slice num'] - config.depth)/2)  # 计算中间32张切片的起始位置
        case_dir = load_dict['case'][i]['GT']
        list_of_case = sorted(os.listdir(case_dir))
        for j in range(loop_times):
            path = load_dict['case'][i]['GT']

            start = list_of_case[start_pos + slice_num * j]
            end = list_of_case[start_pos + config.depth + slice_num * j - 1]

            dict = {'path': path, 'start pos': start, 'end pos': end}
            json_dict['all cases'].append(dict)
            print('start pos:{0},end pos:{1}'.format(start, end))
    with open(os.path.join(json_dir, 'seeds.json'), 'w') as f:
        json.dump(json_dict, f, indent=4)

# 从seeds.json中按6：2：2比例得到训练集、验证集、测试集
def randomDiv(seeds_path, size):
    with open(seeds_path, 'r') as load_f:
        load_dict = json.load(load_f)
    seeds = load_dict['all cases']
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

    json_dict['train case'] = list()
    json_dict['test case'] = list()
    json_dict['validation case'] = list()
    for idx in trainDataset:
        json_dict["train case"].append(load_dict['all cases'][idx])
    for idx in testDataset:
        json_dict['test case'].append(load_dict['all cases'][idx])
    for idx in testDataset:
        json_dict['validation case'].append(load_dict['all cases'][idx])

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

get_all_case(json_dir)
trainData, validationData, testData = randomDiv(json_dir+'seeds.json', (0.6, 0.2, 0.2))
