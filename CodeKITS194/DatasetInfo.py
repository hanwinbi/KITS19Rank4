import os
import config
import json
import cv2 as cv
from collections import OrderedDict

rela_path = config.rela_path + '/'
path = config.abs_path  # 数据集的绝对路径
dir = config.abs_data_root

output_json_folder = dir  # 输出所有数据信息的json文件夹目录
json_dict = OrderedDict()

# 获取数据集信息的方法
def dataset_info(path):
    cases = sorted(os.listdir(path))  # 将文件目录进行读取并排序
    json_dict['case num'] = len(cases)  # 创建字典，数据集中案例的数目
    json_dict['case'] = list()  # 案例列表
    ave_size = 0  # 所有案例的平均图片大小（这里其实是算的总大小）
    ave_slice_num = 0  # 所有案例的平均切片数目
    for case in cases:  # 遍历案例
        GT = str(path+case+'/GT/')  # gt案例路径
        Images = str(path+case+'/Images/')  # 原始图片路径
        slice_path = sorted(os.listdir(GT))  #
        total_image_num = len(slice_path)
        print('slice name', slice_path)
        count = 0  # 同一个分类的切片计数，0表示没有进行增强的
        for slice in slice_path:
            if slice[0] == '0':
                count += 1

        dirfile = str(path + case + '/GT/' + slice_path[0])  # 读取一个案例中的一张图片获得属性
        print('dirfile:', dirfile)
        img = cv.imread(dirfile)
        size = img.shape
        print(size)

        ave_size += size[0]
        ave_slice_num += count
        print("sum_size:{0},sum_slice:{1}".format(ave_size, ave_slice_num))

        # 把信息添加到字典中
        dict = {'GT': GT, "Images": Images, "Total Image Num": total_image_num, "Slice num": count, "Img Size": size}
        json_dict['case'].append(dict)
        print(case)

    json_dict['Average pic size'] = ave_size/len(cases)  # 图片的平均大小
    json_dict['Average slice num'] = ave_slice_num/len(cases)  # 平均的切片数量

    with open(os.path.join(output_json_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

dataset_info(rela_path)

# TODO 单个案例的数目，平均切片数量
# TODO 切片数量，进行补上切片
# TODO 获取bmp的大小