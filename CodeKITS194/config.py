import os

data_root = './data'
rela_source_path = './data/source'  # 原始数据路径
rela_train_path = './data/Train'  # 训练数据路径
rela_test_path = './data/Test'  # 测试数据路径

# 用于配置测试的路径
rela_path = './data/temp'
abs_path = os.path.abspath(rela_path) + '/'
abs_data_root = os.path.abspath(data_root) + '/'

print(rela_path, abs_path)

source_path = os.path.abspath(rela_source_path) + '/'
train_path = os.path.abspath(rela_train_path) + '/'
test_path = os.path.abspath(rela_test_path) + '/'

depth = 32