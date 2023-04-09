import numpy as np


def change_onehot(y):
    mat = np.zeros((y.size, 10))
    for idx, row in enumerate(mat):
        row[y[idx]] = 1
    return mat

def load_data():
    # 读取数据并预处理成neural network接受的格式
    path = './rawdata/mnist.npz'
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    x_train, x_test = x_train / 255.0, x_test / 255.0 #归一化
    x_train = x_train.reshape(-1,784)  # flatten, (60000,28,28)变（60000,784）
    x_test = x_test.reshape(-1,784)  # flatten, (10000,28,28)变（10000,784）
    y_train = change_onehot(y_train) #标签变独热码，才能和前向传播softmax之后的结果维度匹配，才能相减算误差
    y_test = change_onehot(y_test) 
    return x_train, x_test, y_train, y_test

def train_valid_split(x,y,valid_ratio):  #valid_ratio  验证集的比例
    shuffle_index = np.random.permutation(len(x)) 
    valid_size = int(len(x)*valid_ratio) 
    valid_index = shuffle_index[:valid_size] 
    train_index = shuffle_index[valid_size:] 
    train_set = (x[train_index],y[train_index])
    valid_set = (x[valid_index],y[valid_index])
    return train_set, valid_set


# 初步导入数据  train num: 60000   test num: 10000
x_train, x_test, y_train, y_test = load_data()  
# 将训练集进一步划分为   80% train  20% valid        48000:12000
train_set, valid_set = train_valid_split(x_train,y_train,0.2)




