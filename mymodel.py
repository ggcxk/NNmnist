import numpy as np
import matplotlib.pyplot as plt



def ReLU(x):
    return np.maximum(0,x)  

def ReLU_grad(x):
    return np.where(x>=0,1,0)

def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(z)/np.sum(np.exp(z), axis=-1, keepdims=True)

def cross_entropy_error(p, y):
    # 将one-hot label转化为对应数值
    y = y.argmax(axis=1)
    size = p.shape[0]
    return -np.sum(np.log(p[np.arange(size), y] + 1e-7)) / size



    
class TwoLayerNN:
    def __init__(self, lrate, lr_decay, hidden_size, l2, input_size=784, output_size=10):
        # 初始化权重
        self.W1 = np.random.normal(0, np.sqrt(2/input_size), (input_size, hidden_size))
        self.W2 = np.random.normal(0, np.sqrt(2/hidden_size), (hidden_size, output_size))
        self.b1 = np.random.normal(0, np.sqrt(2/input_size), (hidden_size,))
        self.b2 = np.random.normal(0, np.sqrt(2/hidden_size), (output_size,))
        # 初始化超参数
        self.l2 = l2                 # L2正则化强度
        self.lrate = lrate           # 学习率
        self.lr_decay = lr_decay     # 学习率衰减系数
        # 定义一个字典用于保存模型的loss和accuracy信息
        self.info = {'trainLoss':[],'trainAcc':[],'validLoss':[],'validAcc':[]}

    def forward(self,x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = ReLU(self.z1)    
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2) 

    def backward(self,x,y):
        dz2 = (self.a2 - y) / y.shape[0]   # softmax with 交叉熵
        dW2 = np.dot(self.a1.T,dz2) + self.l2 * self.W2
        db2 = np.sum(dz2,axis = 0)
        dz1 = np.dot(dz2,self.W2.T) * ReLU_grad(self.z1)
        dW1 = np.dot(x.T,dz1) + self.l2 * self.W1
        db1 = np.sum(dz1,axis = 0)

        # 梯度下降
        self.W2 -= self.lrate * dW2
        self.W1 -= self.lrate * dW1 
        self.b2 -= self.lrate * db2
        self.b1 -= self.lrate * db1

    def get_loss(self,y):
        # 损失函数为交叉熵加L2正则（F范数的平方和）    
        cross_entropy = cross_entropy_error(self.a2, y)  #添加一个微小值1e-7可以防止np.log(0)的发生
        L2_reg = 1/2*self.l2*(np.sum(self.W1*self.W1)+np.sum(self.W2*self.W2))
        return cross_entropy + L2_reg

    def get_accuracy(self,x,y):
        accurate_num = np.sum(np.argmax(self.a2, axis=1) == np.argmax(y, axis=1)) 
        accuracy = accurate_num / float(x.shape[0])
        return accuracy
    
    def plot_loss_acc(self):
        # 绘制 loss 曲线
        plt.subplot(1,2,1)
        plt.title('Loss Curve')  # 图片标题
        plt.xlabel('epoch')  # x轴变量名称
        plt.ylabel('Loss')  # y轴变量名称
        plt.plot(self.info['trainLoss'], label="$train$")  # 逐点画出loss值并连线
        if self.isvaliding:
            plt.plot(self.info['validLoss'], label="$valid$")  # 逐点画出loss值并连线
        plt.legend()  # 画出曲线图标

        # 绘制 Accuracy 曲线
        plt.subplot(1,2,2)
        plt.title('Accuracy Curve')  # 图片标题
        plt.xlabel('epoch')  # x轴变量名称
        plt.ylabel('Acc')  # y轴变量名称
        plt.plot(self.info['trainAcc'], label="$train$")  # 逐点画出train_acc值并连线
        if self.isvaliding:
            plt.plot(self.info['validAcc'], label="$valid$")  # 逐点画出valid_acc值并连线
        plt.legend()
        plt.show()

    
    def train_model(self,train_set,valid_set=None,print_result=False,batch_size=128,epoch_num=40):
        self.isvaliding = valid_set is not None
        x_train,y_train = train_set
        train_num = x_train.shape[0]
        batch_num = train_num // batch_size
        for epoch in range(epoch_num):
            for _ in range(batch_num): 
                batch_mask = np.random.choice(train_num, batch_size)
                x_batch = x_train[batch_mask]
                y_batch = y_train[batch_mask]
                self.forward(x_batch)
                self.backward(x_batch,y_batch)
            
            # 学习率增减策略
            if epoch == 0:
                ascend_rate = (0.7/self.lrate)**0.1
            if epoch < 10:
                self.lrate *= ascend_rate
            else:
                self.lrate *= self.lr_decay

            # 分别计算train set, valid set上的loss & accuracy
            self.forward(x_train)
            train_loss = self.get_loss(y_train)
            train_acc = self.get_accuracy(x_train,y_train)
            self.info['trainLoss'].append(train_loss)
            self.info['trainAcc'].append(train_acc)

            if self.isvaliding:   # during parameter-search process
                x_valid,y_valid = valid_set
                self.forward(x_valid)
                valid_loss = self.get_loss(y_valid)
                valid_acc = self.get_accuracy(x_valid,y_valid)
                self.info['validLoss'].append(valid_loss)
                self.info['validAcc'].append(valid_acc)
                if print_result:
                    print(
                        'Epoch ',f'{epoch+1}/{epoch_num}',
                        'train loss:',f'{train_loss:.4f}',
                        'train acc:',f'{100*train_acc:.2f}%',
                        'valid loss:',f'{valid_loss:.4f}',
                        'valid acc:',f'{100*valid_acc:.2f}%'
                    )
        
            else:                # not in parameter-search process   找到合适参数后，将60000个样本全部用于训练，故没有验证集
                if print_result:
                    print(
                        'Epoch ',f'{epoch+1}/{epoch_num}',
                        'train loss:',f'{train_loss:.4f}',
                        'train acc:',f'{100*train_acc:.2f}%'
                    )




