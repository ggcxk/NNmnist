from mymodel import TwoLayerNN
from mydata import train_set, valid_set



def lr_search(search_list):
    '''
    lr: 0.001 accuracy: 0.97475
    lr: 0.005 accuracy: 0.9790833333333333
    lr: 0.01 accuracy: 0.97075
    lr: 0.05 accuracy: 0.97725
    lr: 0.1 accuracy: 0.97575
    best learning_rate: 0.005
    '''
    best_acc = 0
    for lr in search_list:
        network = TwoLayerNN(
            learning_rate = lr,
            lr_decay = 0.9,
            hidden_size=150,
            L2_reg = 1e-4)
        network.train_model(train_set, valid_set)
        acc = max(network.info['validAcc'])
        print('lr:',lr,'accuracy:',acc)
        if acc > best_acc:
            best_acc = acc
            best_lr = lr
    print('best learning_rate:', best_lr)


    
def hdsize_search(search_list):
    '''
    hidden size: 50 accuracy: 0.9699166666666666
    hidden size: 100 accuracy: 0.9755
    hidden size: 150 accuracy: 0.9778333333333333
    hidden size: 200 accuracy: 0.9779166666666667
    best hidden size: 200
    '''
    best_acc = 0
    for hdsize in search_list:
        network = TwoLayerNN(
            learning_rate = 0.005,
            lr_decay = 0.9,
            hidden_size=hdsize,
            L2_reg = 1e-4)
        network.train_model(train_set, valid_set)
        acc = max(network.info['validAcc'])
        print('hidden size:',hdsize,'accuracy:',acc)
        if acc > best_acc:
            best_acc = acc
            best_hd = hdsize
    print('best hidden size:', best_hd)

def l2reg_search(search_list):
    '''
    l2 reg: 0.01 accuracy: 0.93775
    l2 reg: 0.001 accuracy: 0.9745833333333334
    l2 reg: 0.0001 accuracy: 0.9785
    l2 reg: 1e-05 accuracy: 0.978
    best l2 reg: 0.0001
    '''
    best_acc = 0
    for l2reg in search_list:
        network = TwoLayerNN(
            learning_rate = 0.005,
            lr_decay = 0.9,
            hidden_size=150,
            L2_reg = l2reg)
        network.train_model(train_set, valid_set)
        acc = max(network.info['validAcc'])
        print('l2 reg:',l2reg,'accuracy:',acc)
        if acc > best_acc:
            best_acc = acc
            best_l2reg = l2reg
    print('best l2 reg:', best_l2reg)


if __name__ == '__main__':
    # lr_search([0.001,0.005,0.01,0.05,0.1])
    # hdsize_search([50,100,150,200])
    # l2reg_search([0.01,0.001,1e-4,1e-5])
    
    # 最终选择如下参数
    network = TwoLayerNN(
        learning_rate = 0.005,
        hidden_size=150,
        L2_reg = 1e-4)
    network.train_model(train_set, valid_set,print_result=True)
    network.plot_loss_acc()




