from mymodel import TwoLayerNN
import numpy as np
import matplotlib.pyplot as plt
import pickle



def plot_para(W, height,width, plot_size):  
     # 可视化每层的网络参数
    m, n = plot_size
    fig, axes = plt.subplots(m, n)
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    for col, ax in zip(W.T, axes.ravel()):
        ax.matshow(col.reshape(height,width), cmap=plt.cm.gray, vmin=0.3*W.min(), vmax=0.3*W.max())
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()


if __name__=='__main__':
    filename = 'mnist_model9819.pkl'
    with open(filename, "br") as fh:
        model = pickle.load(fh)
    plot_para(model.W1, 28, 28, (10, 15))   # plot W1 (784 x hidden_size)
    plot_para(np.dot(model.W1, model.W2), 28, 28, (2, 5))   # plot W1·W2   (784 x 10)



