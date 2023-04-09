# NNmnist

rawdata文件夹中以.npz格式存放了mnist数据集原始文件（train:test=60000:10000）

总共6个.py文件：
    mydata.py --- 该文件进行数据预处理。从 "./rawdata/" 目录下导入mnist数据集，并将数据划分为训练集（48000张图片）、验证集（12000）、测试集（10000）
    mymodel.py --- 模型基本架构文件。定义了TwoLayerNN类，包含前向过程、反向传播、计算损失和准确率、模型训练等函数
    param_search.py --- 该文件进行超参数查找。查找的超参数包括学习率、隐藏层大小、正则化强度，查找过程在“训练集”上训练，在“验证集”上输出分类精度，并以该精度衡量模型优劣
    final_train.py --- 最终训练文件。在“训练集+验证集”上训练最终的模型，并保存为.pkl格式
    final_test.py --- 最终测试文件。导入.pkl文件，并在“测试集”上输出模型精度
    plotW1W2.py --- 该文件进行网络参数的可视化。导入.pkl文件，分别可视化W1和W1·W2

pics文件夹保存了6张图片，其中
    mnist_100.png     是从mnist数据集中任意选取的100张数字图
    lrate.png         可视化了每个epoch的学习率的变化情况（初始学习率设定为0.005）
    loss_acc.png      是以一组表现较好的参数（经过参数查找后选定的），在训练集和验证集上跑40个epoch所绘制出的loss和accuracy曲线
    final_train_result.png    是以上述选定的参数在“训练集+验证集”共60000个样本中训练所得到的loss和accuracy曲线
    W1.png            可视化最终模型的W1
    W1W2.png          可视化最终模型的W1·W2

