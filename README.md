# NNmnist

## 文件说明

rawdata文件夹中以.npz格式存放了mnist数据集原始文件（train:test=60000:10000）

总共6个.py文件:

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


## 训练和测试过程

将整个数据集分为训练集、验证集和测试集

   在训练集上训练、在验证集上验证，不断调整超参数最终选定一组表现良好的超参数。
   
   将训练集和验证集合在一起，以这组超参数训练一个最终的模型并保存。
   
   导入模型，在测试集上输出分类精度。

* 训练过程:
    * 在mymodel.py中编辑好TwoLayerNN类和train_model函数的相关细节（比如设定好batch_size,epoch_num）
    * 在param_search.py中依次执行lr_search,hdsize_search,l2reg_search这三个函数进行参数查找，选定一组表现良好的超参数
    * 运行final_train.py文件，以选定的超参数训练并保存最终的模型，同时绘制出loss和accuracy变化。（final_train_result.png对应了某次训练的结果）

* 测试过程：
    * 确保模型文件放置在"./model/"路径下，运行final_test.py文件，可输出准确率

## 其他注意事项：

* 如果直接运行param_search.py文件，将会以选定的超参数进行训练，并输出loss和accuracy，但不会保存模型。（loss_acc.png对应了某次训练的结果）
* 确保模型文件放置在"./model/"路径下，运行plotW1W2.py，可绘制网络参数的图片
* loss和accuracy的绘制函数位于mymodel.py的TwoLayerNN类中

