from mymodel import TwoLayerNN
from mydata import x_test,y_test
import pickle


def test(filename):  # 评估模型的样本外效果
    with open(filename, "br") as fh:
        model = pickle.load(fh)
    model.forward(x_test)
    acc = model.get_accuracy(x_test,y_test)
    return acc


if __name__ == '__main__':
    # test the final model
    filename = 'mnist_model9819.pkl'
    acc = test(filename)
    print("Test Accuracy:", 100 * acc, "%")
    

