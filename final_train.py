from mymodel import TwoLayerNN
from mydata import x_train,y_train
import pickle


def train_final_model():
    network = TwoLayerNN(
        lrate = 0.005,
        hidden_size=150,
        l2 = 1e-4)
    network.train_model(
        train_set=(x_train,y_train),
        print_result=True)
    return network


def save_model(network,filename):  # network: class defined below
    pickle.dump(network, open(filename, 'wb'))
    print(f'{filename} successfully saved.')



if __name__ == '__main__':
    # train and save the final model
    model = train_final_model()
    filename = 'mnist_model9819.pkl'
    save_model(model,filename)
    model.plot_loss_acc()   # plot the loss and accuracy