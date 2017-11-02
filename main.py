from utils import *

if __name__ == "__main__":
    # Instantiate neural network object
    nn = Net()
    # Pre-process data
    x_train, y_train, x_valid, y_valid, x_test, y_test = readData()
    # Feed data into neural network
    nn.feedData(x_train, y_train, x_valid, y_valid, x_test, y_test)
    # Add the layers
    nn.addLayer(50)
    nn.addLayer(40)
    nn.addLayer(30)
    nn.addLayer(20)
    nn.addLayer(y_train.shape[1])
    # Start the NN solver
    nn.solver()
