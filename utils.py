import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle


class Net:
    def __init__(self):
        self.data = None
        self.W = None

    def feedData(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        self.data = [x_train, y_train, x_valid, y_valid, x_test, y_test]

    def addLayer(self, dim):
        assert(self.data != None), "No data has been added"
        if (self.W == None):
            self.W = []
            self.W.append(np.random.normal(
                0, 1 / np.sqrt(self.data[0].shape[1]), (self.data[0].shape[1], dim)))
        else:
            self.W.append(np.random.normal(
                0, 1 / np.sqrt(self.W[-1].shape[1]), (self.W[-1].shape[1], dim)))

    def solver(self, graphFlag=False):
        assert(self.W != None), "No layers have been added"
        assert(self.W[0].shape[0] == self.data[0].shape[1]
               ), "The network input dimension doesn't match the data"
        assert(self.W[-1].shape[1] == self.data[1].shape[1]
               ), "The network output dimension doesn't match the data"
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.data[
            0], self.data[1], self.data[2], self.data[3], self.data[4], self.data[5]
        # Call bb solver
        bb_solver(self.W, x_train, y_train, x_valid,
                  y_valid, x_test, y_test, graphFlag=False)


def readData(split_ratio=0.7):
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data  # we take all the features.
    y = iris.target

    X, y = shuffle(X, y)

    meanX = np.average(X, axis=0)

    # Subtract the mean from the data to make the training faster
    X -= meanX

    y = y.reshape((-1, 1))

    separating_index = int(split_ratio * y.shape[0])

    x_train, y_train, x_test, y_test = X[:separating_index], y[:
                                                               separating_index], X[separating_index:], y[separating_index:]

    separating_index1 = int(0.33 * y_test.shape[0])
    x_valid = x_test[:separating_index1]
    y_valid = y_test[:separating_index1]
    x_test = x_test[separating_index1:]
    y_test = y_test[separating_index1:]

    def make_hot(a):
        numdata = a.shape[0]
        rval = np.zeros((numdata, 3))
        for i in range(numdata):
            rval[i, a[i]] = 1
        return rval

    # convert labels to ones hot using this function
    y_test = make_hot(y_test)
    y_valid = make_hot(y_valid)
    y_train = make_hot(y_train)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def list_to_vec(list_of_matrices):
    """Convert a list of matrices into a vector"""
    return np.concatenate([l.ravel() for l in list_of_matrices])


def vec_to_list(vec, list_of_matrices):
    """Convert a vector into a list of matrices.  The returned value will have the same shape as list_of_matrices,
    and will overwrite the values in that list"""
    loc = 0
    for m in list_of_matrices:
        shape = m.shape
        m[:] = vec[loc:loc + np.size(m)].reshape(shape)
        loc = loc + np.size(m)
    return list_of_matrices


def logreg_objective(x, D, c):
    z = np.multiply(c, np.dot(D, x))
    l_z = np.zeros_like(z)
    # We take exp() of only the negative values in z
    l_z[z <= 0] = -z[z <= 0] + np.log(1 + np.exp(z[z <= 0]))
    # We take exp(-z) for positive values in z
    l_z[z > 0] = np.log(1 + np.exp(-z[z > 0]))
    return np.sum(l_z)


def logreg_grad(x, D, c):
    z = np.multiply(c, np.dot(D, x))
    cez = np.zeros_like(z)
    cez[z >= 0] = np.multiply(
        c[z >= 0], np.divide(-np.exp(-z[z >= 0]), 1 + np.exp(-z[z >= 0])))
    cez[z < 0] = np.multiply(c[z < 0], -1 / (1 + np.exp(z[z < 0])))
    return np.dot(D.transpose(), cez)


def log_entropy_softmax(z, ones_hot):
    """The log entropy of the softmax layer"""
    # shift everything so we don't have to exponentiate positive numbers
    z = z - np.max(z, axis=1)[:, None]
    # compute the negative log likelihood
    s = np.sum(np.exp(z), axis=1)[:, None]
    nll = -z + np.log(s)
    # sum over the entries corresponding to the correct class
    return np.sum(nll * ones_hot) / z.shape[0]


def net_objective(weights, data, labs):
    """The objective function of a neural net"""
    num_lays = len(weights)
    z = data.dot(weights[0])
    # Each layer performs data*weights.  This way we have 1 feature vector per row of data
    for j in range(1, num_lays):
        z = smrelu(z).dot(weights[j])
    return log_entropy_softmax(z, labs)


def smrelu(x):
    y = np.zeros_like(x)
    # take exp() of only the negative values in x
    y[x <= 0] = np.log(1 + np.exp(x[x <= 0]))
    # take exp(-x) for positive values in x, so exp doesn't blow up
    y[x > 0] = x[x > 0] + np.log(1 + np.exp(-x[x > 0]))
    return y


def smrelu_grad(x):
    """The smoothed relu gradient"""
    rval = np.zeros(x.shape)
    ind = x < 0
    rval[ind] = np.exp(x[ind]) / (1 + np.exp(x[ind]))
    ind = np.negative(ind)
    rval[ind] = 1 / (1 + np.exp(-x[ind]))
    return rval


def log_entropy_grad(z, ones_hot):
    """The gradient of the log entropy of the softmax"""
    # shift everything so we don't have to exponentiate positive numbers
    z = z - np.max(z, axis=1)[:, None]
    # compute the negative log likelihood
    s = np.sum(np.exp(z), axis=1)[:, None]
    grad = -ones_hot + np.exp(z) / s
    return grad / z.shape[0]


def net_grad(weights, data, labs):
    """The gradient of the neural net objective"""
    num_lays = len(weights)
    # Forward pass:  Each layer performs y_next = sigma(y*weights).  This way we have 1 feature vector per column
    z = [data.dot(weights[0])]  # The y's are activations
    y = [data]
    for j in range(1, num_lays):
        y.append(smrelu(z[j - 1]))
        z.append(y[j].dot(weights[j]))

    # Backward pass: loop over the layers a produce gradients
    # This is how much the loss depends on it's input
    dzt = log_entropy_grad(z[-1], labs)
    # this is how much the loss depends on the deepest weight matrix
    dw = [y[-1].T.dot(dzt)]
    # loop over remaining layers
    for j in reversed(range(num_lays - 1)):
        # gradient with respect to z
        dzt = dzt.dot(weights[j + 1].T) * smrelu_grad(z[j])
        # gradient with respect to W
        dw.append(y[j].T.dot(dzt))
    dw.reverse()
    return dw

# We define a separate function to calculate the initial step size for this problem


def getInitialStep(grad, W, x0):
    y = x0 + np.random.normal(0, 0.01, x0.shape)
    return 2 * np.linalg.norm(y - x0) / np.linalg.norm(grad(vec_to_list(y, W)) - grad(vec_to_list(x0, W)))


def bb_solver(W, x_train, y_train, x_valid, y_valid, x_test, y_test, graphFlag=False):
    D = x_train
    L = y_train

    # Define the function handles
    def f(W):
        # run the training data through the network
        return net_objective(W, D, L)

    def grad(W):
        return list_to_vec(net_grad(W, D, L))

    # Run the BB solver
    x0 = list_to_vec(W)
    alpha = 0.1
    res = []
    norm_grad0 = np.linalg.norm(grad(vec_to_list(x0, W)))
    res.append(norm_grad0)
    norm_grad_curr = 1e18  # Initialize it to infinity
    x_curr = x0
    x_old = x0
    step_curr = getInitialStep(grad, W, x0)
    for iter in range(200):
        d = -grad(vec_to_list(x_curr, W))
        delta_x = x_curr - x_old
        delta_g = grad(vec_to_list(x_curr, W)) - grad(vec_to_list(x_old, W))
        norm_delta_x = np.linalg.norm(delta_x)
        if norm_delta_x == 0:
            step_curr = getInitialStep(grad, W, x0)
        else:
            step_curr = norm_delta_x / \
                np.inner(delta_x.flatten(), delta_g.flatten())
        while f(vec_to_list(x_curr + step_curr * d, W)) > ((f(vec_to_list(x_curr, W)) + alpha * np.inner(step_curr * d.flatten(), grad(vec_to_list(x_curr, W)).flatten()))):
            step_curr = step_curr * 0.5
        x_old = x_curr
        x_curr = x_curr + step_curr * d
        norm_grad_curr = np.linalg.norm(grad(vec_to_list(x_curr, W)))
        res.append(norm_grad_curr)
        # print norm_grad_curr

    W = vec_to_list(x_curr, W)  # the final learned weights

    # Calculate the train and test accuracies
    num_layers = len(W)
    output = x_train  # initialize output to input
    for i in range(num_layers - 1):  # last layer has softmax non-linearity
        # new output after passing through a layer
        output = smrelu(np.dot(output, W[i]))
    output = np.dot(output, W[-1])  # multiply with final layer weights
    # the index of the max value in the 10x1 output of the neural net is the predicted label
    train_predicted = np.argmax(output, axis=1)
    accuracy_train = np.sum(train_predicted == np.argmax(
        y_train, axis=1)) * 100.0 / x_train.shape[0]
    print "Training Accuracy = ", accuracy_train

    output = x_test  # initialize output to input
    for i in range(num_layers - 1):  # last layer has softmax non-linearity
        # new output after passing through a layer
        output = smrelu(np.dot(output, W[i]))
    output = np.dot(output, W[-1])  # multiply with final layer weights
    # the index of the max value in the 10x1 output of the neural net is the predicted label
    test_predicted = np.argmax(output, axis=1)
    accuracy_test = np.sum(np.argmax(output, axis=1) == np.argmax(
        y_test, axis=1)) * 100.0 / x_test.shape[0]
    print "Test Accuracy = ", accuracy_test

    if (graphFlag):
        # Plot the convergence curve
        plt.figure(1)
        plt.plot(range(len(res)), res)
        plt.yscale('log')
        plt.title('Residuals of Neural Network with Barzilai-Borwein solver')
        plt.xlabel('Number of iterations')
        plt.ylabel('Residual')
        plt.show()
