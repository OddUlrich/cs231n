import cPickle as pickle
import numpy as np
import os


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """

    # Initialize the list to store the original data
    x_buf = []
    y_buf = []
    
    # Get the train data from the batch
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        x_buf.append(X)
        y_buf.append(Y)    
        X_train = np.concatenate(x_buf)
        Y_train = np.concatenate(y_buf)
        del X, Y

    # Get the test data
    X_test, Y_test = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    
    return X_train, Y_train, X_test, Y_test