import numpy as np


class kNearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
        # Reshape the image data into rows
        X_train = X_train.reshape(X_train.shape[0], -1)
        Y_train = X_train.reshape(X_train.shape[0], -1)
        
        print "Training process finished!"
        return
    
    def distances(self, X_test, dist_loop):
        num_train = self.X_train.shape[0]
        num_test = X_test.shape[0]    
        dists = np.zeros((num_test, num_train))
        
        if dist_loop == 2:
            for i in xrange(num_test):
                for j in xrange(num_train):
                    dists[i, j] = np.sqrt(np.sum((X_test[i] - self.X_train[j]) ** 2))
        elif dist_loop == 1:
            for i in xrange(num_test):
                dists[i, :] = np.sqrt(np.sum((X_test[i] - self.X_train) ** 2, axis = 1))
        elif dist_loop == 0:
            X_test_square_sum = np.sum(X_test ** 2, axis = 1, keepdims = True)
            X_train_square_sum = np.sum(self.X_train ** 2, axis = 1)
            dists = np.sqrt(X_test_square_sum + X_train_square_sum - 2 * np.dot(X_test, self.X_train.T))
        else:
            print "Invalid input of dist_loop."
            print "The argument of dist_loop should be the value among 0, 1 and 2."
            assert(False)
            
        print "Distances computation finished!"
        return dists
    
    
    def predict(self, X_test, k, dist_loop):        
        dists = self.distances(X_test, dist_loop)      
        assert(dists.shape == (X_test.shape[0], self.Y_train.shape[0]))
        num_test = X_test.shape[0]
        Y_pred = np.zeros(num_test)
    
        # np.argsort return an array of the indexes of smallest value in the source list.
        for i in xrange(num_test):
            Y_closest = []
            Y_closest = np.argsort(dists[i, :])[ :k]
            
            # np.argmax return the value appears for the most time in the source array
            cnt = np.bincount(Y_closest)
            Y_pred[i] = np.argmax(cnt)
            
        print "Prediction process finished!"
        return Y_pred
    
    def compute_test_accuracy(self, Y_pred, Y_test):
        num_correct = np.sum(Y_pred == Y_test)
        accuracy = float(num_correct) / num_test
        print
        print "Correct: %d / %d ; The accuracy is %f" % (num_correct, num_test, accuracy)
        
        return
    
    def runModel(self, X_train, Y_train, X_test, Y_test, k=5, dist_loop=2):
        self.train(X_train, Y_train)
        Y_pred = self.predict(X_test, k, dist_loop)
        self.compute_test_accuracy(Y_pred, Y_test)
        
        return