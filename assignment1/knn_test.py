######################################################################################
# k-Nearest Neighbor test
#   This classifier takes the training data and simply remembers it.
#   Every test image is compapred to all training images.
#   The labels of the k most similar training examples is transfered.
#   The value of k is cross-validated.
######################################################################################

import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor


# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
# Print out the size of the training and test data.
# print 'Training data shape: ', X_train.shape
# print 'Training labels shape: ', y_train.shape
# print 'Test data shape: ', X_test.shape
# print 'Test labels shape: ', y_test.shape


# Subsample the data
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 100
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
# print X_train.shape, X_test.shape


# Create a KNN classifier instance and remember the data for training.
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# Compute_distances_two_loops.
distance_type = 2
# dists = classifier.compute_distances_two_loops(X_test, distance_type)
#
# # Predict_labels and compute the accuracy
# y_test_pred = classifier.predict_labels(dists, k=1)
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)




# Cross-validation
# fold = 5 for test
num_folds = 5
k_choices = np.arange(1,10)    # a sery of number for choosing the best k for kNN

# Split up the trainging date into folds.
# X_train_folds = []
# y_train_folds = []
X_train_folds = np.array_split(X_train, num_folds, axis=0)
y_train_folds = np.array_split(y_train, num_folds, axis=0)

# A dictionary holding the accuracies for different values of k
k_accuracy = {}

for k in k_choices:
    k_accuracy[k] = np.zeros(num_folds)
    for i in range(num_folds):
        # Train split 80%
        X_tr = X_train_folds[0:i] + X_train_folds[(i + 1):5]
        X_tr = np.reshape(X_tr, (X_train.shape[0] * 4 / 5, -1))
        y_tr = y_train_folds[0:i] + y_train_folds[(i + 1):5]
        y_tr = np.reshape(y_tr, (X_train.shape[0] * 4 / 5))

        # Validation split 20%
        # Train and predict
        X_tt = np.reshape(X_train_folds[i], (X_train.shape[0] * 1 / 5, -1))
        classifier.train(X_tr, y_tr)
        y_test_pred = classifier.predict(X_tt, k, distance_type)

        # Compute accuracy
        num_correct = np.sum(y_test_pred == y_train_folds[i])
        accuracy = float(num_correct) / num_test
        k_accuracy[k][i] = accuracy

# Print out the computed accuracies to the console and save them into a text file.
f = open('./knn.log','w')
for k in sorted(k_accuracy):
    for accuracy in k_accuracy[k]:
        print >> f, 'k = %d, accuracy = %f' % (k, accuracy)
f.close()


# Plot the raw observations.
for k in k_choices:
    accuracy = k_accuracy[k]
    plt.scatter([k] * len(accuracy), accuracy)

accuracy_mean = np.array([np.mean(v) for k, v in sorted(k_accuracy.items())])
accuracy_std = np.array([np.std(v) for k, v in sorted(k_accuracy.items())])
plt.errorbar(k_choices, accuracy_mean, yerr=accuracy_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()


# Use the best k to get the highest accuracy about 28% on the test data.
accuracy = 0
for k in k_choices:
    accuracy_mean = np.mean(k_accuracy[k])
    if accuracy_mean >= accuracy:
        accuracy = accuracy_mean
        best_k = k

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print '\n'
print 'The best k for cross-validation is %d' % (best_k)
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

