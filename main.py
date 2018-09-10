from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from models.lenet5 import LeNet5
from models.base import OneHotData

mnist = input_data.read_data_sets("data/raw/mnist", reshape=False, one_hot=True)

X_train, y_train = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels


mnist_data = OneHotData('mnist', 10,
                        np.pad(mnist.train.images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant'),
                        mnist.train.labels,
                        np.pad(mnist.validation.images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant'),
                        mnist.validation.labels,
                        np.pad(mnist.test.images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant'),
                        mnist.test.labels)

print("Input shape: {}".format(X_train[0].shape))
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


print("New Input shape: {}".format(X_train[0].shape))

lenet_network = LeNet5()
accuracy = lenet_network.train(mnist_data, epochs=10, batch_size=101)
print("Accuracy on test set: {:.3f}".format(accuracy))