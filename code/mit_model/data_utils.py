import numpy
import matplotlib.pyplot as plt
import tensorflow
import keras
from keras.datasets import cifar10


def GetDataPoint(X, Y, index):
  # Get the feature vector.
  x = X[index]
  # Here we assume that x to be a batch of images.
  # TODO: When we use multiple point (e.g. list of test indices)
  #   we need to make sure that tx.shape = (N, 32, 32, 3)
  # TODO: This may not be needed if image_index is a index list.
  x = x[numpy.newaxis, :]
  # The code assumes y to be a batch of one-hot vectors.
  y = Y[index]
  return x, y


def LoadCIFAR10():
  """Load and preprocess CIFAR10 images.

  """

  num_classes = 10

  # Load CIFAR10 dataset
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  # Convert the class y into one-hot vector.
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  # Normalize the feature vectors
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255.
  x_test /= 255.
  x_train -= 0.5
  x_test -= 0.5

  return (x_train, y_train), (x_test, y_test)


def Visualize(x):
  """Visualize a (preprocessed) point x.

  """

  plt.imshow(x + 0.5)
  plt.show()

