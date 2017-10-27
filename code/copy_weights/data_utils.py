import numpy
import matplotlib.pyplot as plt
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
  """
  Load and preprocess CIFAR10 images.
  """

  num_classes = 10

  # Load CIFAR10 dataset
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  # Convert the class y into one-hot vector.
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  # The model by Madry et al. does not normalize features.

  return (x_train, y_train), (x_test, y_test)


def Visualize(x):
  """
  Visualize a (preprocessed) point x.
  """

  plt.imshow(x)
  plt.show()

def Normalize(x):
  """
  Map values from [0, 255] to [-0.5, 0.5].
  x: numpy array to be normalized.
  """
  x = x.astype(numpy.float32)
  x /= 255.0
  x -= 0.5

  return x

def Unnormalize(x):
  """
  Map values from [-0.5, 0.5] to [0, 255].
  x: numpy array to be unnormalized.
  """
  x += 0.5
  x *= 255.0
  x = x.astype(numpy.unit8)

