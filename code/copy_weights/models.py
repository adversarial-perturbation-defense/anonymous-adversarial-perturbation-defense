import numpy
import tensorflow
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Input
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Permute
from keras.layers.core import Reshape
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.initializers import RandomNormal
from keras.initializers import VarianceScaling

# CIFAR10 dataset assumes the following input shape, and number of classes.
input_shape = (32, 32, 3)
num_classes = 10

def ConvSmall():
  '''
  Load the Conv-Small model in Miyato et al. paper.
  '''
  model = Sequential()
  initializer = VarianceScaling(scale=2.0)

  model.add(Conv2D(96, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer, input_shape=input_shape))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(96, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(96, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(192, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(192, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(192, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(192, (3,3), padding='valid', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(192, (1,1), use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(192, (1,1), use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(AveragePooling2D(pool_size=(6,6)))

  model.add(Flatten())
  model.add(Dense(num_classes, kernel_initializer=initializer, name='logits'))
  model.add(Activation('softmax'))

  return model

def ConvLarge():
  '''
  Load the Conv-Large model in Miyato et al. paper.
  '''
  model = Sequential()
  initializer = VarianceScaling(scale=2.0)

  model.add(Conv2D(128, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer, input_shape=input_shape))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(128, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(128, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(256, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(256, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(256, (3,3), padding='same', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(512, (3,3), padding='valid', use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(256, (1,1), use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Conv2D(128, (1,1), use_bias=False,
      kernel_initializer=initializer))
  model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-6,
      center=True, scale=True))
  model.add(LeakyReLU(alpha=0.1))
  model.add(AveragePooling2D(pool_size=(6,6)))

  model.add(Flatten())
  model.add(Dense(num_classes, kernel_initializer=initializer, name='logits'))
  model.add(Activation('softmax'))

  return model

def ResidualUnit(x, in_filter, out_filter, stride, init_conv, 
                 block_index, unit_index):
  '''
  Residual unit used in ResNet implementation.
  '''
  strides = (stride, stride)

  suffix1 = '_' + str(block_index) + '_' + str(unit_index) + '_1'
  orig_x = x
  x = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3,
      center=True, scale=True, name = 'BN'+suffix1)(x)
  x = LeakyReLU(alpha=0.1)(x)

  x = Conv2D(out_filter, (3,3), strides=strides, padding='same',
      use_bias=False, kernel_initializer=init_conv, input_shape=input_shape,
      name = 'conv'+suffix1)(x)
  
  suffix2 = '_' + str(block_index) + '_' + str(unit_index) + '_2'
  x = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3,
      center=True, scale=True, name = 'BN'+suffix2)(x)
  x = LeakyReLU(alpha=0.1)(x)
  x = Conv2D(out_filter, (3,3), strides=(1,1), padding='same',
      use_bias=False, kernel_initializer=init_conv, input_shape=input_shape,
      name = 'conv'+suffix2)(x)

  if not(in_filter == out_filter):
    orig_x = AveragePooling2D(pool_size=strides, strides=strides, 
                              padding='valid')(orig_x)
    # keras only support padding 1st and 2nd dimention of 3D tensor,
    # while we need to pad the 3rd dimension. To achieve our goal, 
    # we permute the 2nd and 3rd dimention, then pad.
    padding_amount = (out_filter - in_filter)//2
    orig_x = Permute((1,3,2))(orig_x)
    orig_x = ZeroPadding2D(padding=(0,padding_amount))(orig_x)
    orig_x = Permute((1,3,2))(orig_x)

  x = Add()([x, orig_x])
  return x

def ResNet():
  '''
  ResNet model ported from
  https://github.com/tensorflow/models/blob/master/resnet/resnet_model.py

  This model is also available at 
  https://github.com/MadryLab/cifar10_challenge/blob/master/model.py
  '''
  # Input tensor of the model.
  input_tensor = Input(shape=input_shape)

  # x will denote intermediate values in neural net computation.
  x = input_tensor

  # Initializers for each type of layers.
  # Convolutional layer is initialized by random_normal_initializer of 
  #tensorflow. The values for stddev parameter varies as 
  #stddev=numpy.sqrt(2.0/n) where n is the number of all weights on that layer
  #(For example, if we use 3x3 convolution filter, and there are 10 output 
  #filters, then n=3*3*10=90.
  # There are 4 parts in this network using convolutional layers as follows
  # Initial convolutional layer (conv_init)  : n_init=3*3*16
  # Convolutiona layers in 1st residual block: n_res1=3*3*160
  # Convolutiona layers in 2nd residual block: n_res2=3*3*320
  # Convolutiona layers in 3rd residual block: n_res3=3*3*640
  # TODO: Currently unsure if we need to modularize conv layers.
  #               Even though conv layers are modularized in the code
  #               referred. Think about this modularization issue.
  n_init = 3*3*16
  n_res1 = 3*3*160
  n_res2 = 3*3*320
  n_res3 = 3*3*640
  init_conv_init = RandomNormal(mean=0.0, stddev=numpy.sqrt(2.0/n_init))
  init_conv_res1 = RandomNormal(mean=0.0, stddev=numpy.sqrt(2.0/n_res1))
  init_conv_res2 = RandomNormal(mean=0.0, stddev=numpy.sqrt(2.0/n_res2))
  init_conv_res3 = RandomNormal(mean=0.0, stddev=numpy.sqrt(2.0/n_res3))
  # Fully connected layer is initialized by uniform_unit_scaling_initializer
  #(deprecated) of tensorflow, which is equivalent to variance_scaling with 
  #uniform distribution.
  init_fc = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')

  # Initial convolutional layer.
  x = Conv2D(16, (3,3), padding='same', use_bias=False,
      kernel_initializer=init_conv_init, input_shape=input_shape,
      name = 'conv_init')(x)

  # 1st residual block.
  # 1st residual block head.
  # Only this unit activate before residual, so was implemented separately.
  x = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3,
      center=True, scale=True, name = 'BN_1_0_1')(x)
  x = LeakyReLU(alpha=0.1)(x)

  orig_x = x
  orig_x = AveragePooling2D(pool_size=(1,1), strides=(1,1), padding='valid')(x)
  # keras only support padding 1st and 2nd dimention of 3D tensor,
  # while we need to pad the 3rd dimension. To achieve our goal, 
  # we permute the 2nd and 3rd dimention, then pad.
  padding_amount = (160 - 16)//2
  orig_x = Permute((1,3,2))(orig_x)
  orig_x = ZeroPadding2D(padding=(0,padding_amount))(orig_x)
  orig_x = Permute((1,3,2))(orig_x)

  x = Conv2D(160, (3,3), padding='same', use_bias=False,
      kernel_initializer=init_conv_init, input_shape=input_shape,
      name = 'conv_1_0_1')(x)
  x = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3,
      center=True, scale=True, name = 'BN_1_0_2')(x)
  x = LeakyReLU(alpha=0.1)(x)
  x = Conv2D(160, (3,3), padding='same', use_bias=False,
      kernel_initializer=init_conv_init, input_shape=input_shape,
      name = 'conv_1_0_2')(x)

  x = Add()([x, orig_x])

  # 1st residual block tail.
  for i in range(1,5):
    x = ResidualUnit(x, 160, 160, 1, init_conv_res1, 1, i)

  # 2nd residual block.
  # 2nd residual block head.
  x = ResidualUnit(x, 160, 320, 2, init_conv_res2, 2, 0)
  # 2nd residual block tail.
  for i in range(1,5):
    x = ResidualUnit(x, 320, 320, 1, init_conv_res2, 2, i)

  # 3rd residual block.
  # 3rd residual block head.
  x = ResidualUnit(x, 320, 640, 2, init_conv_res3, 3, 0)
  # 3rd residual block tail.
  for i in range(1,5):
    x = ResidualUnit(x, 640, 640, 1, init_conv_res3, 3, i)

  # Last pooling and mean function.
  x = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3,
      center=True, scale=True, name = 'BN_last')(x)
  x = LeakyReLU(alpha=0.1)(x)
  x = GlobalAveragePooling2D()(x)

  # Classification.
  x = Dense(10, name = 'logits')(x)
  x = Activation('softmax')(x)

  # Output tensor of the model.
  output_tensor = x

  # Create and Return the model
  model = Model(input_tensor, output_tensor)
  return model
