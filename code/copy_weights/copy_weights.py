import numpy
import tensorflow
import keras

from models import ResNet

def getNextTrainable(layer_list):
  ''' Find the next layer that we can copy value.
  '''
  layer = next(layer_list)
  while len(layer.get_weights()) == 0:
    layer = next(layer_list)

  return layer

def copyWeights(keras_layer, tf_var_pref, tf_var_list, session):
  '''
  Copy layer weights.
  keras_layer: Keras layer object
  tf_var_pref: Prefix of tensorflow variables
  tf_var_list: List of tensorflow variables,
               ordering follows that of keras layer
  session: Tensorflow session which will evaluate tensorflow variables.
  '''
  print('Copying ' + keras_layer.name)

  weights_list = []

  for tf_var in tf_var_list:
    var_scope = tf_var_pref + tf_var
    v = tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES,
                                  scope = var_scope)
    assert len(v) > 0
    
    v = v[0]

    weights = session.run(v)
    weights_list.append(weights)

  keras_layer.set_weights(weights_list)

def copyResidualUnit(layer_list, block_idx, unit_idx, session):
  '''
  Copy one residual unit.
  This function is only applicable to Madry et al. model.
  
  layer_list: List of layers to search the next layer to be copied.
  block_idx: Index of residual block.
  unit_idx: Index of residual unit.
  '''
  layer = getNextTrainable(layer_list)
  # The next layer is the 1st BN layer of the residual unit 1_1.
  copyWeights(layer,
              'unit_{0}_{1}/residual_only_activation/BatchNorm/'.format(
                                                          block_idx, unit_idx),
              ['gamma:0','beta:0','moving_mean:0','moving_variance:0'], session)

  layer = getNextTrainable(layer_list)
  # The next layer is the 1st conv layer of the residual unit 1_1.
  copyWeights(layer, 'unit_{0}_{1}/sub1/conv1/'.format(block_idx, unit_idx),
              ['DW:0'], session)

  layer = getNextTrainable(layer_list)
  # The next layer is the 2nd BN layer of the residual unit 1_1.
  copyWeights(layer, 'unit_{0}_{1}/sub2/BatchNorm/'.format(block_idx, unit_idx),
              ['gamma:0','beta:0','moving_mean:0','moving_variance:0'], session)

  layer = getNextTrainable(layer_list)
  # The next layer is the 2nd conv layer of the residual unit 1_1.
  copyWeights(layer, 'unit_{0}_{1}/sub2/conv2/'.format(block_idx, unit_idx),
              ['DW:0'], session)

def copiedNaturallyTrainedModel():
  '''
  Return a ResNet model whose weights are copied from Madry et al. model.
  This function is only applicable to Madry et al. model.
  '''
  # Tensorflow variables whose weights will be copied from.
  sess = tensorflow.Session()

  checkpoint_path = './models/naturally_trained/checkpoint-70000'
  saver = tensorflow.train.import_meta_graph(checkpoint_path + '.meta')
  _ = saver.restore(sess, checkpoint_path)

  # Keras model whose weights will be copied to
  model = ResNet()

  # List of layer objects (iterable)
  layer_list = iter(model.layers)

  ## Find the next layer that we can copy value.
  #layer = next(layer_list)
  #while len(layer.get_weights()) == 0:
  #  layer = next(layer_list)
  layer = getNextTrainable(layer_list)

  # The first layer to copy should be a convolutional layer.
  copyWeights(layer, 'input/init_conv/', ['DW:0'], sess)

  layer = getNextTrainable(layer_list)
  # The next layer is the 1st BN layer of the residual unit 1_0.
  copyWeights(layer, 'unit_1_0/shared_activation/BatchNorm/', 
              ['gamma:0','beta:0','moving_mean:0','moving_variance:0'], sess)

  layer = getNextTrainable(layer_list)
  # The next layer is the 1st conv layer of the residual unit 1_0.
  copyWeights(layer, 'unit_1_0/sub1/conv1/', ['DW:0'], sess)

  layer = getNextTrainable(layer_list)
  # The next layer is the 2nd BN layer of the residual unit 1_0.
  copyWeights(layer, 'unit_1_0/sub2/BatchNorm/', 
              ['gamma:0','beta:0','moving_mean:0','moving_variance:0'], sess)

  layer = getNextTrainable(layer_list)
  # The next layer is the 2nd conv layer of the residual unit 1_0.
  copyWeights(layer, 'unit_1_0/sub2/conv2/', ['DW:0'], sess)

  for j in range(1, 5):
    copyResidualUnit(layer_list, 1, j, sess)

  for i in range(2,4):
    for j in range(0,5):
      copyResidualUnit(layer_list, i, j, sess)

  layer = getNextTrainable(layer_list)
  copyWeights(layer, 'unit_last/BatchNorm/',
              ['gamma:0','beta:0','moving_mean:0','moving_variance:0'], sess)

  layer = getNextTrainable(layer_list)
  copyWeights(layer, 'logit/', ['DW:0','biases:0'], sess)

  return model

def copiedAdversariallyTrainedModel():
  '''
  Return a ResNet model whose weights are copied from Madry et al. model.
  This function is only applicable to Madry et al. model.
  '''
  # Tensorflow variables whose weights will be copied from.
  sess = tensorflow.Session()

  checkpoint_path = './models/adv_trained/checkpoint-70000'
  saver = tensorflow.train.import_meta_graph(checkpoint_path + '.meta')
  _ = saver.restore(sess, checkpoint_path)

  # Keras model whose weights will be copied to
  model = ResNet()

  # List of layer objects (iterable)
  layer_list = iter(model.layers)

  ## Find the next layer that we can copy value.
  #layer = next(layer_list)
  #while len(layer.get_weights()) == 0:
  #  layer = next(layer_list)
  layer = getNextTrainable(layer_list)

  # The first layer to copy should be a convolutional layer.
  copyWeights(layer, 'input/init_conv/', ['DW:0'], sess)

  layer = getNextTrainable(layer_list)
  # The next layer is the 1st BN layer of the residual unit 1_0.
  copyWeights(layer, 'unit_1_0/shared_activation/BatchNorm/', 
              ['gamma:0','beta:0','moving_mean:0','moving_variance:0'], sess)

  layer = getNextTrainable(layer_list)
  # The next layer is the 1st conv layer of the residual unit 1_0.
  copyWeights(layer, 'unit_1_0/sub1/conv1/', ['DW:0'], sess)

  layer = getNextTrainable(layer_list)
  # The next layer is the 2nd BN layer of the residual unit 1_0.
  copyWeights(layer, 'unit_1_0/sub2/BatchNorm/', 
              ['gamma:0','beta:0','moving_mean:0','moving_variance:0'], sess)

  layer = getNextTrainable(layer_list)
  # The next layer is the 2nd conv layer of the residual unit 1_0.
  copyWeights(layer, 'unit_1_0/sub2/conv2/', ['DW:0'], sess)

  for j in range(1, 5):
    copyResidualUnit(layer_list, 1, j, sess)

  for i in range(2,4):
    for j in range(0,5):
      copyResidualUnit(layer_list, i, j, sess)

  layer = getNextTrainable(layer_list)
  copyWeights(layer, 'unit_last/BatchNorm/',
              ['gamma:0','beta:0','moving_mean:0','moving_variance:0'], sess)

  layer = getNextTrainable(layer_list)
  copyWeights(layer, 'logit/', ['DW:0','biases:0'], sess)

  return model
