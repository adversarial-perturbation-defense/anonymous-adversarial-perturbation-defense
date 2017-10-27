import keras
import tensorflow
import numpy

import model

from data_utils import LoadCIFAR10
from data_utils import GetDataPoint
from train_utils import load_model

from compare_models import preprocessImage
from compare_models import predictMITModel
from compare_models import predictCopiedModel
from compare_models import areSameOutputs

print("Start testing the copied version of naturally trained MIT model.")
# Load and pick a data point.
(x_train, y_train), (x_test, y_test) = LoadCIFAR10()

# MIT model file.
model_file = tensorflow.train.latest_checkpoint('models/naturally_trained')
assert not(model_file is None)

# Load the MIT model architecture that we copied.
mit_model = model.Model(mode = 'eval')

# We are only interested in the output tensor of MIT model.
mit_model_input = mit_model.x_input
mit_model_output = mit_model.pre_softmax

# Load the saved weight of MIT model.
saver = tensorflow.train.Saver()
sess = tensorflow.Session()
saver.restore(sess, model_file)

# Load copied model which predicts "after softmax".
copied_model = load_model('./saved_models/nat_trained_mit_model')

# MIT model only predicts "before softmax" (does not have softmax layer).
# Therefore we take the part of the copied model before the softmax.
copied_model_before_softmax = keras.models.Model(inputs = copied_model.input, 
              outputs = copied_model.get_layer('logits').output)

print('<<Summary over TEST SET>>')
norm_array = numpy.asarray([])
for idx in range(100):
  #test_image, _ = GetDataPoint(x_test, y_test, idx)
  start_idx = 100 * idx
  end_idx = start_idx + 100
  test_batch = x_test[start_idx:end_idx]

  # MIT model uses preprocessed image (normalization)
  processed_test_batch = preprocessImage(test_batch)

  # Testing for mit model prediction.
  y1 = predictMITModel(sess, mit_model_output, mit_model_input, test_batch)

  # Testing for copied model prediction.
  y2 = predictCopiedModel(copied_model_before_softmax, processed_test_batch)

  norm_array_per_batch = numpy.linalg.norm(y1 - y2, ord=numpy.inf, axis=-1)
  norm_array = numpy.append(norm_array, norm_array_per_batch) 
  if areSameOutputs(y1, y2, 5e-4):
    print 'Same from {0} to {1}'.format(start_idx, end_idx-1)
  else:
    print 'Different somewhere from {0} to {1}'.format(start_idx, end_idx-1)

print('Mean    inf norm diff (TEST SET): {0}'.format(numpy.mean(norm_array)))
print('Std dev inf norm diff (TEST SET): {0}'.format(numpy.std(norm_array)))

print()
print('<<Summary over TRAIN SET>>')
norm_array = numpy.asarray([])
for idx in range(500):
  #test_image, _ = GetDataPoint(x_test, y_test, idx)
  start_idx = 100 * idx
  end_idx = start_idx + 100
  test_batch = x_train[start_idx:end_idx]

  # MIT model uses preprocessed image (normalization)
  processed_test_batch = preprocessImage(test_batch)

  # Testing for mit model prediction.
  y1 = predictMITModel(sess, mit_model_output, mit_model_input, test_batch)

  # Testing for copied model prediction.
  y2 = predictCopiedModel(copied_model_before_softmax, processed_test_batch)

  norm_array_per_batch = numpy.linalg.norm(y1 - y2, ord=numpy.inf, axis=-1)
  norm_array = numpy.append(norm_array, norm_array_per_batch) 
  if areSameOutputs(y1, y2, 5e-4):
    print 'Same from {0} to {1}'.format(start_idx, end_idx-1)
  else:
    print 'Different somewhere from {0} to {1}'.format(start_idx, end_idx-1)

print('Mean    inf norm diff (TEST SET): {0}'.format(numpy.mean(norm_array)))
print('Std dev inf norm diff (TEST SET): {0}'.format(numpy.std(norm_array)))

sess.close()
