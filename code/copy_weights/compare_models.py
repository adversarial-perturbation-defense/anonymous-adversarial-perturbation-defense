import tensorflow
import numpy

def preprocessImage(image):
  '''Preprocess the image as it is done by MIT model.
  image: numpy array corresponding to the image
  '''
  input_image = tensorflow.placeholder(tensorflow.float32,
                                       shape = [None, 32, 32, 3])
  input_standardized = tensorflow.map_fn(
          lambda img: tensorflow.image.per_image_standardization(img),
          input_image)
  with tensorflow.Session() as sess:
    feed_dict = {input_image: image}
    processed_image = input_standardized.eval(feed_dict=feed_dict,session=sess)

  return processed_image

def predictMITModel(session, tf_model, x, image):
  '''Use MIT model(Madry et al.) to make a prediction on an image
  model: Tensor corresponding to the prediction
  x: Input placeholder
  image: numpy array corresponding to the image
  '''
  feed_dict = {x: image}
  return tf_model.eval(feed_dict=feed_dict, session=session)

def predictCopiedModel(keras_model, image):
  '''Use copied model to make a prediction on an image
  model: keras model object
  image: numpy array corresponding to the image
  '''
  return keras_model.predict(image)

def areSameOutputs(y1, y2, tol = 1e-6):
  '''Compare two different output vectors upto some tolerance
  NOTE: The default tolerance 1e-6 is the resolution of numpy.float32 datatype,
        so any value smaller than this can be considered as 0.
  '''
  return numpy.allclose(y1, y2, rtol = 0, atol = tol)

