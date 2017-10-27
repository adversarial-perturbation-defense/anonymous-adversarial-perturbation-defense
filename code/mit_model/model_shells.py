"""Shells wrapping around models.

A model shell is a model, but wrapped with a shell we define. The 'shell'
provides many different benefits. Such as:
(1) Provide the basis for Carlini Li attacks,
(2) Overrided predict interface that enhances the robustness of the original
     model predict().
(3) Uniform interfaces for prediction.
"""
import numpy
import tensorflow
import glog as logging
from keras import backend
from keras.models import Model
from li_attack import CarliniLi


# Some helper functions.
def CreateOneHot(size, index):
  """Create a one-hot vector.

  More specifically we create a numpy array of shape (1, size).

  Args:
    size: Number of columns.
    index: Which column is set to 1.0

  Returns:
    A numpy array of shape (1, size).
  """
  v = numpy.zeros(size)
  v[index] = 1.0
  return numpy.array([v])


# Modified from Carlini to use the model we trained.
#
# TODO: session is really never used; maybe important later.
# TODO: What about models for other datasets? (can be delayed)
class CIFARModelShell(object):
  """A model shell for CIFAR datasets.

  """

  def __init__(self, F, session=None):
    # Constants that will be used in li_attack.py.
    self.num_channels = 3
    self.image_size = 32
    self.num_labels = 10

    # Store the original model F.
    self.model = F

    # Get a model without the last softmax layer, required by Carlini.
    # TODO: We hard wired the name of layer to stop, namely 'logits'.
    layer_name = 'logits'
    self.model_before_softmax = Model(
      inputs=F.input, outputs=F.get_layer(layer_name).output)

    self.session = session

    logging.info('self.model=%s, num_labels=%s', self.model, self.num_labels)

  def predict(self, x):
    """Compute the actual prediction.

    By default we return the prediction of self.model -- what we wrapped over.
    However we can do whatever based on self.model.

    Args:
      data: Data for doing prediction.

    Returns:
      A softmax prediction.
    """
    # The following lines are only for MIT model.
    x_std = tensorflow.map_fn(
            lambda img: tensorflow.image.per_image_standardization(img),
            x)

    # TODO: Can we unify these two different prediction methods?
    if backend.learning_phase() == 0:
      return backend.eval(self.model(x_std))
    else:
      return self.model.predict(backend.eval(x_std))

  def symbolic_predict(self, x):
    """Returns a symbolic experssion of prediction.

    This symbolic prediction is given by model_before_softmax,
    which discared the softmax layer. This is needed by the
    Carlini li attack. See li_attack.py for code such as:

    output = model.symbolic_predict(newimg).

    Args:
      x: The data to create the prediction function.

    Returns:
      A new tensorflow expression for result before softmax.
    # The following lines are only for MIT model.
    """
    # The following lines are only for MIT model.
    x_std = tensorflow.map_fn(
            lambda img: tensorflow.image.per_image_standardization(img),
            x)

    return self.model_before_softmax(x_std)


class CarliniLiModelShell(CIFARModelShell):
  """An example model shell using CarliniLiAttack.

  We try to use CarliniLiAttack to defend adversarial perturbation.
  """

  def __init__(self, F, session=None):
    CIFARModelShell.__init__(self, F, session)

  def predict(self, x):
    """Predict at point x.

    Our method is the following: For class i=0, 1, 2, ... num_classes-1, we use
    CarliniLiAttack to find an adversarial example xi. This induces a softmax
    prediction where class i stands out say w.p. pi. Then we pick label i that
    has the largest pi -- i.e. we pick the most confident adversarial example.

    Args:
      x: A point to predict.

    Returns:
      The softmax vector computed for prediction.
    """

    logging.info('------ CarliniLiModelShell predict ------')

    if self.model is None:
      logging.info('Cannot have None model for prediction, abort...')
      assert False

    backend.set_learning_phase(0)
    ## For slower, stronger attack (Default CW const_factor)
    #CarliniAttack = CarliniLi(self.session, self, initial_tau = 1/32.)
    # For faster, weaker attack
    CarliniAttack = CarliniLi(self.session, self, const_factor = 10000.0,
                              fix_tau = True, initial_tau = 1/32.)

    # Prediction of self.model now.

    # The following lines are only for MIT model.
    x_std = tensorflow.map_fn(
            lambda img: tensorflow.image.per_image_standardization(img),
            x)

    softmax_x = backend.eval(self.model(x_std))
    prediction_x = numpy.argmax(softmax_x)
    confidence_x = numpy.amax(softmax_x)
    logging.info('self.model in the model shell gives: '
                 'softmax_at_x: %s, prediction_x: %s, confidence_x: %s',
                 softmax_x, prediction_x, confidence_x)

    # For each label that is not prediction_x, do a targeted attack.
    label_scores_list = [[prediction_x, confidence_x]]
    for label in range(self.num_labels):
      logging.info('CarliniAttack for label %s', label)
      if label == prediction_x:
        logging.info('Skip CarliniAttack for label %s as it is the current '
                     'prediction of self.model we wrapped over.', label)
        continue
      # Construct a one-hot vector for the targeted label.
      label_v = CreateOneHot(self.num_labels, label)
      xp = CarliniAttack.attack(x, label_v, verbose=False)

      # The following lines are only for MIT model.
      xp_std = tensorflow.map_fn(
              lambda img: tensorflow.image.per_image_standardization(img),
              xp)

      # Compute the prediction at x'.
      softmax_xp = backend.eval(self.model(xp_std))
      pred_xp = numpy.argmax(softmax_xp)
      confidence_xp = numpy.amax(softmax_xp)
      logging.info('pred_xp: %s with confidence: %s', pred_xp, confidence_xp)

      # Nearby x we find another point where we predict it is LABEL
      # with confidence CONFIDENCE.
      label_scores_list.append([pred_xp, confidence_xp])
      
    label_scores_list.sort(key=lambda x: x[1], reverse=True)
    logging.info('label_scores_list: %s', label_scores_list)

    prediction = label_scores_list[0][0]
    logging.info('Final prediction is: %s', prediction)
    logging.info('-----------------------------------------')
    return CreateOneHot(self.num_labels, prediction)
