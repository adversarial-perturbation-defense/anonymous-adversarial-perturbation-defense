# Attacks will be merged to "attacks.py" (if it will be created later)
import random
import numpy
import glog as logging
from keras import backend
from keras.utils.np_utils import to_categorical
from li_attack import CarliniLi
from model_shells import CIFARModelShell

def CarliniLiAttack(model_shell, x, y):
  """Attack a model shell at point x using Carlini inf norm attack.

  Args:
    Model: A model shell to attack.
    x: Point to be perturbed
    y: Label, depending on if the attack is targeted:
       If targeted, it should be the one-hot vector of target label.
       If not, it should be the one-hot vector of predicted label.

  Returns:
    Return the value from Carlini's attack.
  """
  #session = backend.get_session()
  #model = CIFARModel(F, session=session)

  # Change the learning phase in order to perform attack.
  backend.set_learning_phase(0)

  # Create the object for performing Carlini's inf norm attack.
  # More parameters can be used to.
  #
  # TODO: This interface is only for non-targeted attack?
  # If so rename to make this more clear.
  CarliniAttack = CarliniLi(model_shell.session, model_shell, targeted=False)

  # Return the value returned by Carlini's attack.
  return CarliniAttack.attack(x, y, verbose=False)


# Whatever attack method we use, and whatever dataset we use,
# functions below should work independently of them in the future
# But now, we assume Carlini inf norm attack on CIFAR10 dataset for convenience.

def TestCorrectness(model_shell, x, y):
  """Test correctness of a model F at point x: Check if F(x) == y.

  Args:
    model_shell: Model shell to test correctness.
    x: Point at which we test F's correctness.
    y: Correct label (as one-hot vector).

  Returns:
    True if correct, otherwise False.
  """
  logging.info('TestCorrectness: Test one-hot vector is %s', y)
  # Extract correct label from y.
  correct_label = numpy.argmax(y)
  logging.info('The correct label is %s', correct_label)

  # Compute softmax, prediction, and confidence at x.
  softmax = model_shell.predict(x)
  pred = numpy.argmax(softmax)
  confidence = numpy.amax(softmax)
  logging.info('The softmax vector: %s', softmax)
  logging.info('The predicted label is %s, with confidence %s',
               pred, confidence)
  is_correct = (pred == correct_label)
  if is_correct:
      logging.info('The model is correct at the given point')
  else:
      logging.info('The model is NOT correct at the given point')

  return is_correct

# TODO: Support other attack methods (can be delayed a bit).

def TestRobustness(model_shell, x, eps):
  """Test robustness of a model F at point x.

  It performs Carlini's inf norm attack to generate x', and then checks
  if "F(x) == F(x')"

  Args:
    F: Model to be checked.
    x: Point at which we test F's robustness
    eps: Radius of norm constraint

  Returns:
    True if robust, otherwise False, and the perturbed image.
  """
  logging.info('TestRobustness with norm constraint eps=%s.', eps)

  # Compute the prediction of model_shell at x.
  logging.info('Compute model shell prediction at x...')
  softmax_x = model_shell.predict(x)
  pred_x = numpy.argmax(softmax_x)
  confidence_of_pred_x = numpy.amax(softmax_x)

  # Generate perturbed image x' (x prime, or x_perturbed).
  # The 2nd input of Carlini's untargeted attack is the vector representing
  # the label whose confidence should be decreased In our case, it is the
  # *one-hot vector* of predicted label.
  logging.info('Generating perturbed image...')
  pred_x_categorical = to_categorical(pred_x, model_shell.num_labels)
  xp = CarliniLiAttack(model_shell, x, pred_x_categorical)

  # Compute the inf norm of perturbation.
  norm = numpy.max(numpy.abs(xp - x))
  logging.info('Perturbation size (inf norm): %s', norm)
  logging.info('Checking robustness...')

  # Compute the prediction at x'
  # softmax_xp = backend.eval(F(backend.constant(xp)))
  logging.info('Compute model shell prediction at xp...')
  softmax_xp = model_shell.predict(xp)
  pred_xp = numpy.argmax(softmax_xp)
  confidence_of_pred_xp = numpy.amax(softmax_xp)

  # NOTE: There are two cases that we are robust:
  # (1) The norm bound is beyond eps specified.
  # (2) The norm bound of the attack found is within eps but the predictions
  #     are in fact consistent.
  is_robust = False
  if norm > eps:
    logging.info(
      'Norm of the attack (%s) exceeds norm constraint (%s) ', norm, eps)
    is_robust = True
  else:
    logging.info(
      'Norm of the attack (%s) is within constraint (%s) ', norm, eps)
    if pred_x == pred_xp:
      logging.info('pred_x [%s] (confidence: %s) '
                   '== pred_xp [%s] (confidence %s): Invalid attack',
                   pred_x, confidence_of_pred_x, 
                   pred_xp, confidence_of_pred_xp)
      is_robust = True
    else:
      logging.info('pred_x [%s] (confidence: %s) '
                   '!= pred_xp [%s] (confidence %s): Valid Attack',
                   pred_x, confidence_of_pred_x,
                   pred_xp, confidence_of_pred_xp)

  if is_robust:
    logging.info('The model is robust at the given point.')
  else:
    logging.info('The model is NOT robust at the given point.')

  # Return robustness predicate and the perturbed image.
  return is_robust, xp

def TestCorrectnessAndRobustness(model_shell, x, y, eps,
                                 test_robustness = False, xp = None):
  """Test both correctness and robustness of a model F at point x
  Correctness: Check if F(x) == y.
  Robustness: Check if F(x) == F(x') after generating perturbed image x'.
  Carlini's inf norm attack is used to generate x'.

  Args:
    model_shell: Model shell to test correctness.
    x: Point at which we test F's correctness.
    y: Correct label (as one-hot vector).
    eps: Radius of norm constraint.
    test_robustness: Whether to test robustness at all.
    xp: Adversarial perturbation to use (if exists).

  Returns:
    True if correct, otherwise False.
  """
  logging.info('TestCorrectness: Test one-hot vector is %s', y)
  # Extract correct label from y.
  correct_label = numpy.argmax(y)
  logging.info('The correct label is %s', correct_label)

  # Compute softmax, prediction, and confidence at x.
  logging.info('Compute model shell prediction at x...')
  softmax_x = model_shell.predict(x)
  pred_x = numpy.argmax(softmax_x)
  confidence_of_pred_x = numpy.amax(softmax_x)
  logging.info('The softmax vector: %s', softmax_x)
  logging.info('The predicted label is %s, with confidence %s',
               pred_x, confidence_of_pred_x)
  is_correct = (pred_x == correct_label)
  if is_correct:
      logging.info('The model is correct at the given point')
  else:
      logging.info('The model is NOT correct at the given point')

  # Skip the robustness test if we want.
  if test_robustness == False:
    return is_correct, None, None

  logging.info('TestRobustness with norm constraint eps=%s.', eps)

  # Generate perturbed image x' (x prime, or x_perturbed).
  # The 2nd input of Carlini's untargeted attack is the vector representing
  # the label whose confidence should be decreased In our case, it is the
  # *one-hot vector* of predicted label.
  # If there already exists an adversarial perturbation, skip the attack.
  if xp is None:
    logging.info('Generating perturbed image...')
    pred_x_categorical = to_categorical(pred_x, model_shell.num_labels)
    xp = CarliniLiAttack(model_shell, x, pred_x_categorical)
  else:
    logging.info('There already is a perturbation. Skipping...')

  assert (xp is not None)
  # Compute the inf norm of perturbation.
  norm = numpy.max(numpy.abs(xp - x))
  logging.info('Perturbation size (inf norm): %s', norm)
  logging.info('Checking robustness...')

  # Compute the prediction at x'
  # softmax_xp = backend.eval(F(backend.constant(xp)))
  logging.info('Compute model shell prediction at xp...')
  softmax_xp = model_shell.predict(xp)
  pred_xp = numpy.argmax(softmax_xp)
  confidence_of_pred_xp = numpy.amax(softmax_xp)

  # NOTE: There are two cases that we are robust:
  # (1) The norm bound is beyond eps specified.
  # (2) The norm bound of the attack found is within eps but the predictions
  #     are in fact consistent.
  is_robust = False
  if norm > eps:
    logging.info(
      'Norm of the attack (%s) exceeds norm constraint (%s) ', norm, eps)
    is_robust = True
  else:
    logging.info(
      'Norm of the attack (%s) is within constraint (%s) ', norm, eps)
    if pred_x == pred_xp:
      logging.info('pred_x [%s] (confidence: %s) '
                   '== pred_xp [%s] (confidence %s): Invalid attack',
                   pred_x, confidence_of_pred_x, 
                   pred_xp, confidence_of_pred_xp)
      is_robust = True
    else:
      logging.info('pred_x [%s] (confidence: %s) '
                   '!= pred_xp [%s] (confidence %s): Valid Attack',
                   pred_x, confidence_of_pred_x,
                   pred_xp, confidence_of_pred_xp)

  if is_robust:
    logging.info('The model is robust at the given point.')
  else:
    logging.info('The model is NOT robust at the given point.')

  # Return robustness predicate and the perturbed image.
  return is_correct, is_robust, xp

def FindOneVulnerablePoint(model_shell, X, Y, eps):
  """Find a point of our interest (call it vulnerable point).
  This point of interest should be a point x such that,
  1. the model is correct on x
  2. the model is not correct on x and given radius eps.

  X: Set of samples where x lies in
  Y: Set of labels for correctness test.
  eps: Radius of ball defining an unperceivable perturbation.

  Return: the index of vulnerable point x in X, and adversarial example xp.
  """

  num_point = X.shape[0]
  logging.info('There are %d points to be checked', num_point)

  # Run through a shuffled list of indices.
  for index in random.sample(range(num_point), num_point):
    logging.info('Checking sample %d', index)
    x = X[index:index+1]
    y = Y[index:index+1]

    is_correct = TestCorrectness(model_shell, x, y)
    if not is_correct:
      logging.info('The model is NOT correct on this point. Skipping...')
      continue
    is_robust, xp = TestRobustness(model_shell, x, eps)

    if is_robust:
      logging.info('The model is robust on this point. Skipping...')
      continue

    logging.info('Sample %d is the point that the model is vulnerable.', index)
    return x, y, xp

  return None, None, None
