"""Controller of training, testing and tuning a model of certain type."""
import numpy
import tensorflow
import keras
from keras import backend
from keras.preprocessing.image import ImageDataGenerator
import glog as logging
from data_utils import GetDataPoint
from losses import nilLoss
from losses import regularizationLossMetric
from losses import totalLoss
from models import ConvSmall
from model_shells import CIFARModelShell
from model_shells import CarliniLiModelShell
from model_shells import AcceleratedCarliniLiModelShell
from test_utils import FindOneVulnerablePoint
from test_utils import TestCorrectness
from test_utils import TestRobustness
from test_utils import TestCorrectnessAndRobustness
from train_utils import get_scheduler
from train_utils import load_model
from train_utils import save_model

class ModelController(object):
  """Controller of various actions on a model."""

  def __init__(self, model_name=None, model_save_directory=None):
    """Initialize a model controller.

    Args:
      model_name: Name of the model to control.
      model_save_directory: Where to save the model named model_name.
    """

    self.model_name = model_name
    self.model_save_directory = model_save_directory
    self.model_shell = None
    self.save_path = None
    # The point and label to be tested.
    # NOTE: These are only required when the flag is set to find a
    #               vulnerable point
    self.vul_x = None
    self.vul_y = None
    # Adversarial example of the test point.
    self.vul_xp = None

    # Stronger model shell when model shell failed.
    # Used only for the experiments.
    self.model_shell_stronger = None

  def Train(self, x_train, y_train, x_test, y_test, training=False):
    """Train a model.

    Args:
      training: Whether to do training after all!
    """

    pass

  def Tune(self, x_train, y_train, x_test, y_test, x, eps,
           starting_model_save_path=None, tuning=False):
    """Tune a model at test time.

    Logically, this method should be called in Test(), to mimic the
    effect that we "tune" a model in the test time.

    Args:
      x_train: Training features.
      y_train: Training labels.
      x_test: Test features.
      y_test: Test labels.
      x: The data point to do tuning.
      eps: Norm bound we want to enforce around x.
      starting_model_save_path: Save path to load the model to start with.
      tuning: Whether to do tuning after all!
    """

    pass


  def FindVulnerablePoint(self, find_vul_point=False, 
                          X=None, Y=None, eps=None,
                          starting_model_save_path=None):
    # Set self.model_shell.
    # TODO: Find a better point to set model shell.
    if self.model_shell is None:
      # Load the previously saved model from save path.
      logging.info(
        'Load from: \'%s\' and convert to a model shell',
                                          starting_model_save_path)
      self.model_shell = CIFARModelShell(
        load_model(starting_model_save_path), backend.get_session())
    assert self.model_shell is not None

    # Find a vulnerable point.
    if find_vul_point:
      logging.info('Find a new point, the given point will not be used.')
      self.vul_x, self.vul_y, self.vul_xp = FindOneVulnerablePoint(
                                                    self.model_shell, X, Y, eps)
      if self.vul_x is None:
        logging.info('There is no vulnerable point of this model. Abort')
        assert False
    else:
      logging.info('Don\'t find a new point, use the given point.')

    # Reset self.model_shell for later use.
    self.model_shell=None

  def EnterTestStage(self):
    """Tell controller to enter the test stage.

    In Train and Tune we make a ready for testing. This function does some
    preparation for testing. Currently:

    1. Load the model from save_path to model.
       Our Train and Tune will save model to self.save_path.
       We load it to self.model.
    """

    logging.info('ModelController: Enter test stage.')
    # Set self.model_shell.
    if self.model_shell is None:
      # Load the previously saved model from save path.
      logging.info(
        'Load from: \'%s\' and convert to a model shell', self.save_path)
      self.model_shell = CIFARModelShell(
        load_model(self.save_path), backend.get_session())
    else:
      logging.info('self.model_shell is already set.')
    logging.info('Done setting self.model_shell.')

    # TODO: We cannot set learning phase to 0 early as we cannot call
    # F.predict otherwise.
    # backend.set_learning_phase(0)

    logging.info('ModelController: We are ready for testing.')

  # TODO: The following interface only supports testing correctness
  # and robustness at a SINGLE point. Support further:
  # 1. Testing (correctness and robustness) a batch, and
  # 2. Perhaps test generalization only.
  def Test(self, x, y, eps):
    """Test both correctness and robustness.

    Args:
      x: The feature vector.
      y: The *correct* labeling of x.
      eps: The norm bound for a valid attack.
    """

    logging.info('--- ModelController: Test ---')

    if self.model_shell is None:
      logging.info('Model shell is None: Call EnterTestStage to prepare.')
      assert False
    # If there is a vulnerable point found before, replace the point
    if self.vul_x is not None:
      logging.info('There is a vulnerable point found before. Reset...')
      x = self.vul_x
      y = self.vul_y

    is_correct, is_robust, _ = self._TestCorrectnessAndRobustness(
                                                   self.model_shell, x, y, eps)
    if (self.model_shell_stronger is not None) and not(is_correct and is_robust):
      self._TestCorrectnessAndRobustness(self.model_shell_stronger, x, y, eps)

  def TestTransfer(self, x, y, eps, ref_model_save_path):
    """Test transfer attack.

    Given a ref model loaded from ref_model_save_path, we do the following:
    1. Test robustness of the ref_model, and get a perturbed image.
    2. Test the perturbed image on our model.

    Args:
      x: A feature vector.
      y: The *correct* label of x.
      ref_model_save_path: The reference model where the attack is
        transferred from.
    """

    logging.info('--- ModelController: TestTransfer ---')

    # Give up if we have not model shell to test...
    if self.model_shell is None:
      logging.info('Model shell is None: Call EnterTestStage to prepare.')
      assert False

    # Give up if we have no reference model.
    if ref_model_save_path is None:
      logging.info('Must specify ref model path for transfer attack, return...')
      return

    # If there is a vulnerable point found before, replace the point
    if self.vul_x is not None:
      logging.info('There is a vulnerable point found before. Reset...')
      x = self.vul_x
      y = self.vul_y

    # Step 1: Test the reference model.
    # Load the reference model.
    logging.info('----- TestTransfer: Test reference model -----')
    logging.info('TestTransfer: Load the reference model from \'%s\' '
                 'and convert to a model shell.',
                 ref_model_save_path)
    ref_model_shell = CIFARModelShell(
      load_model(ref_model_save_path), backend.get_session())
    # Record the perturbed image found by the attack for doing transfer test.
    _, _, xp = self._TestCorrectnessAndRobustness(ref_model_shell, x, y, eps)
    logging.info('----- TestTransfer: DONE testing reference model -----')

    # Step 2: Test the self.model: Are we correct on xp? Note that this is
    # nothing but a correctness test!
    logging.info('----- TestTransfer: Is self.model_shell robust at xp? -----')
    is_correct = TestCorrectness(self.model_shell, xp, y)
    logging.info('self.model_shell is robust at xp: %s', is_correct)
    logging.info('-----------------------------------------------------')

  # ##########################
  # Internal/private methods.
  # ##########################
  def _TestCorrectnessAndRobustness(self, model_shell, x, y, eps):
    """Test both correctness and robustness of a given model.

    This is supposed to be an internal interface that is not visable outside.

    Args:
      model_shell: The model shell to test.
      x: Point to test.
      y: The correct label.
      eps: Norm bound of a valid attack.
    """

    # Check robustness F(x)==F(x').
    is_correct, is_robust, xp = TestCorrectnessAndRobustness(model_shell, x, y,
                                     eps, test_robustness=True, xp=self.vul_xp)

    # Print a summary of the test.
    logging.info('Test summary: is_correct=%s, is_robust=%s',
                 is_correct, is_robust)

    return is_correct, is_robust, xp


class NilModelController(ModelController):
  """Controller for a nil model."""

  DEFAULT_MODEL_SAVE_DIRECTORY = (
    '/nobackup-15T/transductive-defense-of-adversarial-perturbation/'
    'saved_models/null/')
  DEFAULT_MODEL_NAME = 'NN-null'
  DEFAULT_NUM_EPOCHS = 120

  def __init__(self, model_name=None, model_save_directory=None,
               num_epochs=None):
    # Initialize super.
    ModelController.__init__(self, model_name, model_save_directory)

    # Constants used in training.
    self.tr_batch_size = 32
    self.val_batch_size = 100

    self.num_epochs = num_epochs
    self.decay_start_epoch = 80
    self.num_iters_per_epoch = 400

    # TODO: We use a fixed learning rate, is this really what we want?
    self.learning_rate = 0.0001

    self.beta_1_before_decay = 0.9
    self.beta_1_after_decay = 0.5

    if self.model_name is None:
      logging.info('No model_name given, fall back to default \'%s\'.',
                   NilModelController.DEFAULT_MODEL_NAME)
      self.model_name = NilModelController.DEFAULT_MODEL_NAME

    if self.model_save_directory is None:
      logging.info('No model_save_directory given, fall back to default \'%s\'',
                   NilModelController.DEFAULT_MODEL_SAVE_DIRECTORY)
      self.model_save_directory = (
        NilModelController.DEFAULT_MODEL_SAVE_DIRECTORY)

    if self.num_epochs is None:
      logging.info('No num_epochs given, fall back to default %s.',
                   NilModelController.DEFAULT_NUM_EPOCHS)
      self.num_epochs = NilModelController.DEFAULT_NUM_EPOCHS

    # Note that null models are saved under null subdirectory.
    self.save_path = '{}{}'.format(self.model_save_directory, self.model_name)
    logging.info('Model save path: %s', self.save_path)

  def Train(self, x_train, y_train, x_test, y_test, training=False):
    """Train the model.

    Args:
      training: Whether to do training after all!
    """

    logging.info('--- NilModelController training ---')
    # TODO: Better check whether model actually exists, if not, retrain.
    if not training:
      logging.info('Skip training... Supposedly the model has been '
                   'trained and saved.')
      return

    # Use the ConvSmall architecture in Miyato et al. paper.
    # TODO: Parameterize this to train different architectures?
    model = ConvSmall()

    # Define optimizer to be used.
    opt = keras.optimizers.adam(lr=self.learning_rate,
                                beta_1=self.beta_1_before_decay)

    # Compile the model with optimizer and loss.
    model.compile(loss=nilLoss, optimizer=opt, metrics=['accuracy', nilLoss])

    # Data generators used in data augmentation (we may use it if needed).
    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    # Scheduler controls learning rate decay and momentum.
    scheduler = get_scheduler(num_epochs=self.num_epochs,
                              decay_start_epoch=self.decay_start_epoch,
                              learning_rate=self.learning_rate,
                              beta_1_before_decay=self.beta_1_before_decay,
                              beta_1_after_decay=self.beta_1_after_decay)

    # Train the model.
    logging.info('Training starts.')
    with tensorflow.device('/gpu:0'):
      model.fit_generator(datagen.flow(x_train, y_train,
                                       batch_size=self.tr_batch_size),
                          steps_per_epoch=self.num_iters_per_epoch,
                          epochs=self.num_epochs,
                          callbacks=[scheduler],
                          validation_data=datagen.flow(
                              x_test, y_test, batch_size=self.val_batch_size),
                          validation_steps=x_test.shape[0]//self.val_batch_size)
    logging.info('Training is done.')

    # Save the model.
    save_model(self.save_path, model)
    logging.info('Model saved.')

  def Tune(self, x_train, y_train, x_test, y_test, x, eps,
           starting_model_save_path=None, tuning=False):
    """Tune a model.

    Args:
      x_train: Training features.
      y_train: Training labels.
      x_test: Test features.
      y_test: Test labels.
      x: The data point to do tuning.
      eps: Norm bound we want to enforce around x.
      starting_model_save_path: Save path to load the model to start with.
      tuning: Whether to do tuning after all!
    """

    logging.info('--- NilModelController tuning ---')
    logging.info('Skipping... Nil model has no tuning.')


class TunedModelController(ModelController):
  """Controller for a tuned model."""

  DEFAULT_MODEL_SAVE_DIRECTORY = (
    '/nobackup-15T/transductive-defense-of-adversarial-perturbation/'
    'saved_models/one_shot_cheating/')
  DEFAULT_MODEL_NAME = 'NN-null-tune'
  DEFAULT_NUM_EPOCHS = 60

  def __init__(self, model_name=None, model_save_directory=None,
               num_epochs=None):
    # Initialize super.
    ModelController.__init__(self, model_name, model_save_directory)

    # Constants used in training.
    self.tr_batch_size = 32
    self.val_batch_size = 100

    self.num_epochs = num_epochs
    self.decay_start_epoch = 40
    self.num_iters_per_epoch = 400

    # TODO: We use a fixed learning rate, is this what we want?
    self.learning_rate = 0.0001

    self.beta_1_before_decay = 0.9
    self.beta_1_after_decay = 0.5
    self.lam = 1.0

    if self.model_name is None:
      logging.info('No model_name given, fall back to default \'%s\'',
                   TunedModelController.DEFAULT_MODEL_NAME)
      self.model_name = TunedModelController.DEFAULT_MODEL_NAME

    if self.model_save_directory is None:
      logging.info('No model_save_directory given, fall back to default \'%s\'',
                   TunedModelController.DEFAULT_MODEL_SAVE_DIRECTORY)
      self.model_save_directory = (
        TunedModelController.DEFAULT_MODEL_SAVE_DIRECTORY)

    if self.num_epochs is None:
      logging.info('No num_epochs given, fall back to default %s',
                   TunedModelController.DEFAULT_NUM_EPOCHS)
      self.num_epochs = TunedModelController.DEFAULT_NUM_EPOCHS

    # Path to save the tuned model.
    self.save_path = '{}{}'.format(self.model_save_directory, self.model_name)
    logging.info('Model save path: %s', self.save_path)

  def Train(self, x_train, y_train, x_test, y_test, training=False):
    """Train the model.

    Args:
      training: Whether to do training after all!
    """

    logging.info('--- TunedModelController training ---')
    logging.info('Skipping.. supposed to start with a model.'
                 'We need a test point: '
                 'If you start with a random model with a test point, this is '
                 ' equivalent to training, though at test time.')

  def Tune(self, x_train, y_train, x_test, y_test, x, eps,
           starting_model_save_path=None, tuning=False):
    """Tune the model.

    This is supposed to be called at test time (to be fully comply with theory.

    Args:
      x_train: Training features.
      y_train: Training labels.
      x_test: Test features.
      y_test: Test labels.
      x: The data point to do tuning.
      eps: Norm bound we want to enforce around x.
      starting_model_save_path: Save path to load the model to start with.
      tuning: Whether to do tuning after all!
    """

    logging.info('--- TunedModelController tuning ---')

    # If the user does not need tuning, just return! The assumption is that there
    # has been a model at save path.
    if not tuning:
      logging.info('Skip actual tuning, the model is assumed to be trained, '
                   'tuned and saved at \'%s\'', self.save_path)
      return

    logging.info('TunedModelController: Norm bound to shoot for is %s', eps)
    # We start with an existing model: Load it.
    # TODO: Handle when path is None, use a random initialization.
    assert starting_model_save_path is not None
    logging.info('Load starting model from \'%s\'', starting_model_save_path)
    model = load_model(starting_model_save_path)

    # x should be converted to a constant tensor to be used in loss function.
    # Call such constant tensor tx.
    #
    # TODO: We don't want to tune a point having small confidence.
    # Is it better for us to have some functionality to check confidence?
    tx = keras.backend.constant(x)

    # Define optimizer to be used.
    opt = keras.optimizers.adam(lr=self.learning_rate,
                                beta_1=self.beta_1_before_decay)

    # Compile the model with optimizer and loss
    model.compile(loss=totalLoss(model=model, x=tx, eps=eps,
                                 lam=self.lam),
                  optimizer=opt,
                  metrics=['accuracy', nilLoss,
                           regularizationLossMetric(
                               model=model, x=tx, eps=eps)])

    # Data generators used in data augmentation (we may use it if needed).
    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    # Scheduler controls learning rate decay and momentum.
    scheduler = get_scheduler(num_epochs=self.num_epochs,
                              decay_start_epoch=self.decay_start_epoch,
                              learning_rate=self.learning_rate,
                              beta_1_before_decay=self.beta_1_before_decay,
                              beta_1_after_decay=self.beta_1_after_decay)

    # Train the model.
    logging.info('Tuning starts.')
    with tensorflow.device('/gpu:0'):
      model.fit_generator(datagen.flow(x_train, y_train,
                                       batch_size=self.tr_batch_size),
                          steps_per_epoch=self.num_iters_per_epoch,
                          epochs=self.num_epochs,
                          callbacks=[scheduler],
                          validation_data=datagen.flow(
                              x_test, y_test, batch_size=self.val_batch_size),
                          validation_steps=x_test.shape[0]//self.val_batch_size)
    logging.info('Tuning is done.')

    # Save the model.
    save_model(self.save_path, model)
    logging.info('Model saved.')


class CarliniLiModelShellController(ModelController):
  """Controller for CarliniLiModelShell."""

  DEFAULT_MODEL_NAME = 'NN_null_carlini_shell'

  def __init__(self, model_name=None, model_save_directory=None,
               num_epochs=None):
    # Initialize super.
    ModelController.__init__(self, model_name, model_save_directory)

    # Number of classes.
    # TODO: Hard-wiring again! Switch to parameters.
    self.num_classes = 10
    if self.model_name is None:
      logging.info('No model_name given, fall back to default \'%s\'',
                   CarliniLiModelShellController.DEFAULT_MODEL_NAME)
      self.model_name = CarliniLiModelShellController.DEFAULT_MODEL_NAME

  def Train(self, x_train, y_train, x_test, y_test, training=False):
    """Train the model.

    Args:
      training: Whether to do training after all!
    """

    logging.info('--- CarliniLiModelShellController training ---')
    logging.info('Skipping.. suppose to start with a model.')

  def Tune(self, x_train, y_train, x_test, y_test, x, eps,
           starting_model_save_path=None, tuning=False):
    """Tune the model.

    This is supposed to be called at test time (to be fully comply with theory.

    Args:
      x_train: Training features.
      y_train: Training labels.
      x_test: Test features.
      y_test: Test labels.
      x: The data point to do tuning.
      eps: Norm bound we want to enforce around x.
      starting_model_save_path: Save path to load the model to start with.
      tuning: Whether to do tuning after all!
    """

    logging.info('--- CarliniLiModelShell tuning ---')
    assert starting_model_save_path is not None
    logging.info('Load starting model from \'%s\'', starting_model_save_path)
    model = load_model(starting_model_save_path)
    # Set self.model_shell, which is the object used tests in controller.
    logging.info('Set model_shell...')
    self.model_shell = CarliniLiModelShell(model, backend.get_session())

  def Test(self, x, y, eps):
    """Test correctness and robustness for CarliniLiModelShell.

    """

    logging.info('CarliniLiModelShell: Note the following two things:\n'
                 '(1) Correctness: It is tested for the shell predict(), '
                 'it is different from predict of the wrapped trained model.\n'
                 '(2) Robustness: The Carlini Li attack is performed against the '
                 'wrapped trained model and the adversarial image is found for '
                 'that model, but we then check the shell predict can give the '
                 'correct result.')
    super(CarliniLiModelShellController, self).Test(x, y, eps)


class ModelShellExperimentController(ModelController):
  """Controller for CarliniLiModelShell."""

  DEFAULT_MODEL_NAME = 'NN_null_carlini_shell'

  def __init__(self, model_name=None, model_save_directory=None,
               num_epochs=None):
    # Initialize super.
    ModelController.__init__(self, model_name, model_save_directory)

    # Number of classes.
    # TODO: Hard-wiring again! Switch to parameters.
    self.num_classes = 10
    if self.model_name is None:
      logging.info('No model_name given, fall back to default \'%s\'',
                   ModelShellExperimentController.DEFAULT_MODEL_NAME)
      self.model_name = ModelShellExperimentController.DEFAULT_MODEL_NAME

  def Train(self, x_train, y_train, x_test, y_test, training=False, ):
    """Train the model.

    Args:
      training: Whether to do training after all!
    """

    logging.info('--- CarliniLiModelShellController training ---')
    logging.info('Skipping.. suppose to start with a model.')

  def SampleTestPoint(self, sample_test_points=True,
                      X=None, Y=None, eps=None,
                      starting_model_save_path=None, num_samples=50):
    # Set self.model_shell.
    # TODO: Find a better point to set model shell.
    if self.model_shell is None:
      # Load the previously saved model from save path.
      logging.info(
        'Load from: \'%s\' and convert to a model shell',
                                          starting_model_save_path)
      self.model_shell = CIFARModelShell(
        load_model(starting_model_save_path), backend.get_session())
    assert self.model_shell is not None

    # Sample test points.
    self.num_samples = num_samples
    if sample_test_points:
      logging.info('We sample test points.')
      import random
      # Fix the seed for reproducibility.
      random.seed(123)
      # Permuted list of indices.
      random_indices = random.sample(range(10000), 10000)
      # List of indices sampled.
      self.sample_indices = []
      # List of images and labels.
      self.images = []
      self.labels = []
      # Boolean list recording if attack succeeded.
      self.is_attack_successful = []
      count_successful_attack = 0
      # List of perturbed images and predictions.
      self.perturbed_images = []
      
      trial_count = 0
      prev_trial_end = 28
      sample_index_start = 25
      while len(self.sample_indices) < num_samples:
        trial_count += 1
        logging.info('Trial number %d', trial_count)
        logging.info('Sampling candidate for sample %d',
                              sample_index_start+len(self.sample_indices)+1)
        sample_index = random_indices.pop()
        logging.info('Sample index %d', sample_index)
        if trial_count <= prev_trial_end:
          continue
        x, y = GetDataPoint(X, Y, sample_index)
        is_correct,_ = TestCorrectness(self.model_shell, x, y)
        if is_correct:
          logging.info('Adding sample index %d', sample_index)
          self.sample_indices.append(sample_index)
          self.images.append(x)
          self.labels.append(y)
        else:
          continue

        is_robust, xp, _ = TestRobustness(self.model_shell, x, eps)
        if is_robust:
          logging.info('Attack failed')
          self.is_attack_successful.append(False)
          self.perturbed_images.append(xp)
        else:
          count_successful_attack += 1
          logging.info('Attack succeeded')
          self.is_attack_successful.append(True)
          self.perturbed_images.append(xp)
    else:
      logging.info('You need to sample points for experiment.')
      assert False

    # Reset self.model_shell for later use.
    self.model_shell=None
    # Print the attack success rate.
    logging.info('Number of succesful attacks: %d', count_successful_attack)

  def Tune(self, x_train, y_train, x_test, y_test, x, eps,
           starting_model_save_path=None, tuning=False):
    """Tune the model.

    This is supposed to be called at test time (to be fully comply with theory.

    Args:
      x_train: Training features.
      y_train: Training labels.
      x_test: Test features.
      y_test: Test labels.
      x: The data point to do tuning.
      eps: Norm bound we want to enforce around x.
      starting_model_save_path: Save path to load the model to start with.
      tuning: Whether to do tuning after all!
    """

    logging.info('--- CarliniLiModelShell tuning ---')
    assert starting_model_save_path is not None
    logging.info('Load starting model from \'%s\'', starting_model_save_path)
    model = load_model(starting_model_save_path)
    # Set self.model_shell, which is the object used tests in controller.
    logging.info('Set model_shell...')
    self.model_shell = AcceleratedCarliniLiModelShell(model,
                                                      backend.get_session())
    self.model_shell_stronger = CarliniLiModelShell(model, backend.get_session())

  def Test(self, x, y, eps):
    """Test correctness and robustness for CarliniLiModelShell.

    """

    logging.info('CarliniLiModelShell: Note the following two things:\n'
                 '(1) Correctness: It is tested for the shell predict(), '
                 'it is different from predict of the wrapped trained model.\n'
                 '(2) Robustness: The Carlini Li attack is performed against the '
                 'wrapped trained model and the adversarial image is found for '
                 'that model, but we then check the shell predict can give the '
                 'correct result.')

    for i in range(self.num_samples):
      logging.info('Testing sample %d', i + 1)
      logging.info('Sample index: %d', self.sample_indices[i])
      x = self.images[i]
      y = self.labels[i]
      xp = self.perturbed_images[i]
      is_correct,softmax = TestCorrectness(self.model_shell, x, y)
      #if not is_correct:
      #  # If not correct, try with stronger setting.
      #  logging.info('CarliniWagnerShell is not correct with weaker setting, '
      #               'try again with stronger parameter...')
      #  is_correct,softmax_stronger = TestCorrectness(self.model_shell_stronger,
      #                                                x, y)

      #if not(self.is_attack_successful[i]):
      #  # If the attack has failed, skip.
      #  logging.info('The base model is already robust at this point')
      #  logging.info('Skip the test for robustness.')
      #  continue

      is_robust,_,_ = TestRobustness(self.model_shell, x, eps, xp, softmax)
      #if not (is_correct and is_robust):
      #  # If not correct or not robust, try with stronger setting.
      #  logging.info('CarliniWagnerShell is not correct or not robust '
      #              'with weaker setting, try again with stronger parameter...')
      #  is_robust,_,_ = TestRobustness(self.model_shell_stronger, x, eps, xp,
      #                                   softmax_stronger)

