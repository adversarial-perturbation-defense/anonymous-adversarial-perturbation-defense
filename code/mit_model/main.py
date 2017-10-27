"""A driver program for end-to-end train and test.

Examples:
1. Train and test a nil model, see shell_scripts/nil_model_test_run.sh
2. Train and test a tuned model, see shell_scripts/tuned_model_test_run.sh.
"""

# TODO: The following import is correct (google.apputils) but pylint
# has a bug and reports a false positive of 'no-name-in-module'. See here:
#
# https://github.com/PyCQA/pylint/issues/1524
#
# Before this bug is fixed, disble pylint for now.
from google.apputils import app  # pylint: disable=no-name-in-module
import gflags
import glog as logging
from data_utils import GetDataPoint
from data_utils import LoadCIFAR10
from data_utils import Visualize  # pylint: disable=unused-import
import model_controller


# This is the norm bound we enforce on a valid attack.
#
# This constant is used in the following paper when they evaluated the
# Carlini-Wagner attacks:
#
# Towards Deep Learning Models Resistant to Adversarial Attacks
#
# by Madry, Makelov, Schmidt, Tsipras, and Vladu.
# https://arxiv.org/pdf/1706.06083.pdf
#
# TODO: We may want to figure out how is this determined.
DEFAULT_NORM_BOUND = 8.0 / 255.0


# ######################
# Model parameters.
# ######################

# Model type. Now support:
# 1. nil
# 2. tuned
# 3. carilini_shell
#
# TODO: Our tuned model is only for one-shot cheating,
# Support other games and add flags here..
gflags.DEFINE_string('model_type', 'nil',
                     'Type of model')

gflags.DEFINE_string('model_name', None,
                     'Name of the model')

gflags.DEFINE_string('model_save_directory', None,
                     'Directory path to save a model named model_name.')

# We save models -- the following two flags control whether we want to
# train again to overwrite the saved models or not.
gflags.DEFINE_boolean('force_training', False,
                      'Whether to rain it again.')

gflags.DEFINE_boolean('force_tuning', False,
                      'Whether to tune it again.')

# ######################
# Training parameters.
# ######################
gflags.DEFINE_integer('num_epochs', 1,
                      'Number of epochs to train/tune.')

# ######################
# Tuning parameters.
# ######################
# Tuning requires starting with a model. We pass the path where it's saved.
gflags.DEFINE_string('starting_model_save_path', None,
                     'Name of a model to start with')

# ######################
# Testing parameters.
# ######################

gflags.DEFINE_boolean('find_vul_point', False,
                      'Whether to find a vulnerable point.')

gflags.DEFINE_integer('test_image_index', None,
                      'Index of the test image to test.')

gflags.DEFINE_string('ref_model_save_path', None,
                     'Name of a reference model for transfer attack.')

FLAGS = gflags.FLAGS

def main(_):
  """Application entrance.

  """

  controller = None
  if FLAGS.model_type == 'nil':
    logging.info('Nil model controller.')
    # This model is special, it is without Batch Normalization.
    #
    # TODO: Bring back BN?
    controller = model_controller.NilModelController(
      model_name=FLAGS.model_name,
      model_save_directory=FLAGS.model_save_directory,
      num_epochs=FLAGS.num_epochs)
  elif FLAGS.model_type == 'tuned':
    logging.info('Tuned model controller.')
    controller = model_controller.TunedModelController(
      model_name=FLAGS.model_name,
      model_save_directory=FLAGS.model_save_directory,
      num_epochs=FLAGS.num_epochs)
  elif FLAGS.model_type == 'carlini_shell':
    logging.info('CarliniLiModelShell model controller.')
    controller = model_controller.CarliniLiModelShellController(
      model_name=FLAGS.model_name,
      model_save_directory=FLAGS.model_save_directory,
      num_epochs=FLAGS.num_epochs)
  else:
    logging.info('Unknown model type, abort...')
    assert False

  # Load the CIFAR10 dataset.
  # TODO: Now only handles the CIFAR10 dataset, handle others?\
  #
  # NOTE: The first compoent (training data) is never used, skip with _.
  #
  # NOTE: It seems better that we handle loading image using index here,
  # instead of insiding model controller. The reason is that later in the online
  # model we will first perturb an image and then test it. In that case there is
  # *no* index for that perturbed image.
  # Load the CIFAR10 dataset.
  (x_train, y_train), (x_test, y_test) = LoadCIFAR10()
  x, y = GetDataPoint(x_test, y_test, FLAGS.test_image_index)

  # Train the model.
  controller.Train(x_train, y_train, x_test, y_test, 
                   training=FLAGS.force_training)

  # Find a vulnerable point of the given model.
  controller.FindVulnerablePoint(find_vul_point=FLAGS.find_vul_point,
                                X=x_test, Y=y_test, eps=DEFAULT_NORM_BOUND,
                   starting_model_save_path=FLAGS.starting_model_save_path)

  # Tune the model.
  controller.Tune(x_train, y_train, x_test, y_test, x, DEFAULT_NORM_BOUND,
                  starting_model_save_path=FLAGS.starting_model_save_path,
                  tuning=FLAGS.force_tuning)

  # Ready for testing. Enter test stage.
  controller.EnterTestStage()

  # Test the model for both correctness and robustness.
  controller.Test(x, y, DEFAULT_NORM_BOUND)

  # Test the model against a transfer attack on a ref model.
  controller.TestTransfer(x, y, DEFAULT_NORM_BOUND, FLAGS.ref_model_save_path)

if __name__ == '__main__':
  app.run()
