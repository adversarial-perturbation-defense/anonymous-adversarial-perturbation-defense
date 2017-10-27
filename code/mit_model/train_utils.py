import numpy as np
import keras

from keras.models import model_from_yaml

class Scheduler(keras.callbacks.Callback):
    '''
    Callback controlling the learning rate and momentum according to the epochs
    This scheduler is mainly designed to work with adam optimizer
    '''
    def __init__(self, lr_schedule, b1_schedule):
        super(Scheduler, self).__init__()
        self.lr_schedule = lr_schedule
        self.b1_schedule = b1_schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if not hasattr(self.model.optimizer, 'beta_1'):
            raise ValueError('Optimizer must have a "beta_1" attribute.')
            
        lr = self.lr_schedule(epoch)
        b1 = self.b1_schedule(epoch)
        
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError(
                    'The output of the "lr_schedule" function should be float.')
        if not isinstance(b1, (float, np.float32, np.float64)):
            raise ValueError(
                    'The output of the "b1_schedule" function should be float.')
        
        from keras import backend as K
        
        print 'current_lr:', lr
        print 'current_b1:', b1
        
        K.set_value(self.model.optimizer.lr, lr)
        K.set_value(self.model.optimizer.beta_1, b1)

def get_scheduler(num_epochs = 120, 
                decay_start_epoch = 80, 
                learning_rate = 0.0001, 
                beta_1_before_decay = 0.9, 
                beta_1_after_decay = 0.5):
    '''
    Function returning scheduler object
    It assumes learning rate to linearly decrease after decay_start_epoch,
    momentum changes right after decay_start_epoch
    '''
    # Learning rate scheduler
    def lr_scheduler(epoch):
        if epoch < decay_start_epoch:
            ret_lr = learning_rate
        else:
            ret_lr = (float(num_epochs - epoch) / 
                        float(num_epochs - decay_start_epoch)) * learning_rate
            
        return ret_lr

    # Momentum scheduler
    def b1_scheduler(epoch):
        if epoch < decay_start_epoch:
            ret_beta_1 = beta_1_before_decay
        else:
            ret_beta_1 = beta_1_after_decay
            
        return ret_beta_1

    # Return the comprehensive scheduler
    return Scheduler(lr_schedule=lr_scheduler, b1_schedule=b1_scheduler)

def save_model(path, model):
    # Save the model architecture
    yaml_string = model.to_yaml()
    with open(path + ".yaml", "w") as model_file:
        model_file.write(yaml_string)
        
    # Save the model weights
    model.save_weights(path + ".h5")

    print 'Saving model...'

def load_model(path):
    # Load model architecture
    with open(path + ".yaml", 'r') as model_file:
        yaml_string = model_file.read()

    model = model_from_yaml(yaml_string)

    # Load model weights
    model.load_weights(path + ".h5")

    # Return the loaded model
    print 'Loading model...'
    return model
