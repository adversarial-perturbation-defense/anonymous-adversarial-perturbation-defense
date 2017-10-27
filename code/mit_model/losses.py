from keras import backend

from keras.losses import categorical_crossentropy, kullback_leibler_divergence

# NOTES
# PGD may be merged to 'perturbations.py' (If such script is created later)
# PGD will be modified to take the following parameters as inputs
#RANDOM_START = True
#num_steps = 7
#num_classes = 10
#step_size = eps / 4
def PGDPerturbation(model, x, eps):
    '''
    Function that computes and return the perturbation using PGD

    model: Current model
    x: Batch of images that robustness is required
    eps: Radius of the norm constraint epsilon
    '''
    RANDOM_START = True
    num_steps = 7
    num_classes = 10
    step_size = eps / 4
    # Choose a random initial point near x
    if RANDOM_START:
        x_curr = x + backend.random_uniform_variable(
                                                backend.int_shape(x), 0, eps )
    else:
        x_curr = x

    # Construct a fake y based on model prediction at the input x
    fake_y = backend.one_hot(backend.argmax(model(x)), 10)

    # Prediction at initial point (initial point may be different from x)
    pred_x = model(x_curr)

    for i in range(num_steps):
        # We want new x to be as different prediction as possible
        # from fake y based on model prediction
        loss = categorical_crossentropy(fake_y, pred_x)

        # The descending direction d
        grad = backend.gradients(loss, [x_curr])[0]
        d = backend.stop_gradient(grad)

        # Perturbation tensor w.r.t the input point x
        pert = (x_curr + step_size * backend.sign(d)) - x
        # Clip perturbation
        pert_clipped = backend.clip(pert, -eps, eps)

        # Update the current x
        x_curr = x + pert_clipped
        x_curr = backend.clip(x_curr, -0.5, 0.5)

        # Update the prediction at the current point
        pred_x = model(x_curr)

    return x_curr

def regularizationLoss(model, x, eps):
    '''
    Function that actually computes and returns the regularization loss

    model: Current model
    x: Batch of images that robustness is required
    eps: Radius of the norm constraint epsilon
    '''

    # Find virtual adversarial perturbation r
    r = PGDPerturbation(model, x, eps)
    # Reset the learning phase

    # Prediction at point x
    pred_x = model(x)
    # Prediction at point (x+r)
    pred_pert_x = model(x + r)

    # Compute virtual adversarial loss below
    # Ideally the prediction at point (x+r) should be close to prediction
    # at point x. Therefore in this case, y_true = pred_x.
    # KL divergence can be used similarly instead of crossentropy
    reg_loss = categorical_crossentropy(pred_x, pred_pert_x)

    return reg_loss

def regularizationLossMetric(model, x, eps):
    '''
    Function returning the regularization loss function
    Only used to be used as a function for metric

    model: Current model
    x: Batch of images that robustness is required
    eps: Radius of the norm constraint epsilon

    NOTE: keras only takes specific form of loss, or metric function taking
        (y_true, y_pred) as inputs
    '''
    def regLoss(y_true, y_pred):
        reg_loss = regularizationLoss(model, x, eps)
        return reg_loss

    return regLoss

def nilLoss(y_true, y_pred):
    '''
    Function returning the nil loss
    Not actually needed, but implemented as a function for metric

    NOTE: keras only takes specific form of loss, or metric function taking
        (y_true, y_pred) as inputs
    '''
    return categorical_crossentropy(y_true, y_pred)

def totalLoss(model, x, eps, lam):
    '''
    Function returning the total loss function

    model: Current model
    x: Batch of images that robustness is required
    eps: Radius of the norm constraint epsilon
    lam: The regularization factor lambda

    NOTE: keras only takes specific form of loss, or metric function taking
        (y_true, y_pred) as inputs
    '''
    def lossFunc(y_true, y_pred):
        nil_loss = categorical_crossentropy(y_true, y_pred)

        # Compute virtual adversarial loss below
        # Ideally the prediction at point (x+r) should be close to prediction
        # at point x. Therefore in this case, y_true = pred_x.
        # KL divergence can be used similarly instead of crossentropy
        reg_loss = regularizationLoss(model, x, eps)

        return nil_loss + lam * reg_loss
    return lossFunc
