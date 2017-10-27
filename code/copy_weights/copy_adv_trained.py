from copy_weights import copiedAdversariallyTrainedModel
from train_utils import save_model

model = copiedAdversariallyTrainedModel()
save_model('./saved_models/adv_trained_mit_model', model)
