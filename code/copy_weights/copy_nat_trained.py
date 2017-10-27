from copy_weights import copiedNaturallyTrainedModel
from train_utils import save_model

model = copiedNaturallyTrainedModel()
save_model('./saved_models/nat_trained_mit_model', model)
