import keras
import numpy as np
from models import ResNet

model = ResNet()
count = 0
for layer in model.layers:
  if len(layer.get_weights()) == 0:
    continue
  count += 1
  print count,':',layer.name
  #print np.asarray(layer.get_weights()).shape
  print layer.weights
  #print layer.get_weights()
