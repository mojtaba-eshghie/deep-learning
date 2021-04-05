import numpy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy

from MiniBatchGDNN import MiniBatchGDNN
from utils import *

train_X, train_Y, train_y = unpickle('dataset/data_batch_1')
n_train_X = normalize(train_X)


nn = MiniBatchGDNN(k=10, d=5)
nn.train(n_train_X[0:5, :], train_Y, eta=0.1, epochs=20)

print(nn.accuracy(n_train_X[0:5, :], train_y, nn.W, nn.b))