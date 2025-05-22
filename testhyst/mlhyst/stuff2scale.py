import csv
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltbis

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from matplotlib import pyplot
import numpy as np
import keras
path = os.getcwd()
from tensorflow.keras.optimizers import Adagrad, Adam, Adadelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping


keras.utils.set_random_seed(1234)


sys.path.insert(0, '/Users/macbookn/activopmwkspc/edgedev/opm-common/python/opm/ml')

from ml_tools import export_model
from ml_tools import  MinMaxScalerLayer, MinMaxUnScalerLayer


# scaling 10x1
data: np.ndarray = np.random.uniform(-500, 500, (5, 1))
feature_ranges: list[tuple[float, float]] = [(0.0, 1.0), (-3.7, 0.0)]
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
data_min = 10.0
model = Sequential()
model.add(keras.layers.Input([10]))
model.add(MinMaxScalerLayer(feature_range=(0.0, 1.0)))
model.add(Dense(10,activation='tanh'))
model.add(Dense(10,activation='tanh'))
model.add(Dense(10,activation='tanh'))
model.add(Dense(10,activation='tanh'))
model.add(MinMaxUnScalerLayer(feature_range=(-3.7, -1.0)))
# #
model.get_layer(model.layers[0].name).adapt(data=data)
model.get_layer(model.layers[-1].name).adapt(data=data)


model.compile(loss='mean_squared_error', optimizer='adamax')

model.fit(test_x, test_y, epochs=1, verbose=True)

export_model(model, 'scalemodel.model')
