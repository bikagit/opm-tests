from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from numpy import asarray
from matplotlib import pyplot
import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, '/Users/macbookn/hackatonwork/opm-common/python/opm/ml/')

# sys.path.insert(0, '../../../opm-common/python/opm/ml/python/opm/ml')
# /Users/macbookn/hackatonwork/opm-common/python/opm/ml/ml_tools/__init__.py
from ml_tools import export_model


model = Sequential()

model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
  optimizer='adam',
  loss='mean_squared_error',
  metrics=['accuracy']
)

#data = np.genfromtxt('bestpath.csv', delimiter=',')

path = 'bestpath.csv'



df = pd.read_csv(path)

x_train = df.drop('bestTol',axis=1)
#del df['pressure']
y_train = df['bestTol']

x_train=x_train.to_numpy()
x_train1 = x_train.reshape((len(x_train), 4))
y_train=y_train.to_numpy()

y_train1 = y_train.reshape((len(y_train), 1))


#print("data[1:, :4]")
#print(data[1:, :1])
#
#print("data[1:, 4]")
#print(data[1:, 1])



#x_train = data[1:, :4]
#y_train = data[1:, 4]
#
#
model.fit(
  x_train1,
  y_train1,
  epochs=1000,
  validation_split=0.33
)

yhat = model.predict(x_train1)

print('MSE: %.3f' % mean_squared_error(yhat, y_train1))
# print(yhat_plot)
# print('blah: %.3f' % mean_squared_error(y_plot, yhat_plot))

# blah = np.random.random((len(x_train), 4))
# # blah = blah.reshape((len(blah), 1))

# # yhat = model.predict(x_train)
# yhat = model.predict(np.array( [[0,0,0,0,0],] ))

# print(yhat)


# from kerasify import export_model
# lazy hack to be changed

export_model(model, 'linredNN.model')
