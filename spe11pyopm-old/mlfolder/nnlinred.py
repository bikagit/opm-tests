from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
# from keras.models import Sequential
# from keras.layers import Dense
from numpy import asarray
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os, sys
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adagrad, Adam, Adadelta

keras.utils.set_random_seed(1234)

sys.path.insert(0, '/Users/macbookn/hackatonwork/opm-common/python/opm/ml/')

# sys.path.insert(0, '../../../opm-common/python/opm/ml/python/opm/ml')
# /Users/macbookn/hackatonwork/opm-common/python/opm/ml/ml_tools/__init__.py
from ml_tools import export_model


model = Sequential()

model.add(Dense(8, activation='relu', input_dim=7))
model.add(Dense(16, activation='tanh'))
model.add(Dense(16, activation='tanh'))
# model.add(Dense(16, activation='tanh'))
# model.add(Dense(16, activation='tanh'))
model.add(Dense(16, activation='relu'))

model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile the model with AdaGrad optimizer
# learning_rate = 0.01
optimizer = Adam()
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['accuracy'])


# model.compile(
#   optimizer='adagrad',
#   loss='mean_squared_error',
#   # learning_rate='0.01',
#   metrics=['accuracy']
# )

#data = np.genfromtxt('bestpath.csv', delimiter=',')

path = 'bestpath.csv'

# Using sklearn to shuffle rows
from sklearn.utils import shuffle


df = pd.read_csv(path)
# boot = shuffle(df)

# print(df2)
boot = resample(df, replace=True, n_samples=51,stratify=df, random_state=42)
# boot = df
# boot = df.sample(n=2).reset_index()
# boot = boot.apply(lambda x: x.sample(frac=4, replace=True).values)

x_train = boot.drop('bestTol',axis=1)
#del df['pressure']
y_train = boot['bestTol']

x_train=x_train.to_numpy()


x_train1 = x_train.reshape((len(x_train), 7))
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
history = model.fit(
        x_train1,
        y_train1,
        epochs=1000,
        validation_split=0.30,
        # batch_size=4
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


# history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# from kerasify import export_model
# lazy hack to be changed

export_model(model, 'linredNN.model')
