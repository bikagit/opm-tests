import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
from sklearn.metrics import mean_squared_error


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
  loss='binary_crossentropy',
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
  epochs=100,
  validation_split=0.2
)

yhat = model.predict(x_train1)

print('MSE: %.3f' % mean_squared_error(yhat, y_train1))
# print(yhat_plot)
# print('blah: %.3f' % mean_squared_error(y_plot, yhat_plot))

blah = np.random.random((len(x_train), 4))
# blah = blah.reshape((len(blah), 1))

# yhat = model.predict(x_train)
yhat = model.predict(np.array( [[0,0,0,0],] ))

print(yhat)
