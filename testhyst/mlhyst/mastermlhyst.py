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


# Function to create a model with a specified number of layers and neurons
def create_model(input_dim, layers, neurons, optimizer):
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=input_dim, activation='relu'))
    for i in range(1, layers):
        model.add(Dense(neurons[i], activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def PreprocessData(Sh, data, wp):

    SandSmax = []

    with open(path+'/sat.csv', 'w', newline='') as file:
        for S in Sh:
            smax = .0
            for s in S:
                smax = max(s,smax)
                file.write(str(s)+"\n")
                SandSmax.append([s, smax])

    pathCall = "/Users/macbookn/activopmwkspc/edgedev/build/opm-common/bin/hysteresis " + data +".DATA sat.csv relperms.csv " + wp + " 0"
    os.system(pathCall)

    krw = []
    krnw = []
    krM = []
    sT = []
    S2 = []

    with open(path+'/relperms.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            S2.append(float(row[0]))
            krnw.append(float(row[1]))
            krw.append(float(row[2]))
            krM.append(float(row[3]))
            sT.append([float(row[4])])

    i = 0
    start = 0
    end = 0

    for S in Sh:
        end = len(S) + start

        input = "D" + str(i)
        if S[0] > S[-1]:
            input = "I" + str(i)
            i = i + 1
        # plt.plot(S, krnw[start:end], label = "KRNW"+input)
        # plt.plot(S, krw[start:end],label = "KRW"+input)

        start = end

    return krnw,krw, SandSmax

def trainNN(krnw, Smax):
    x = np.array(Smax)
    y = np.array(krnw)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    feature_ranges: list[tuple[float, float]] = [(0.0, 1.0), (-3.7, 0.0)]

    data: np.ndarray = np.random.uniform(0, 1, X_train.shape[1])
    data_min = 0.0
    data_max = 1.0
    # print(max(X_train[:,0]))
    model = Sequential()
    model.add(keras.layers.Input([X_train.shape[1]]))
    # model.add(MinMaxScalerLayer(data_min = 0.0, data_max = 1.0))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    # model.add(MinMaxUnScalerLayer(data_min = 0.0,data_max = 1.0))
    # # # #
    # model.get_layer(model.layers[0].name).adapt(data=data)
    # model.get_layer(model.layers[-1].name).adapt(data=data)

    optimizer = Adam(learning_rate=0.01)



    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True,start_from_epoch=1,)
    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # # ft the model on the training dataset
    history = model.fit(X_train, y_train, epochs=350, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
    # make predictions for the input data
    # model.save("models/trainNNhyst.keras")


    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')

    y_pred = model.predict(X_test)
    # print('MSE: %.3f' % mean_squared_error(Make, y))

    # a = plt.axes(aspect='equal')
    # plt.scatter(y_test, y_pred)
    # plt.xlabel('True values')
    # plt.ylabel('Predicted values')
    # # plt.xlim([0, 50000])
    # # plt.ylim([0, 50000])
    # # plt.plot([0, 50000], [0, 50000])
    # plt.plot()
    #
    # plt.savefig("predloss.png")

    pltbis.plot(history.history['loss'])
    pltbis.plot(history.history['val_loss'])

    pltbis.savefig("loss.png")

    return model


swl = 0.05
path = os.getcwd()
S = np.linspace(0.0, 1.0-swl, 700)
smax = 0.
S2 = 1.0 - S - swl;

SS = []


for smax in S:
    SS.append(np.linspace(smax,0, 100))

krnw, krw, SandSmax = PreprocessData(SS, "/Users/macbookn/activopmwkspc/pyDavid/pyopmnearwell/examples/hysteresis_models/Killough/CO2_KILLOUGH", "GW")

modelnonwett = trainNN(krnw, SandSmax)

#
satspace = np.linspace(0.1, 0.95, 30)

satMaxspace = np.linspace(0.5, 0.8, 2)

for valsatmax in satMaxspace:
    for valsat in satspace:
        if valsatmax>valsat:
            xhat = np.array([[valsat,valsatmax]])
            yhatkrnw = modelnonwett.predict(xhat)

export_model(modelnonwett, 'oldmodelkrnw.model')

plt.savefig("newKillough.png")
