import csv
import numpy as np
import os, sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import numpy as np
import keras
path = os.getcwd()

sys.path.insert(0, '/Users/macbookn/activopmwkspc/stable_releases/opm-common/python/opm/ml')

from ml_tools import export_model
from ml_tools import  MinMaxScalerLayer, MinMaxUnScalerLayer


def PreprocessData(Sh, data, wp):
    # smax = .0
    krnw = []
    krw = []
    SandSmax = []
    for S in Sh:
       smax = .0
       for s in S:
            smax = max(s,smax)
            pathCall = "/Users/macbookn/hackatonwork/build/opm-common/bin/hysteresis " + data +".DATA " + str(s) + " " + str(smax) + " " + wp + " 0 > relperms.csv"
            x = os.system(pathCall)
            SandSmax.append([s, smax])
                # print(s,smax,float(output)

            with open(path+'/relperms.csv', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:   
                    krnw.append(float(row[0]))
                    krw.append(float(row[1]))                  
    
    i = 0
    start = 0
    end = 0
    for S in Sh:
        end = len(S) + start
        input = "D" + str(i)
        if S[0] > S[-1]:
            input = "I" + str(i)
            i = i + 1
        
        plt.plot(S, krw[start:end], label = "KRW"+input)
        plt.plot(S, krnw[start:end], label = "KRNW"+input)
        start = end
    return krnw,krw, SandSmax


def trainNN(krnw, Smax):

    x = np.array(Smax)
    y = np.array(krnw)
    
    model = Sequential()
    model.add(Dense(3, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1))
    # define the loss function and optimization algorithm
    model.compile(loss='mse', optimizer='adam')
    # # ft the model on the training dataset
    model.fit(x, y, epochs=2000, batch_size=100, verbose=0)
    # make predictions for the input data
    model.save("models/trainNNhyst.keras")

    return model


def predictNN(Sh):
    oldmodel = keras.models.load_model("models/trainNNhyst.keras")
    xhat = np.array([Sh])
    yhat = oldmodel.predict(xhat)
    return yhat


swl = 0.05
path = os.getcwd()
S = np.linspace(0.0, 1.0-swl, 10)
smax = 0.
S2 = 1.0 - S - swl;
# evaluate2([S, S2], "CO2", "GW")

SS = [S]

for smax in S:
    SS.append(np.linspace(smax, 0, 10))

krnw, krw, SandSmax = PreprocessData(SS, "../CO2", "GW")

# model1 = trainNN(krnw, SandSmax)
# model2 = trainNN(krw, SandSmax)

# yhat = predictNN([0.3,0.5])

satspace = np.linspace(0.1, 0.95, 30)

satMaxspace = np.linspace(0.6, 0.9, 2)

for valsatmax in satMaxspace:
    for valsat in satspace:
        if valsatmax>valsat:
            xhat = np.array([[valsat,valsatmax]])

            oldmodelkrnw = keras.models.load_model("models/finemodel.keras")
            oldmodelkrw = keras.models.load_model("models/krwtrainNNhyst.keras")

            yhatkrnw = oldmodelkrnw.predict(xhat)
            yhat2krw = oldmodelkrw.predict(xhat)

            pyplot.plot(xhat.flat[0],  yhatkrnw,marker="o", markersize=5, markeredgecolor="red",label="Predictedkrnw")
            pyplot.plot(xhat.flat[0],  yhat2krw,marker="*", markersize=5, markeredgecolor="blue",label="Predictedkrw")

            export_model(oldmodelkrnw, 'modelkrnw.model')
            export_model(oldmodelkrw, 'modelkrw.model')


plt.legend()
plt.savefig("output/Killough.png")
plt.show()

