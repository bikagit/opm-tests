import csv
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltbis
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

keras.utils.set_random_seed(1234)

sys.path.insert(0, '/Users/macbookn/activopmwkspc/edgedev/opm-common/python/opm/ml')

from ml_tools import export_model
from ml_tools import MinMaxScalerLayer, MinMaxUnScalerLayer


# Define the custom environment
class CustomEnv:
    def __init__(self, Sh, data, wp):
        self.Sh = Sh
        self.data = data
        self.wp = wp

    def preprocess_data(self):
        SandSmax = []
        with open('sat.csv', 'w', newline='') as file:
            for S in self.Sh:
                smax = .0
                for s in S:
                    smax = max(s, smax)
                    file.write(str(s) + "," + str(smax) + "\n")
                    SandSmax.append([s, smax])

        pathCall = f"/Users/macbookn/activopmwkspc/edgedev/build/opm-common/bin/hysteresis {self.data}.DATA sat.csv relperms.csv {self.wp} 0"
        os.system(pathCall)

        krw, krnw, krM, sT, S2 = [], [], [], [], []
        with open('relperms.csv', newline='') as csvfile:
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
        # plt.plot(1-S, krnw[0:100], label = "KRNW0")
        # plt.plot(1-S, krnw[100:200], label = "KRNW1")
        # plt.plot(1-S, krnw[200:300], label = "KRNW2")
        # plt.plot(1-S, krnw[300:400], label = "KRNW3")
        # plt.plot(1-S, krnw[400:500],label = "KRNW4")
        # plt.plot(1-S, krnw[500:600],label = "KRNW4")
        # plt.plot(1-S, krnw[600:700],label = "KRNW6")
        # # plt.plot(1-S, krnw[700:800],label = "KRNW7")
        # # plt.plot(1-S, krnw[800:900],label = "KRNW8")
        for S in self.Sh:
            end = len(S) + start

            input = "D" + str(i)
            # plt.plot(1-S, krnw[start:end],label = "KRW"+input)

            if S[0] < S[-1]:
                input = "I" + str(i)
                i = i + 1
            plt.plot(1-S, krnw[start:end],label = "KRW"+input)

            print(start)
            print(end)
            # plt.plot(S, krnw[600:700], label = "KRNW"+input)
            plt.legend(fontsize=12)

            start = end
        plt.show()
        return np.array(krnw), np.array(krw), np.array(SandSmax)


# Function to create a model with a specified number of layers and neurons
def create_model(input_dim, layers, neurons, optimizer):
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=input_dim, activation='tanh'))
    for i in range(1, layers):
        model.add(Dense(neurons[i], activation='tanh'))
    model.add(Dense(1, activation='tanh'))  # Output layer for regression
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def train_nn(env):
    krnw, krw, SandSmax = env.preprocess_data()
    x = np.array(SandSmax)
    y = np.array(krnw)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Use create_model method to create the model
    model = create_model(input_dim=X_train.shape[1], layers=2, neurons=[4, 4], optimizer=Adam(learning_rate=0.01))

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, start_from_epoch=100)
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    # Fit the model on the training dataset
    history = model.fit(X_train, y_train, epochs=350, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')

    y_pred = model.predict(X_test)

    pltbis.plot(history.history['loss'])
    pltbis.plot(history.history['val_loss'])
    pltbis.savefig("loss.png")

    return model


# Example usage
Sh = np.linspace(0.0, 1.0 - 0.05, 70)
SS = [np.linspace(smax, 0, 10) for smax in Sh]

env = CustomEnv(SS, "/Users/macbookn/activopmwkspc/pyDavid/pyopmnearwell/examples/hysteresis_models/Killough/CO2_KILLOUGH", "GW")
model_nonwett = train_nn(env)

satspace = np.linspace(0.1, 0.95, 30)
satMaxspace = np.linspace(0.5, 0.8, 2)

for valsatmax in satMaxspace:
    for valsat in satspace:
        if valsatmax > valsat:
            xhat = np.array([[valsat, valsatmax]])
            yhatkrnw = model_nonwett.predict(xhat)

export_model(model_nonwett, 'oldmodelkrnw.model')

plt.savefig("newKillough.png")


