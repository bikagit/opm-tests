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
from tensorflow.keras.optimizers import Adamax, SGD, RMSprop
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

keras.utils.set_random_seed(1234)

sys.path.insert(0, '/Users/macbookn/activopmwkspc/edgedev/opm-common/python/opm/ml')

from ml_tools import export_model
from ml_tools import MinMaxScalerLayer, MinMaxUnScalerLayer

# Function to create a model with a specified number of layers and neurons
#def create_model(input_dim, layers, neurons, optimizer):
#    model = Sequential()
#    model.add(Dense(neurons[0], input_dim=input_dim, activation='relu'))
#    for i in range(1, layers):
#        model.add(Dense(neurons[i], activation='relu'))
#    model.add(Dense(1, activation='linear')) # Output layer for regression
#    model.compile(optimizer=optimizer, loss='mean_squared_error')
#    return model

def PreprocessData(Sh, data, wp):
    SandSmax = []
    krw = []
    krnw = []
    pc = []
    S2 = []

    with open(path+'/data/NewKrPcData_Swi0_06.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            S2.append(1-float(row[3]))
            krnw.append(float(row[6]))
            krw.append(float(row[5]))
            pc.append(float(row[4]))
            SandSmax.append([float(row[3]),float(row[2])])

    return krnw, krw, SandSmax
def generate_permuted_data(x, y, num_samples):
    permuted_x = []
    permuted_y = []

    for _ in range(num_samples):
        permuted_x_sample, permuted_y_sample = shuffle(x, y, random_state=42)
        permuted_x.append(permuted_x_sample)
        permuted_y.append(permuted_y_sample)

    permuted_x = np.vstack(permuted_x)
    permuted_y = np.hstack(permuted_y)

    return permuted_x, permuted_y
    
from sklearn.utils import resample

def trainNN(krnw, Smax):
    x = np.array(Smax)
    y = np.array(krnw)


    # Generate permuted data
    permuted_x, permuted_y = generate_permuted_data(x, y, num_samples=15)

    # Combine original and permuted data
    x_combined = np.vstack((x, permuted_x))
    y_combined = np.hstack((y, permuted_y))

#    X_train, X_test, y_train, y_test = train_test_split(x_combined, y_combined, test_size=0.2, random_state=42)
    
#    # Use bootstrapping to enlarge the dataset
    x_resampled, y_resampled = resample(x, y, n_samples=len(x) * 15, random_state=42)
##
    X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(keras.layers.Input([X_train.shape[1]]))
#    model.add(Dense(2048, activation='tanh'))
#    model.add(Dense(1024, activation='tanh'))
    # model.add(Dense(512, activation='tanh'))
    # model.add(Dense(256, activation='tanh'))
    # model.add(Dense(128, activation='tanh'))
    # model.add(Dense(64, activation='tanh'))
    # model.add(Dense(32, activation='tanh'))
    # model.add(Dense(4, activation='tanh'))
    # model.add(Dense(4, activation='tanh'))
    # model.add(Dense(4, activation='tanh'))
    # model.add(Dense(4, activation='tanh'))
    # model.add(Dense(4, activation='tanh'))
    # model.add(Dense(4, activation='tanh'))
    # model.add(Dense(4, activation='tanh'))
    # model.add(Dense(4, activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(1, activation='tanh'))

    optimizer = Adam(learning_rate=0.01)

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, start_from_epoch=150)
    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # Fit the model on the training dataset
    history = model.fit(X_train, y_train, epochs=350, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')

    y_pred = model.predict(X_test)

    return model

swl = 0.05
path = os.getcwd()
S = np.linspace(0.0, 1.0-swl, 700)
smax = 0.
S2 = 1.0 - S - swl

SS = []

for smax in S:
    SS.append(np.linspace(smax, 0, 100))

krnw, krw, SandSmax = PreprocessData(SS, "/Users/macbookn/activopmwkspc/pyDavid/pyopmnearwell/examples/hysteresis_models/Killough/CO2_KILLOUGH", "GW")

modelnonwett = trainNN(krnw, SandSmax)

satspace = np.linspace(0.0e-1, 1.0e0, 1000)
satMaxspace = np.linspace(4.15e-1, 4.15e-1, 1)

for valsat in satspace:
    for valsatmax in satMaxspace:
        # Drainage curves
        if valsatmax > valsat:
            xhat = np.array([[valsat, valsatmax]])
            yhatkrnw = modelnonwett.predict(xhat)
            pyplot.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue")
        # Imbibition curves
        if valsatmax < valsat:
            xhat = np.array([[valsat, valsatmax]])
            yhatkrnw = modelnonwett.predict(xhat)
            pyplot.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red")

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(r"$S_w$", fontsize=18)
plt.ylabel(r"$k_{rn} $", fontsize=18)
plt.legend(fontsize=12)
plt.savefig("output_figures/Killoughbis.png")
plt.show()

export_model(modelnonwett, 'model/oldmodelkrnw.model')

plt.savefig("newKillough.png")
