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
from sklearn.utils import resample

import keras
keras.utils.set_random_seed(1234)


sys.path.insert(0, '/Users/macbookn/activopmwkspc/edgedev/opm-common/python/opm/ml')

from ml_tools import export_model
from ml_tools import  MinMaxScalerLayer, MinMaxUnScalerLayer
import os
from sklearn.metrics import r2_score



def evaluate(Sh, data, wp):

    with open(path+'/sat.csv', 'w', newline='') as file:
        for S in Sh:        
            for s in S:        
                file.write(str(s)+"\n")

    pathCall = "/Users/macbookn/activopmwkspc/edgedev/build/opm-common/bin/hysteresis " + data +".DATA sat.csv relperms.csv " + wp + " 0"
    os.system(pathCall)

    krw = []
    krnw = []
    krM = []
    sT = []


    with open(path+'/relperms.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            #S2.append(float(row[0]))
            krnw.append(float(row[2]))
            krw.append(float(row[1]))
            krM.append(float(row[3]))
            sT.append(float(row[4]))

    i = 0
    start = 0
    end = 0
    for S in Sh:
        end = len(S) + start
        # input = "D" + str(i)
        # if 1-S[0] < 1-S[-1]:
        input = "I" + str(1-i)
        i = i + 1
       
        plt.plot(1-S, krnw[start:end], label = "KRNW"+input)
        # plt.plot(S, krw[start:end],label = "KRW"+input)
        start = end


# Function to create a model with a specified number of layers and neurons
def create_model(input_dim, layers, neurons, optimizer):
    model = Sequential()
    model.add(keras.layers.Input([input_dim]))
    for i in range(1, layers):
        model.add(Dense(neurons[i], activation='sigmoid'))
    model.add(Dense(1, activation='tanh'))  # Output layer for regression
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def PreprocessData(Sh, data, wp):

    SandSmax = []

    krw = []
    krnw = []
    pc = []
    S2 = []
    swi = []

    # with open(path+'/data/KrPcData_Combinedt050.csv', newline='') as csvfile:
    with open(path+data, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            S2.append(1-float(row[3]))
            krnw.append(float(row[6]))
            swi.append(float(row[7]))
            krw.append(float(row[5]))
            pc.append(float(row[4]))
            # SandSmax.append([1.0-float(row[3]),1.0-float(row[2]),float(row[7]),float(row[0])])
            SandSmax.append([1.0-float(row[3]),1.0-float(row[2])])

    return krnw, krw, SandSmax

def generate_synthetic_data(x, y, num_samples):
    synthetic_x = []
    synthetic_y = []

    for _ in range(num_samples):
        idx = np.random.randint(0, len(x))
        noise = np.random.normal(0, 0.01, x.shape[1])
        synthetic_x.append(x[idx] + noise)
        synthetic_y.append(y[idx] + noise[0])  # Assuming y is a single column

    synthetic_x = np.array(synthetic_x)
    synthetic_y = np.array(synthetic_y)

    return synthetic_x, synthetic_y


def trainNN(krnw, Smax):
    x = np.array(Smax)
    y = np.array(krnw)

    x_resampled, y_resampled = resample(x, y, n_samples=len(x) * 5, random_state=42)
    synthetic_x, synthetic_y = generate_synthetic_data(x_resampled, y_resampled, num_samples=len(x) * 5)
    x_combined = np.vstack((x, synthetic_x))
    y_combined = np.hstack((y, synthetic_y))
    X_train, X_test, y_train, y_test = train_test_split(x_combined, y_combined, test_size=0.3, random_state=42)
    X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # sensitive to learning rate 0.01 is default, 0.005 is better for this data, 0.001 is too tight 
    model = create_model(X_train.shape[1], 10, [19]*10, Adam(learning_rate=0.01))
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, start_from_epoch=100)
    history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
    return model, history, X_test, y_test, x_val, y_val


swl = 0.05
path = os.getcwd()
S = np.linspace(0.0, 1.0, 7)
smax = 0.
S2 = 1.0 - S - swl

SS = []

for smax in S:
    SS.append(np.linspace(smax, 0, 100))

SatSpace = []

SS = []
for smax in S:
    SS.append(np.linspace(0,1-smax,30))


index_ranges_imbib = [
                        (99, 167),
                    #   (345, 398),
                      (399,435),
                    #   (436,452)
                      ]

SatSpace.append(np.linspace(0.41,0.78,30))
# SatSpace.append(np.linspace(0.25,0.78,30))
SatSpace.append(np.linspace(0.0618,0.78,30))


evaluate(SatSpace, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P-1D-OILINJMULTI", "WO")

train_file = '/data/curated_data/KrPcData_Combinedtot.csv'

krnw, krw, SandSmax = PreprocessData(SS, train_file, "GW")

original_krnw, original_krw, original_SandSmax = PreprocessData(SS, '/data/curated_data/KrPcData_Swi0_06-CORRECTED.csv', "GW")

modelnonwett, history, X_test, y_test, x_val, y_val = trainNN(krnw, SandSmax)


export_model(modelnonwett, 'model/oldmodelkrnwSPE1CASE2_2P.model')
export_model(modelnonwett, 'model/oldmodelkrnw.model')

# Create an output folder if it doesn't exist
output_folder = "output_figures"
os.makedirs(output_folder, exist_ok=True)

# Subplot 4: DNN vs Original Data (Drainage + Imbibition)
# plt.subplot(1, 6, 4)
x = np.array(SandSmax)
y = np.array(krnw)

original_x = np.array(original_SandSmax)
original_y = np.array(original_krnw)


y_pred_all = modelnonwett.predict(x)
# Predictions
y_pred_test = modelnonwett.predict(X_test)
y_pred_val = modelnonwett.predict(x_val)
y_pred_all = modelnonwett.predict(x)


# # plt.subplot(1, 4, 2)
# satspace = np.linspace(0.1, 1.0, 100)
# satMaxspace = np.linspace(0.1, 0.91, 7)
# tag = 0
# imbibition_label_added = False
# drainage_label_added = False

# for valsat in satspace:
#     for valsatmax in satMaxspace:
#         xhat = np.array([[valsat, valsatmax]])
#         yhatkrnw = modelnonwett.predict(xhat)
#         # if valsatmax > valsat:
#         #     if not imbibition_label_added:
#         #         plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue", label="Imbibition")
#         #         imbibition_label_added = True
#         #     else:
#         #         plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue")
#         if valsatmax < valsat:
#             if not drainage_label_added:
#                 plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red", label="Drainage")
#                 drainage_label_added = True
#             else:
#                 plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red")

# plt.plot(label='Drainage')
# plt.title('Drainage and Imbibition')
# plt.xlabel('$S_n$'); plt.ylabel('$k_{rn}$'); plt.grid(True)


for i, (start, end) in enumerate(index_ranges_imbib):
    plt.scatter(original_x[start:end, 0], original_y[start:end], alpha=0.6, color='cyan', label='Original Imbib' if i == 0 else "")
    plt.scatter(x[start:end, 0], y_pred_all[start:end], alpha=0.6, color='black', marker="o", label='DNN Imbib' if i == 0 else "")
    # plt.scatter(x_val[start:end, 0], y_pred_val[start:end], alpha=0.6, color='red', label='Original Imbib' if i == 0 else "")

plt.xlabel(r"$S_n$", fontsize=14)
plt.ylabel(r"$k_{rn}$", fontsize=14)
plt.title('Imbibition View', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)


# Save the figure
output_path = os.path.join(output_folder, "comparingwithKillough.png")
plt.tight_layout()
plt.savefig(output_path)
# plt.show()
