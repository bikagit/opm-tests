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


# Function to create a model with a specified number of layers and neurons
def create_model(input_dim, layers, neurons, optimizer):
    model = Sequential()
    model.add(keras.layers.Input([input_dim]))
    # model.add(Dense(neurons[0], input_dim=input_dim, activation='tanh'))
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
        noise = np.random.normal(0, 0.19, x.shape[1])
        synthetic_x.append(x[idx] + noise)
        synthetic_y.append(y[idx] + noise[0])  # Assuming y is a single column

    synthetic_x = np.array(synthetic_x)
    synthetic_y = np.array(synthetic_y)

    return synthetic_x, synthetic_y


def trainNN(krnw, Smax):
    x = np.array(Smax)
    y = np.array(krnw)

    # Use bootstrapping to enlarge the dataset
    x_resampled, y_resampled = resample(x, y, n_samples=len(x) * 15, random_state=42)
    #
    # X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.3, random_state=42)
    # Generate synthetic data
    synthetic_x, synthetic_y = generate_synthetic_data(x_resampled, y_resampled, num_samples=len(x) * 15)

    # Combine original and synthetic data
    x_combined = np.vstack((x, synthetic_x))
    y_combined = np.hstack((y, synthetic_y))

    X_train, X_test, y_train, y_test = train_test_split(x_combined, y_combined, test_size=0.3, random_state=42)


    # Define model parameters
    input_dim = X_train.shape[1]
    # layers = 8  # Number of hidden layers
    # neurons = [512,256,128,64,32,16,8,4] * layers  # 4 neurons per layer
    # layers = 10  # Number of hidden layers
    # neurons = [30] * layers  # 4 neurons per layer
    layers = 10  # Number of hidden layers
    neurons = [30] * layers  # 4 neurons per layer
    optimizer = Adam(learning_rate=0.01)
    
    # Create model using the reusable function
    model = create_model(input_dim, layers, neurons, optimizer)

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, start_from_epoch=100)

    # Compile and train the model
    history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')


    return model, history, X_test, y_test


swl = 0.05
path = os.getcwd()
S = np.linspace(0.0, 1.0-swl, 700)
smax = 0.
S2 = 1.0 - S - swl

SS = []

for smax in S:
    SS.append(np.linspace(smax, 0, 100))

# krnw, krw, SandSmax = PreprocessData(SS, "/Users/macbookn/activopmwkspc/pyDavid/pyopmnearwell/examples/hysteresis_models/Killough/CO2_KILLOUGH", "GW")
krnw, krw, SandSmax = PreprocessData(SS, '/data/arkivdata/KrPcData_Combined050.csv', "GW")
krnw_val, krw_val, SandSmax_val = PreprocessData(SS, '/data/arkivdata/KrPcData_Swi0_50.csv', "GW")

modelnonwett, history, X_test, y_test = trainNN(krnw, SandSmax)


export_model(modelnonwett, 'model/oldmodelkrnw.model')

# Create an output folder if it doesn't exist
output_folder = "output_figures"
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(30, 6))

# Subplot 1: Actual vs Predicted Values
plt.subplot(1, 6, 1)
y_pred = modelnonwett.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.6, color='green', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)
plt.title('Final Model: Actual vs Predicted', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# Subplot 2: Loss Function Plot
plt.subplot(1, 6, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss Function Over Epochs', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# # Subplot 3: Drainage and Imbibition Curves
# plt.subplot(1, 6, 3)
# satspace = np.linspace(0.0e-1, 1.0e0, 100)
# satMaxspace = np.linspace(1.0e-1, 9.1e-1, 7)
# # satMaxspace = np.linspace(6.0e-1, 6.0e-1, 1)
# imbibition_label_added = False
# drainage_label_added = False

# for valsat in satspace:
#     for valsatmax in satMaxspace:
#         # xhat = np.array([[valsat, valsatmax, 0.0, 1.0]])  # Assuming the last two values are zero for simplicity
#         # yhatkrnw = modelnonwett.predict(xhat)
#         if valsatmax > valsat:
#             if not imbibition_label_added:
#                 xhat = np.array([[valsat, valsatmax, 0.06, 1.0]])  # Assuming the last two values are zero for simplicity
#                 yhatkrnw = modelnonwett.predict(xhat)
#                 plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue", label="Imbibition")
#                 imbibition_label_added = True
#             else:
#                 xhat = np.array([[valsat, valsatmax, 0.06, 1.0]])  # Assuming the last two values are zero for simplicity
#                 yhatkrnw = modelnonwett.predict(xhat)
#                 plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue")
#         if valsatmax < valsat:
#             if not drainage_label_added:
#                 xhat = np.array([[valsat, valsatmax, 0.06, 0.0]])  # Assuming the last two values are zero for simplicity
#                 yhatkrnw = modelnonwett.predict(xhat)
#                 plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red", label="Drainage")
#                 drainage_label_added = True
#             else:
#                 xhat = np.array([[valsat, valsatmax, 0.06, 0.0]])  # Assuming the last two values are zero for simplicity
#                 yhatkrnw = modelnonwett.predict(xhat)
#                 plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red")

# plt.xlabel(r"$S_w$", fontsize=14)
# plt.ylabel(r"$k_{rn}$", fontsize=14)
# plt.title('Drainage and Imbibition Curves', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True)

# Subplot 4: DNN vs Original Data (Drainage + Imbibition)
plt.subplot(1, 6, 4)
x = np.array(SandSmax)
y = np.array(krnw)
y_pred_all = modelnonwett.predict(x)

index_ranges_drain = [(0, 96), (97, 166), (167, 181), (182, 214), (215, 269), (270, 338)]

for i, (start, end) in enumerate(index_ranges_drain):
    plt.scatter(x[start:end, 0], y[start:end], alpha=0.6, color='pink',marker="*", label='Original Drain' if i == 0 else "")
    plt.scatter(x[start:end, 0], y_pred_all[start:end], alpha=0.6, color='black', marker="*", label='DNN Drain' if i == 0 else "")


plt.xlabel(r"$S_w$", fontsize=14)
plt.ylabel(r"$k_{rn}$", fontsize=14)
plt.title('DNN vs Original krnw Data Drainage', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# Subplot 5: Zoomed View (Optional)
plt.subplot(1, 6, 5)

index_ranges_imbib = [(339, 395), (396, 432), (433, 450)]

for i, (start, end) in enumerate(index_ranges_imbib):
    plt.scatter(x[start:end, 0], y[start:end], alpha=0.6, color='cyan', label='Original Imbib' if i == 0 else "")
    plt.scatter(x[start:end, 0], y_pred_all[start:end], alpha=0.6, color='black', marker="o", label='DNN Imbib' if i == 0 else "")

plt.xlabel(r"$S_w$", fontsize=14)
plt.ylabel(r"$k_{rn}$", fontsize=14)
plt.title('Imbibition View', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)


# Load validation dataset
# krnw_val, krw_val, SandSmax_val = PreprocessData(SS, '/data/arkivdata/KrPcData_Swi0_20.csv', "GW")
X_val = np.array(SandSmax_val)
y_val = np.array(krnw_val)

# Predict on validation data
y_val_pred = modelnonwett.predict(X_val)

# Subplot 6: Actual vs Predicted Values (Validation Set)
plt.subplot(1, 6, 6)
plt.scatter(y_val, y_val_pred, alpha=0.6, color='purple', label='Predicted vs Actual')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)
plt.title('Validation Set: Actual vs Predicted', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)



# Save the figure
output_path = os.path.join(output_folder, "final_model_plots_with_comparison.png")
plt.tight_layout()
plt.savefig(output_path)
# plt.show()
