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



def evaluate(Sh, data, wp, labelfig=None):

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
            krnw.append(float(row[2]))
            krw.append(float(row[1]))
            krM.append(float(row[3]))
            sT.append(float(row[4]))

    i = 0
    start = 0
    end = 0
    first = True

    for S in Sh:
        end = len(S) + start
        # input = "D" + str(i)
        # if 1-S[0] < 1-S[-1]:
        input = "I" + str(1-i)
        i = i + 1
       
        if labelfig is not None:
            # plt.plot(1-S, krnw[start:end],color="blue", label="Carlson model")
            plt.plot(1-S, krnw[start:end], label = labelfig)
        else:
            plt.plot(1-S, krnw[start:end])
        start = end



# Function to create a model with a specified number of layers and neurons
def create_model(input_dim, layers, neurons, optimizer):
    model = Sequential()
    model.add(keras.layers.Input([input_dim]))
    # model.add(Dense(neurons[0], input_dim=input_dim, activation='relu'))
    # model.add(Dense(8, activation='tanh'))

    for i in range(1, layers):
        model.add(Dense(neurons[i], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for regression
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def PreprocessData(Sh, data, wp):
    SandSmax = []
    krw = []
    krnw = []
    pc = []
    S2 = []
    swi = []

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


def Preprocess_analytical_Model_Data(Sh, data, wp):
    # smax = .0
    krnw = []
    krw = []
    SandSmax = []

    with open(path+'/satanalyt.csv', 'w', newline='') as file:
        for S in Sh:        
            for s in S:        
                file.write(str(s)+"\n")
    for S in Sh:
       smax = .0
       for s in S:
            smax = max(s,smax)
            pathCall = "/Users/macbookn/hackatonwork/build/opm-common/bin/hysteresis " + data +".DATA satanalyt.csv relperms.csv " + wp + " 0"
            x = os.system(pathCall)

    with open(path+'/relperms.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:   
            krnw.append(float(row[2]))
            krw.append(float(row[1]))                  
            SandSmax.append([1-float(row[0]),float(row[3])])
    return krnw,krw, SandSmax


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

    x_resampled, y_resampled = resample(x, y, n_samples=len(x) * 30, random_state=42)
    synthetic_x, synthetic_y = generate_synthetic_data(x_resampled, y_resampled, num_samples=len(x) * 30)
    x_combined = np.vstack((x, synthetic_x))
    y_combined = np.hstack((y, synthetic_y))
    X_train, X_test, y_train, y_test = train_test_split(x_combined, y_combined, test_size=0.3, random_state=42)
    X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # sensitive to learning rate 0.01 is default, 0.005 is better for this data, 0.001 is too tight 
    model = create_model(X_train.shape[1], 10, [9]*10, Adam(learning_rate=0.01))
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, start_from_epoch=0)
    history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])
    return model, history, X_test, y_test, x_val, y_val


swl = 0.05
path = os.getcwd()
S = np.linspace(0.0, 1.0, 100)
smax = 0.
S2 = 1.0 - S - swl
SatSpace = []
SS = []

for smax in S:
    SS.append(np.linspace(0,1-smax,4))

index_ranges_imbib = [
                    #     (99, 167),
                    #   (345, 398),
                    #   (399,435),
                    #   (436,452),
                    #   (453,475),
                    #   (400,490),
                      (0,1470),
                    #   (915,1055)
                      ]


# SatSpace.append(np.linspace(0.9,1,30))
SatSpace.append(np.linspace(0.75,1,100))
SatSpace.append(np.linspace(0.7,1,100))

SatSpace.append(np.linspace(0.65,1,100))

SatSpace.append(np.linspace(0.6,1,100))
SatSpace.append(np.linspace(0.5,1,100))

# SatSpace.append(np.linspace(0.4,1,100))
# SatSpace.append(np.linspace(0.35,1,30))

# SatSpace.append(np.linspace(0.2,1,100))
# SatSpace.append(np.linspace(0.,1,100))


# SatSpace.append(np.linspace(0.41,1,30))
# SatSpace.append(np.linspace(0.35,1,30))

# SatSpace.append(np.linspace(0.25,1,30))
# SatSpace.append(np.linspace(0.0618,1,30))

SSI = []
for smax in S:
    SSI.append(np.linspace(1-smax,1,10))


# evaluate(SSI, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/CORE_Example", "WO")

train_file = '/../../data/curated_data/KrPcData_Combinedtot.csv'

krnw, krw, SandSmax = PreprocessData(SSI, train_file, "GW")

original_krnw, original_krw, original_SandSmax = PreprocessData(SSI, '/../../data/curated_data/KrPcData_Swi0_06-CORR-OCT2025.csv', "GW")

krnw_modl, krw_modl, SandSmax_modl = Preprocess_analytical_Model_Data(SSI, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/CASE-A/test035/CORE_ExampleKillough", "WO")


modelnonwett, history, X_test, y_test, x_val, y_val = trainNN(krnw_modl, SandSmax_modl)


export_model(modelnonwett, 'model/oldmodelkrnwCORE_ExampleKillough.model')
# export_model(modelnonwett, 'model/oldmodelkrnw.model')

# Create an output folder if it doesn't exist
output_folder = "output_figures"
os.makedirs(output_folder, exist_ok=True)


x = np.array(SandSmax)
y = np.array(krnw)


original_x = np.array(original_SandSmax)
original_y = np.array(original_krnw)

# y_pred_all = modelnonwett.predict(x)
# Predictions
y_pred_test = modelnonwett.predict(X_test)
y_pred_val = modelnonwett.predict(x_val)
y_pred_all = modelnonwett.predict(x)


# satspace = np.linspace(0.0, 0.8, 100)
# satMaxspace = np.linspace(0.0, 0.8, 30)

# for valsat in satspace:
#     for valsatmax in satMaxspace:
#         xhatbis = np.array([[valsat, valsatmax]])
#         yhatkrnw = modelnonwett.predict(xhatbis)
#         if valsatmax > valsat:
#             plt.plot(xhatbis.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue", label="Imbibition")

# # for i, (start, end) in enumerate(index_ranges_imbib):
# #     # plt.scatter(original_x[start:end, 0], original_y[start:end], alpha=0.6, color='cyan', label='Original Imbib' if i == 0 else "")
# #     # plt.scatter(x[start:end, 0], y_pred_all[start:end], alpha=0.6, color='black', marker="o", label='DNN Imbib' if i == 0 else "")
# #     # plt.scatter(xhatbis[start:end, 0], yhatkrnw[start:end], alpha=0.6, color='red', label='Original Imbib' if i == 0 else "")
# #     plt.scatter(X_test[start:end, 0], y_pred_test[start:end], alpha=0.6, color='red', label='Original Imbib' if i == 0 else "")

# plt.xlabel(r"$S_n$", fontsize=14)
# plt.ylabel(r"$k_{rn}$", fontsize=14)
# plt.title('Imbibition View', fontsize=16)
# # plt.legend(fontsize=12)
# plt.grid(True)


# # Save the figure
# output_path = os.path.join(output_folder, "comparingwithCarlson.png")
# plt.tight_layout()
# plt.savefig(output_path)
# # plt.show()




# # Plotting
# plt.figure(figsize=(20, 6))

# # 1. Test Set
# plt.subplot(1, 3, 1)
# plt.scatter(y_test, y_pred_test, alpha=0.6, color='green')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.title('Test Set: Actual vs Predicted')
# plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.grid(True)

# # # 2. Loss
# # plt.subplot(1, 4, 2)
# # plt.plot(history.history['loss'], label='Train')
# # plt.plot(history.history['val_loss'], label='Val')
# # plt.title('Loss Over Epochs')
# # plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

# # 3. Drainage/Imbibition
# plt.subplot(1, 3, 2)
satspace = np.linspace(0.1, 1.0, 100)
# satMaxspace = np.linspace(0.1, 0.91, 7)
tag = 0
imbibition_label_added = False
drainage_label_added = False

satspace = np.linspace(0.0, 1.0, 100)
satMaxspace = np.linspace(0.3, 0.5, 3)

# for valsat in satspace:
#     for valsatmax in satMaxspace:
#         xhatbis = np.array([[valsat, valsatmax]])
#         yhatkrnw = modelnonwett.predict(xhatbis)
#         if valsatmax > valsat:
#             plt.plot(xhatbis.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue")

#     # plt.plot(label="Imbibition")


first = True


# Plotting lines instead of dots
for valsatmax in {0.25,0.3,0.35, 0.4, 0.5}:
    # valsatmax= 0.59
    x_vals = []
    y_vals = []
    for valsat in satspace:
        if valsatmax > valsat:
            y = modelnonwett.predict(np.array([[valsat, valsatmax]]))
            x_vals.append(valsat)
            y_vals.append(y[0])
    if x_vals and y_vals:
        plt.plot(x_vals, y_vals, color='black', marker="*", label="ML-Killough Imbib. scan curves" if valsatmax == 0.25 else "")


# # Plotting lines instead of dots
# for valsatmax in {0.01,0.25}:
#     # valsatmax= 0.59
#     x_vals = []
#     y_vals = []
#     for valsat in satspace:
#         if valsatmax < valsat:
#             y = modelnonwett.predict(np.array([[valsat, valsatmax]]))
#             x_vals.append(valsat)
#             y_vals.append(y[0])
#     if x_vals and y_vals:
#         plt.scatter(x_vals, y_vals, color='cyan', marker="*", label="DNN drainage. scan curves" if valsatmax == 0.25 else "")


# plt.xlabel("valsat")
# plt.ylabel("Prediction")
# plt.title("Model Predictions Over Saturation Space")
# plt.legend()
# plt.grid(True)
# plt.show()


maincurves0 = []
maincurves1 = []
maincurves2 = []
maincurves3 = []
maincurves4 = []

maincurves0.append(np.linspace(1.0,0.0,30))
maincurves1.append(np.linspace(0.0,1.0,30))

evaluate(maincurves0, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/CASE-A/test035/CORE_ExampleKillough", "WO",labelfig="Primary drainage")
evaluate(maincurves1, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/CASE-A/test035/CORE_ExampleKillough", "WO",labelfig="First imbibition")

evaluate(SatSpace, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/CASE-A/test035/CORE_ExampleKillough", "WO",labelfig="Imbib. Killough scan curve")


# plt.legend()
# plt.xlabel("sat")
# plt.ylabel("prediction")
# plt.tight_layout()
# plt.show()



# for valsat in satspace:
#     for valsatmax in satMaxspace:
#         xhat = np.array([[valsat, valsatmax]])
#         yhatkrnw = modelnonwett.predict(xhat)
#         if valsatmax > valsat:
#             if not imbibition_label_added:
#                 plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue", label="Imbibition")
#                 imbibition_label_added = True
#             else:
#                 plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue")
#         if valsatmax < valsat:
#             if not drainage_label_added:
#                 plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red", label="Drainage")
#                 drainage_label_added = True
#             else:
#                 plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red")

plt.plot(label='Drainage')
plt.title('Drainage and Imbibition')
plt.xlabel('Saturation 'r"$S_n$", fontsize=14)
plt.ylabel('Rel. permeability ' r"$(k_{rn})$", fontsize=14); plt.grid(True)
plt.legend(fontsize=11)

# # 4. DNN vs Original Drainage
# plt.subplot(1, 6, 4)
# plt.scatter(x_val[:, 0], y_val, color='pink', marker='*', label='Original')
# plt.scatter(x_val[:, 0], y_pred_val, color='black', marker='*', label='DNN')
# plt.title('DNN vs Original Drainage')
# plt.xlabel('$S_w$'); plt.ylabel('$k_{rn}$'); plt.legend(); plt.grid(True)

# # 5. DNN vs Original Imbibition
# plt.subplot(1,4, 3)
# plt.scatter(x_val[:, 0], y_val, color='cyan', label='Original')
# plt.scatter(x_val[:, 0], y_pred_val, color='black', marker='o', label='DNN')
# plt.title('DNN vs Original Imbibition')
# plt.xlabel('$S_n$'); plt.ylabel('$k_{rn}$'); plt.legend(); plt.grid(True)

# # 6. Validation Set
# plt.subplot(1, 3, 3)
# plt.scatter(y_val, y_pred_val, alpha=0.6, color='purple')
# plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
# plt.title('Validation Set: Actual vs Predicted')
# plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.grid(True)

# Save
os.makedirs("output_figures", exist_ok=True)
plt.tight_layout()
plt.savefig("output_figures/final_model_plots_with_comparisonKillough.png")

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import os
# import plotly.express as px

# # # Replace these with your actual model outputs
# # y_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5])         # Example actual values
# # y_pred_test = np.array([0.12, 0.19, 0.31, 0.39, 0.52])  # Example predicted values

# # Compute metrics
# rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
# mae = mean_absolute_error(y_test, y_pred_test)
# r2 = r2_score(y_test, y_pred_test)

# # Endpoint errors
# endpoint_error_start = abs(y_test[0] - y_pred_test[0])
# endpoint_error_end = abs(y_test[-1] - y_pred_test[-1])

# # # Parity plot
# # fig = px.scatter(x=y_test, y=y_pred_test, labels={'x': 'Actual Values', 'y': 'Predicted Values'},
# #                  title='Parity Regression Plot')
# # fig.add_shape(type='line', x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test),
# #               line=dict(color='red', dash='dash'))

# # # Save plot
# # os.makedirs("output_figures", exist_ok=True)
# # fig.write_image("output_figures/parity_plot.png")
# # fig.write_json("output_figures/parity_plot.json")

# # Print metrics
# print("RMSE:", rmse)
# print("MAE:", mae)
# print("R² Score:", r2)
# print("Endpoint Error (Start):", endpoint_error_start)
# print("Endpoint Error (End):", endpoint_error_end)
