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

    with open(path+'/data/sat.csv', 'w', newline='') as file:
        for S in Sh:
            smax = .0
            for s in S:
                smax = max(s,smax)
                file.write(str(1-s)+"\n")
                SandSmax.append([s, smax])

    pathCall = "/Users/macbookn/activopmwkspc/edgedev/build/opm-common/bin/hysteresis " + data +".DATA data/sat.csv data/relperms.csv " + wp + " 0"
    os.system(pathCall)

    krw = []
    krnw = []
    krM = []
    sT = []
    S2 = []

    with open(path+'/data/relperms.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            S2.append(1-float(row[0]))
            krnw.append(float(row[1]))
            krw.append(float(row[2]))
            krM.append(float(row[3]))
            sT.append([float(row[4])])

    i = 0
    start = 0
    end = 0
    # plt.plot(1-S, krw[0:200], label = "KRNW0")
    # plt.plot(1-S, krw[100:200], label = "KRNW1")
    # plt.plot(S, krw[200:300], label = "KRNW2")
    # plt.plot(S, krw[300:400], label = "KRNW3")
    # plt.plot(S, krw[400:500],label = "KRNW4")
    # plt.plot(S, krw[500:600],label = "KRNW4")
    # plt.plot(S, krw[600:700],label = "KRNW6")
    # plt.plot(1-S, krw[700:800],label = "KRNW7")
    # plt.plot(1-S, krw[1300:1400],label = "KRNW8")
    for S in Sh:
        end = len(S) + start

        input = "D" + str(i)
        if S[0] > S[-1]:
            input = "I" + str(i)
            i = i + 1
        # plt.plot(S, krnw[start:end],label = "KRW"+input)
        plt.plot(S, krw[start:end],label = "KRNW"+input)

        print(start)
        print(end)
        # plt.plot(S, krnw[600:700], label = "KRNW"+input)
        plt.legend(fontsize=12)

        start = end
        
        # plt.show()

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

    # Generate synthetic data
    synthetic_x, synthetic_y = generate_synthetic_data(x, y, num_samples=len(x) * 2)

    # Combine original and synthetic data
    x_combined = np.vstack((x, synthetic_x))
    y_combined = np.hstack((y, synthetic_y))

    # X_train, X_test, y_train, y_test = train_test_split(x_combined, y_combined, test_size=0.2, random_state=42)


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # feature_ranges: list[tuple[float, float]] = [(0.0, 1.0), (-3.7, 0.0)]

    data: np.ndarray = np.random.uniform(0, 1, X_train.shape[1])
    data_min = 0.0
    data_max = 1.0
    # print(max(X_train[:,0]))
    model = Sequential()
    model.add(keras.layers.Input([X_train.shape[1]]))
    # model.add(MinMaxScalerLayer(data_min = 0.0, data_max = 1.0))
    # model.add(Dense(4, activation='tanh'))
    # model.add(Dense(4, activation='tanh'))
    # model.add(Dense(4, activation='tanh'))
    # model.add(Dense(4, activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    # model.add(MinMaxUnScalerLayer(data_min = 0.0,data_max = 1.0))
    # # # #
    # model.get_layer(model.layers[0].name).adapt(data=data)
    # model.get_layer(model.layers[-1].name).adapt(data=data)

    optimizer = Adam(learning_rate=0.01)



    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True,start_from_epoch=500,)
    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # # ft the model on the training dataset
    history = model.fit(X_train, y_train, epochs=1550, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
    # make predictions for the input data
    # model.save("models/trainNNhyst.keras")


    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    # print(f'Test loss: {loss}')

    y_pred = model.predict(X_test)
    # print('MSE: %.3f' % mean_squared_error(Make, y))

    # pltbis.plot(history.history['loss'])
    # pltbis.plot(history.history['val_loss'])



    # Create an output folder if it doesn't exist
    # output_folder = "output_figures"
    # os.makedirs(output_folder, exist_ok=True)

    # output_path = os.path.join(output_folder, "loss.png")


    # pltbis.savefig(output_path)

    return model, history, X_test, y_test


# Step 3: Plot results
def plot_results(model, history, X_test, y_test, X, y):
    os.makedirs("output_figures", exist_ok=True)
    plt.figure(figsize=(24, 6))

    # Subplot 1: Actual vs Predicted
    plt.subplot(1, 5, 1)
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.6, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)

    # Subplot 2: Loss curve
    plt.subplot(1, 5, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Subplot 3: Drainage and Imbibition
    plt.subplot(1, 5, 3)
    for smax in np.linspace(0.2, 0.9, 5):
        sw_vals = np.linspace(0.05, smax, 50)
        xhat = np.array([[sw, smax] for sw in sw_vals])
        yhat = model.predict(xhat)
        plt.plot(sw_vals, yhat, label=f'Sw_max={smax:.2f}')
    plt.title("Drainage Curves")
    plt.xlabel("Sn")
    plt.ylabel("krnw")
    plt.legend()
    plt.grid(True)


    plt.subplot(1, 5, 4)
    satspace = np.linspace(0.0e-1, 1.0e0, 100)
    satMaxspace = np.linspace(1.0e-1, 9.1e-1, 7)
    # satMaxspace = np.linspace(2.0e-1, 2.0e-1, 1)
    imbibition_label_added = False
    drainage_label_added = False

    for valsat in satspace:
        for valsatmax in satMaxspace:
            xhat = np.array([[valsat, valsatmax]])
            yhatkrnw = model.predict(xhat)
            if valsatmax > valsat:
                if not imbibition_label_added:
                    plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue", label="Imbibition")
                    imbibition_label_added = True
                else:
                    plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue")
            if valsatmax < valsat:
                if not drainage_label_added:
                    plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red", label="Drainage")
                    drainage_label_added = True
                else:
                    plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red")

    plt.xlabel(r"$S_n$", fontsize=14)
    plt.ylabel(r"$k_{rn}$", fontsize=14)
    plt.title('Drainage and Imbibition Curves', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Subplot 4: DNN vs Original (Drainage)
    plt.subplot(1, 5, 5)
    X = np.array(X)
    # y = np.array(y)
    y_pred_all = model.predict(X)
    plt.scatter(X[:, 0], y,  label='Original', color='blue')
    plt.scatter(X[:, 0], y_pred_all, label='DNN', color='black')
    plt.title("DNN vs Original krnw")
    plt.xlabel("Sn")
    plt.ylabel("krnw")
    plt.legend()
    plt.grid(True)

    # # Subplot 5: Zoomed View
    # plt.subplot(1, 5, 5)
    # mask = (X[:, 1] > 0.5) & (X[:, 1] < 0.6)
    # plt.scatter(X[mask, 0], y[mask], alpha=0.4, label='Original', color='cyan')
    # plt.scatter(X[mask, 0], y_pred_all[mask], alpha=0.4, label='DNN', color='black')
    # plt.title("Zoomed View (Sw_max ~ 0.55)")
    # plt.xlabel("Sn")
    # plt.ylabel("krnw")
    # plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.savefig("output_figures/final_model_plots_with_comparison.png")
    print("Plot saved to output_figures/final_model_plots_with_comparison.png")






swl = 0.05
path = os.getcwd()
S = np.linspace( 0,1.0-swl, 7)
smax = 0.
S2 = 1.0 - S - swl;

SS = []


for smax in S:
    SS.append(np.linspace(0,smax, 100))

# krnw, krw, SandSmax = PreprocessData(SS, "/Users/macbookn/activopmwkspc/pyDavid/pyopmnearwell/examples/hysteresis_models/Killough/CO2_KILLOUGH", "GW")
krnw, krw, SandSmax = PreprocessData(SS, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")
# krnw, krw, SandSmax = PreprocessData(SS, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/testhyst/CO2", "GW")


# model, history, X_test, y_test = trainNN(krnw, SandSmax)
# plot_results(model, history, X_test, y_test, SandSmax, krnw)
#
# satspace = np.linspace(0.1, 0.95, 30)

# satMaxspace = np.linspace(0.5, 0.8, 2)

# for valsatmax in satMaxspace:
#     for valsat in satspace:
#         if valsatmax>valsat:
#             xhat = np.array([[valsat,valsatmax]])
#             yhatkrnw = model.predict(xhat)



# Create an output folder if it doesn't exist
model_folder = "models"
os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, "oldmodelkrnw.model")


# export_model(model, model_path)

plt.savefig("newKillough.png")

# satspace = np.linspace(0., 0.95, 100)

# satMaxspace = np.linspace(0.5, 0.9, 20)

# for valsatmax in satMaxspace:
#     for valsat in satspace:
#         if valsatmax>valsat:
#             xhat = np.array([[valsat,valsatmax]])

#             # oldmodelkrnw = keras.models.load_model(model)
#             # oldmodelkrw = keras.models.load_model(model)

#             yhatkrnw = model.predict(xhat)
#             # yhat2krw = oldmodelkrw.predict(xhat)

#             pyplot.plot(xhat.flat[0],  yhatkrnw,marker="o", markersize=5, markeredgecolor="red",label="Predictedkrnw")
#             # pyplot.plot(xhat.flat[0],  yhat2krw,marker="*", markersize=5, markeredgecolor="blue",label="Predictedkrw")

