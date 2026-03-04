
# Revised script: export each plot as its own PNG
# File: killough-data_vs_ML-Killough_separate_png.py

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample

import keras
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax, Adagrad, Adadelta

# Custom tools from your environment
sys.path.insert(0, '/Users/macbookn/activopmwkspc/edgedev/opm-common/python/opm/ml')
from ml_tools import export_model
from ml_tools import MinMaxScalerLayer, MinMaxUnScalerLayer

keras.utils.set_random_seed(1234)

path = os.getcwd()

# -----------------------------
# Helper functions
# -----------------------------
def evaluate(Sh, data, wp, labelfig=None):
    """Calls external hysteresis tool and plots relperm curves for each saturation path in Sh.
    Sh: list of arrays (saturation path sets)
    data: deck path (without .DATA suffix)
    wp: 'WO' or 'GW'
    labelfig: optional label for plotted curves
    """
    with open(os.path.join(path, 'sat.csv'), 'w', newline='') as file:
        for S in Sh:
            for s in S:
                file.write(str(s) + "\n")

    # External call (expects your environment to have the binary)
    pathCall = (
        "/Users/macbookn/activopmwkspc/edgedev/build/opm-common/bin/hysteresis "
        + data + ".DATA sat.csv relperms.csv " + wp + " 0"
    )
    os.system(pathCall)

    krw, krnw, krM, sT = [], [], [], []
    with open(os.path.join(path, 'relperms.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            krnw.append(float(row[2]))
            krw.append(float(row[1]))
            krM.append(float(row[3]))
            sT.append(float(row[4]))

    start = 0
    for S in Sh:
        end = len(S) + start
        # Plot kr_n vs 1 - S
        if labelfig is not None:
            plt.plot( S, krnw[start:end], label=labelfig)
        else:
            plt.plot( S, krnw[start:end])
        start = end


def create_model(input_dim, layers, neurons, optimizer):
    """Create a simple feed-forward model with given layers and neurons."""
    model = Sequential()
    model.add(keras.layers.Input([input_dim]))
    for i in range(1, layers):
        model.add(Dense(neurons[i], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output for regression (bounded [0,1])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def newPreprocessData(Sh, data, data2, wp):
    SandSmax = []
    krw = []
    krnw = []
    # pc = []
    # S2 = []
    # swi = []

    # with open(path+data, newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',')
    #     next(reader)
    #     for row in reader:
    #         # S2.append(1-float(row[3]))
    #         krnw.append(float(row[4]))
    #         # swi.append(float(row[7]))
    #         krw.append(float(row[4]))
    #         # pc.append(float(row[4]))
    #         # SandSmax.append([1.0-float(row[3]),1.0-float(row[2]),float(row[7]),float(row[0])])
    #         SandSmax.append([1.0-float(row[3]),1.0-float(row[9])])
    
    with open(path+'/satanalyt.csv', 'w', newline='') as file:
        for S in Sh:        
            for s in S:        
                file.write(str(s)+"\n")
    for S in Sh:
       smax = .0
       for s in S:
            smax = max(s,smax)
            pathCall = "/Users/macbookn/hackatonwork/build/opm-common/bin/hysteresis " + data2 +".DATA satanalyt.csv relperms.csv " + wp + " 0"
            x = os.system(pathCall)

    with open(path+'/relperms.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:   
            krnw.append(float(row[2]))
            krw.append(float(row[1]))                  
            SandSmax.append([1-float(row[0]),float(row[3])])

    return krnw, krw, SandSmax


def PreprocessData(Sh, data, wp):
    SandSmax, krw, krnw = [], [], []
    pc, S2, swi = [], [], []
    with open(path + data, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            S2.append(1 - float(row[3]))
            krnw.append(float(row[6]))
            swi.append(float(row[7]))
            krw.append(float(row[5]))
            pc.append(float(row[4]))
            SandSmax.append([1.0 - float(row[3]), 1.0 - float(row[2])])
    return krnw, krw, SandSmax


def Preprocess_analytical_Model_Data(Sh, data, wp):
    krnw, krw, SandSmax = [], [], []
    with open(os.path.join(path, 'satanalyt.csv'), 'w', newline='') as file:
        for S in Sh:
            for s in S:
                file.write(str(s) + "\n")

    for S in Sh:
        smax = 0.0
        for s in S:
            smax = max(s, smax)
        pathCall = (
            "/Users/macbookn/hackatonwork/build/opm-common/bin/hysteresis "
            + data + ".DATA satanalyt.csv relperms.csv " + wp + " 0"
        )
        os.system(pathCall)
        with open(os.path.join(path, 'relperms.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                krnw.append(float(row[2]))
                krw.append(float(row[1]))
                SandSmax.append([1 - float(row[0]), float(row[3])])
    return krnw, krw, SandSmax


def generate_synthetic_data(x, y, num_samples):
    synthetic_x, synthetic_y = [], []
    for _ in range(num_samples):
        idx = np.random.randint(0, len(x))
        noise = np.random.normal(0, 0.01, x.shape[1])
        synthetic_x.append(x[idx] + noise)
        synthetic_y.append(y[idx] + noise[0])  # y is 1D
    return np.array(synthetic_x), np.array(synthetic_y)


def trainNN(krnw, Smax):
    x = np.array(Smax)
    y = np.array(krnw)

    x_resampled, y_resampled = resample(x, y, n_samples=len(x) * 30, random_state=42)
    synthetic_x, synthetic_y = generate_synthetic_data(x_resampled, y_resampled, num_samples=len(x) * 30)
    x_combined = np.vstack((x, synthetic_x))
    y_combined = np.hstack((y, synthetic_y))

    X_train, X_test, y_train, y_test = train_test_split(x_combined, y_combined, test_size=0.3, random_state=42)
    X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Sensitive to learning rate: 0.01 default
    model = create_model(X_train.shape[1], 10, [9] * 10, Adam(learning_rate=0.01))
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, start_from_epoch=100)
    history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])
    return model, history, X_test, y_test, x_val, y_val


# -----------------------------
# Main workflow
# -----------------------------
if __name__ == "__main__":
    swl = 0.05
    S = np.linspace(0.0, 1.0, 100)

    # Build SSI paths
    SSI = []
    for smax in S:
        SSI.append(np.linspace(1 - smax, 1, 10))

    # Data paths (as in your original script)
    newtrain_file = '/runKillough.csv'
    train_file = '/../../data/curated_data/KrPcData_Combinedtot.csv'

    # Load data
    krnw, krw, SandSmax = PreprocessData(SSI, train_file, "GW")
    newkrnw, newkrw, newSandSmax = newPreprocessData(SSI, newtrain_file,
        "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/CASE-A/test006/CORE_ExampleKillough", "WO")
    original_krnw, original_krw, original_SandSmax = PreprocessData(SSI, '/../../data/curated_data/KrPcData_Swi0_06-CORR-OCT2025.csv', "GW")
    krnw_modl, krw_modl, SandSmax_modl = Preprocess_analytical_Model_Data(SSI, 
        "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/CASE-A/test006/CORE_ExampleKillough", "WO")

    # Train model on new data
    modelnonwett, history, X_test, y_test, x_val, y_val = trainNN(newkrnw, newSandSmax)

    # Export model
    # os.makedirs('model', exist_ok=True)
    # export_model(modelnonwett, 'model/oldmodelkrnwCORE_ExampleKillough.model')

    # # Predictions
    # x_all = np.array(SandSmax)
    # y_pred_test = modelnonwett.predict(X_test)
    # y_pred_val = modelnonwett.predict(x_val)
    # y_pred_all = modelnonwett.predict(x_all)

    # Output folder
    output_folder = os.path.join(path, 'output_figures')
    os.makedirs(output_folder, exist_ok=True)

    # -----------------------------
    # 1) Test parity plot (single PNG)
    # # -----------------------------
    # plt.figure(figsize=(6, 5))
    # plt.scatter(y_test, y_pred_test, alpha=0.6, color='green')
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    # plt.title('Test Set: Actual vs Predicted')
    # plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, 'test_parity.png'), dpi=300)
    # plt.close()

    # -----------------------------
    # 2) Drainage & Imbibition curves (single PNG)
    # -----------------------------
    plt.figure(figsize=(7, 5))


    # Killough reference curves via evaluate()
    maincurves0 = [np.linspace(1.0, 0.0, 3000)]
    maincurves1 = [np.linspace(0.0, 1.0, 3000)]
    evaluate(maincurves0, 
             "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/CASE-A/test006/CORE_ExampleKillough", 
             "WO", labelfig="Main drainage")
    evaluate(maincurves1, 
             "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/CASE-A/test006/CORE_ExampleKillough", 
             "WO", labelfig="Main imbibition")

    SatSpace = []
    # SatSpace.append(np.linspace(0.75, 1, 100))
    # SatSpace.append(np.linspace(0.7, 1, 100))
    # SatSpace.append(np.linspace(0.65, 1, 100))
    # SatSpace.append(np.linspace(0.6, 1, 3000))
    SatSpace.append(np.linspace(0.5, 1, 3000))
    SatSpace.append(np.linspace(0.45, 1, 3000))
    SatSpace.append(np.linspace(0.4, 1, 3000))
    # SatSpace.append(np.linspace(0.25, 1, 3000))


    evaluate(SatSpace,
             "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/CASE-A/test006/CORE_ExampleKillough",
             "WO", labelfig="Imbib. Killough scan curve")



    # ML imbibition scan curves
    satspace = np.linspace(0.0, 1.0, 100)
    for valsatmax in { 0.6, 0.55, 0.5}:
        x_vals, y_vals = [], []
        for valsat in satspace:
            if valsatmax > valsat:
                yhat = modelnonwett.predict(np.array([[valsat, valsatmax]]))
                x_vals.append(1-valsat)
                y_vals.append(yhat[0])
        if x_vals and y_vals:
            plt.plot(x_vals, y_vals,alpha=0.6, color='black', marker='*',
                     label='ML-Killough Imbib. scan curves' if valsatmax == 0.5 else '')


    # plt.title('Drainage and Imbibition')
    plt.xlabel('Saturation ' + r"($S_w$)", fontsize=14)
    plt.ylabel('Rel. permeability ' + r"($k_{rn}$)", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'drainage_imbibition.png'), dpi=300)
    plt.close()

    # -----------------------------
    # 3) Validation parity plot (single PNG)
    # -----------------------------
    # plt.figure(figsize=(6, 5))
    # plt.scatter(y_val, y_pred_val, alpha=0.6, color='purple')
    # plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    # plt.title('Validation Set: Actual vs Predicted')
    # plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, 'validation_parity.png'), dpi=300)
    # plt.close()

    # -----------------------------
    # Metrics
    # -----------------------------
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    # mae = mean_absolute_error(y_test, y_pred_test)
    # r2 = r2_score(y_test, y_pred_test)
    # endpoint_error_start = abs(y_test[0] - y_pred_test[0])
    # endpoint_error_end = abs(y_test[-1] - y_pred_test[-1])

    # print("RMSE:", rmse)
    # print("MAE:", mae)
    # print("R² Score:", r2)
    # print("Endpoint Error (Start):", endpoint_error_start)
    # print("Endpoint Error (End):", endpoint_error_end)

    print("Saved PNGs:")
    print(" -", os.path.join(output_folder, 'test_parity.png'))
    print(" -", os.path.join(output_folder, 'drainage_imbibition.png'))
