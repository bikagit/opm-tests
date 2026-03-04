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
    # model = Sequential()
    # model.add(keras.layers.Input([input_dim]))
    # for i in range(1, layers):
    #     model.add(Dense(neurons[i], activation='tanh'))
    # model.add(Dense(1, activation='tanh'))  # Output layer for regression
    # # model.compile(optimizer=optimizer, loss='mean_squared_error')
    model = Sequential()

    model.add(Dense(8, activation='tanh', input_dim=input_dim))
    model.add(Dense(16, activation='tanh'))
    # model.add(Dense(16, activation='tanh'))
    # # model.add(Dense(16, activation='tanh'))
    # # model.add(Dense(16, activation='tanh'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(64, activation='tanh'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    # Compile the model with AdaGrad optimizer
    # learning_rate = 0.01
    optimizer = Adam()
    model.compile(optimizer=optimizer,
                loss='mean_squared_error',)

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


def Preprocess_analytical_Model_Data(Sh, data, wp):
    # smax = .0
    krnw = []
    krw = []
    SandSmax = []

    for S in Sh:
       smax = .0
       for s in S:
            smax = max(s,smax)
            pathCall = "/Users/macbookn/hackatonwork/build/opm-common/bin/hysteresis " + data +".DATA sat.csv relperms.csv " + wp + " 0"
            x = os.system(pathCall)
            # SandSmax.append([s, smax])

    import csv
    from pathlib import Path

    out_path = Path(path) / 'satsat.csv'

    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=',', lineterminator='\n')
        # w.writerow(['value', 'smax'])

        for S in Sh:
            if len(S) == 0:            # safe for Python lists
                continue
            smax = max(S)              # safe when S is a flat list of scalars
            for s in S:
                w.writerow([s, smax])

    # with open(path+'/satsat.csv', 'w', newline='') as file:
    #     writer = csv.writer(file, delimiter=";", lineterminator="\n")


        # for S in Sh:
        #     smax = .0        
        #     for s in S: 
        #         smax = max(s,smax)      
        #         # file.write(str(s)+"\n")
        #         file.write(str(s)+"\n")
        #     file.write(str(smax)+"\n")
                # SandSmax.append([s])
            # file.write(str(s)+str(" blah")+str(smax)+"\n")
            # SandSmax1.append([SandSmax,smax])

    # with open("my_file.csv", "w") as f:
        # writer = csv.writer(f, delimiter=";", lineterminator="\n")
        # writer.writerow(field)
        # writer.writerows(data)

    with open(path+'/relperms.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:   
            krnw.append(float(row[2]))
            krw.append(float(row[1]))                  
            SandSmax.append([1-float(row[0]),float(row[3])])


    # with open(path+'/satsat.csv', newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',')
    #     for row in reader:
    #         SandSmax.append([1-float(row[0]),1-float(row[1])])


    # i = 0
    # start = 0
    # end = 0
    # for S in Sh:
    #     end = len(S) + start
    #     input = "D" + str(i)
    #     if S[0] > S[-1]:
    #         input = "I" + str(i)
    #         i = i + 1
        
    #     # plt.plot(S, krw[start:end], label = "KRW"+input)
    #     # plt.plot(S, krnw[start:end], label = "KRNW"+input)
    #     start = end
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
    synthetic_x, synthetic_y = generate_synthetic_data(x_resampled, y_resampled, num_samples=len(x) * 35)
    x_combined = np.vstack((x, synthetic_x))
    y_combined = np.hstack((y, synthetic_y))
    X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.3, random_state=42)
    X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # sensitive to learning rate 0.01 is default, 0.005 is better for this data, 0.001 is too tight 
    model = create_model(X_train.shape[1], 10, [30]*10, Adam(learning_rate=0.005))
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, start_from_epoch=100)
    history = model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stopping])
    return model, history, X_test, y_test, x_val, y_val


swl = 0.05
path = os.getcwd()
S = np.linspace(0.0, 0.80, 50)
smax = 0.
S2 = 1.0 - S - swl

# SS = []

# for smax in S:
#     SS.append(np.linspace(smax, 0, 100))

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

SatSpace.append(np.linspace(0.41,0.78,30))
SatSpace.append(np.linspace(0.35,0.78,30))

SatSpace.append(np.linspace(0.25,0.78,30))
SatSpace.append(np.linspace(0.0618,0.78,30))

SSI = []
for smax in S:
    SSI.append(np.linspace(1-smax,1,10))

# SSI.append(np.linspace(0.41,0.78,30))
# SSI.append(np.linspace(0.35,0.78,30))

# SSI.append(np.linspace(0.25,0.78,30))
# SSI.append(np.linspace(0.0618,0.78,30))

evaluate(SSI, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P-1D-OILINJMULTI", "WO")

train_file = '/data/curated_data/KrPcData_Combinedtot.csv'

krnw, krw, SandSmax = PreprocessData(SSI, train_file, "GW")

original_krnw, original_krw, original_SandSmax = PreprocessData(SSI, '/data/curated_data/KrPcData_Swi0_06-CORRECTED.csv', "GW")


krnw_modl, krw_modl, SandSmax_modl = Preprocess_analytical_Model_Data(SSI, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P-1D-OILINJMULTI", "WO")

print(len(krnw_modl))
print(len(SandSmax_modl))


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

S7rev = np.linspace(0.41,0.78,10)
S8rev = np.linspace(0.35,0.78,10)
# y_pred_all = modelnonwett.predict(x)
# Predictions
y_pred_test = modelnonwett.predict(X_test)
# y_pred_val = modelnonwett.predict([0.3,0.4])
y_pred_all = modelnonwett.predict(x)


satspace = np.linspace(0.0, 0.8, 100)
satMaxspace = np.linspace(0.0, 0.8, 30)

for valsat in satspace:
    for valsatmax in satMaxspace:
        xhatbis = np.array([[valsat, valsatmax]])
        yhatkrnw = modelnonwett.predict(xhatbis)
        if valsatmax > valsat:
            plt.plot(xhatbis.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue", label="Imbibition")

# for i, (start, end) in enumerate(index_ranges_imbib):
#     # plt.scatter(original_x[start:end, 0], original_y[start:end], alpha=0.6, color='cyan', label='Original Imbib' if i == 0 else "")
#     # plt.scatter(x[start:end, 0], y_pred_all[start:end], alpha=0.6, color='black', marker="o", label='DNN Imbib' if i == 0 else "")
#     plt.scatter(xhatbis[start:end, 0], yhatkrnw[start:end], alpha=0.6, color='red', label='Original Imbib' if i == 0 else "")
    # plt.scatter(X_test[start:end, 0], y_pred_test[start:end], alpha=0.6, color='red', label='Original Imbib' if i == 0 else "")

plt.xlabel(r"$S_n$", fontsize=14)
plt.ylabel(r"$k_{rn}$", fontsize=14)
plt.title('Imbibition View', fontsize=16)
# plt.legend(fontsize=12)
plt.grid(True)



#evaluate([S1,S2,S3,S4], "1D_3PHASE_KILLOUGH_BOTH", "W")

# Save the figure
output_path = os.path.join(output_folder, "comparingwithKillough.png")
plt.tight_layout()
plt.savefig(output_path)
# plt.show()
