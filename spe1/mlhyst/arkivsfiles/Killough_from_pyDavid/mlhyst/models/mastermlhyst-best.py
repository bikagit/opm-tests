import csv
import numpy as np
import os, sys
import matplotlib.pyplot as plt

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

keras.utils.set_random_seed(1234)


sys.path.insert(0, '/Users/macbookn/activopmwkspc/stable_releases/opm-common/python/opm/ml')

from ml_tools import export_model
from ml_tools import  MinMaxScalerLayer, MinMaxUnScalerLayer


# def PreprocessData(Sh, data, wp):
   
#     SandSmax = []

#     with open(path+'/sat.csv', 'w', newline='') as file:
#         for S in Sh:   
#             smax = .0     
#             for s in S:
#                 smax = max(s,smax)        
#                 file.write(str(s)+"\n")
#                 SandSmax.append([s, smax])

#     pathCall = "/Users/macbookn/hackatonwork/build/opm-common/bin/hysteresis " + data +".DATA sat.csv relperms.csv " + wp + " 0"
#     os.system(pathCall)

#     krw = []
#     krnw = []
#     krM = []
#     sT = []
#     S2 = []

#     with open(path+'/relperms.csv', newline='') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         for row in reader:
#             S2.append(float(row[0]))
#             krnw.append(float(row[1]))
#             krw.append(float(row[2]))
#             krM.append(float(row[3]))
#             sT.append([float(row[4])])

#     i = 0
#     start = 0
#     end = 0

#     for S in Sh:
#         end = len(S) + start

#         input = "D" + str(i)
#         if S[0] > S[-1]:
#             input = "I" + str(i)
#             i = i + 1
#         plt.plot(S, krnw[start:end], label = "KRNW"+input)
#         plt.plot(S, krw[start:end],label = "KRW"+input)

#         start = end

#     return krnw,krw, SandSmax

# def trainNN(krnw, Smax):
#     x = np.array(Smax)
#     y = np.array(krnw)
    
#     model = Sequential()
#     model.add(Dense(3, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(1))
#     # define the loss function and optimization algorithm
#     model.compile(loss='mse', optimizer='adam')
#     # # ft the model on the training dataset
#     model.fit(x, y, epochs=2000, batch_size=100, verbose=0)
#     # make predictions for the input data
#     model.save("models/trainNNhyst.keras")

#     return model


# def predictNN(Sh):
#     oldmodel = keras.models.load_model("models/trainNNhyst.keras")
#     xhat = np.array([Sh])
#     yhat = oldmodel.predict(xhat)
#     return yhat


# swl = 0.05
# path = os.getcwd()
# S = np.linspace(0.0, 1.0-swl, 30)
# smax = 0.
# S2 = 1.0 - S - swl;
# # evaluate([S], "CO2", "GW")

# SS = []
# # # SS.append(np.linspace(0.0, 1.0-swl, 100))


# for smax in S:
#     SS.append(np.linspace(smax,0, 100))

# krnw, krw, SandSmax = PreprocessData(SS, "CO2", "GW")


# trainNN(krnw, SandSmax)

# # # # # yhat = predictNN([0.3,0.5])

# satspace = np.linspace(0.1, 0.95, 30)

# satMaxspace = np.linspace(0.5, 0.8, 2)

# for valsatmax in satMaxspace:
#     for valsat in satspace:
#         if valsatmax>valsat:
#             xhat = np.array([[valsat,valsatmax]])

#             oldmodelkrnw = keras.models.load_model("models/finemodel.keras")
#             oldmodelkrw = keras.models.load_model("models/krwtrainNNhyst.keras")

#             yhatkrnw = oldmodelkrnw.predict(xhat)
#             yhat2krw = oldmodelkrw.predict(xhat)

#             pyplot.plot(xhat.flat[0],  yhatkrnw,marker="o", markersize=5, markeredgecolor="red",label="Predictedkrnw")
#             pyplot.plot(xhat.flat[0],  yhat2krw,marker="*", markersize=5, markeredgecolor="blue",label="Predictedkrw")

#             export_model(oldmodelkrnw, 'oldmodelkrnw.model')
#             export_model(oldmodelkrw, 'oldmodelkrw.model')

# plt.legend()
# #plt.savefig("CARLSON.png")
# plt.savefig("output/newKillough.png")
# plt.show()

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

    pathCall = "/Users/macbookn/hackatonwork/build/opm-common/bin/hysteresis " + data +".DATA sat.csv relperms.csv " + wp + " 0"
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
    
    # model = Sequential()
    # model.add(Dense(3, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(15, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(15, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(15, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(15, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(15, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(1))

    # # model = Sequential()
    # Build the MLP model

    # Standardize the data
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize the data
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # model = Sequential()
    # model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(1, activation='linear'))  # Output layer for regression

    # Build the MLP model

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='tanh'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # # ft the model on the training dataset
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)
    # make predictions for the input data
    # model.save("models/trainNNhyst.keras")
    

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')

    y_pred = model.predict(X_test)

    # print('MSE: %.3f' % mean_squared_error(Make, y))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.savefig("loss.png")

    return model


# def predictNN(Sh):
#     oldmodel = keras.models.load_model("models/trainNNhyst.keras")
#     xhat = np.array([Sh])
#     yhat = oldmodel.predict(xhat)
#     return yhat


swl = 0.05
path = os.getcwd()
S = np.linspace(0.0, 1.0-swl, 700)
smax = 0.
S2 = 1.0 - S - swl;
# evaluate([S], "CO2", "GW")

SS = []
# # SS.append(np.linspace(0.0, 1.0-swl, 100))


for smax in S:
    SS.append(np.linspace(smax,0, 100))

krnw, krw, SandSmax = PreprocessData(SS, "../CO2", "GW")


# modelwetting = trainNN(krw, SandSmax)
modelnonwett = trainNN(krnw, SandSmax)

# # # # yhat = predictNN([0.3,0.5])

satspace = np.linspace(0.1, 0.95, 30)

satMaxspace = np.linspace(0.5, 0.8, 2)

for valsatmax in satMaxspace:
    for valsat in satspace:
        if valsatmax>valsat:
            xhat = np.array([[valsat,valsatmax]])

            # oldmodelkrnw = keras.models.load_model("models/finemodel.keras")
            # oldmodelkrw = keras.models.load_model("models/krwtrainNNhyst.keras")

            # yhatkrnw = oldmodelkrnw.predict(xhat)
            # yhat2krw = oldmodelkrw.predict(xhat)

            yhatkrnw = modelnonwett.predict(xhat)
            # yhat2krw = modelwetting.predict(xhat)

            # pyplot.plot(xhat.flat[0],  yhatkrnw,marker="o", markersize=5, markeredgecolor="red",label="Predictedkrnw")
            # pyplot.plot(xhat.flat[0],  yhat2krw,marker="*", markersize=5, markeredgecolor="blue",label="Predictedkrw")

export_model(modelnonwett, 'oldmodelkrnw.model')
            # export_model(modelwetting, 'oldmodelkrw.model')

            # export_model(oldmodelkrnw, 'oldmodelkrnw.model')
            # export_model(oldmodelkrw, 'oldmodelkrw.model')

# plt.legend()
# #plt.savefig("CARLSON.png")
plt.savefig("output/newKillough.png")



# plt.show()


# # # plt.plot(history.history['loss'])
# # # plt.plot(history.history['val_loss'])
# # # plt.title('model loss')
# # # plt.ylabel('loss')
# # # plt.xlabel('epoch')
# # # plt.legend(['train', 'val'], loc='upper left')
# # # plt.show()

# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from tensorflow.keras.models import Sequential
# # # from tensorflow.keras.layers import Dense
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import StandardScaler



# # # Generate some synthetic data for demonstration purposes
# # # Replace this with your actual data
# # X = np.array(SandSmax)
# # y = np.array(krnw)


# # # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Standardize the data
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)

# # # Define different architectures and optimizers to test
# # architectures = [
# #     [256, 128, 64, 32, 16],
# #     [128, 64, 32, 16, 8],
# #     [512, 256, 128, 64, 32],
# #     [256, 128, 64],
# #     [128, 64, 32]
# # ]

# # optimizers = [
# #     Adam(),
# #     SGD(),
# #     RMSprop()
# # ]

# # best_loss = float('inf')
# # best_architecture = None
# # best_optimizer = None

# # # Test different architectures and optimizers
# # for neurons in architectures:
# #     for optimizer in optimizers:
# #         layers = len(neurons)
# #         model = create_model(X_train.shape[1], layers, neurons, optimizer)
# #         history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
# #         loss = model.evaluate(X_test, y_test, verbose=0)
# #         print(f'Architecture: {neurons}, Optimizer: {optimizer.get_config()["name"]}, Test loss: {loss}')
# #         if loss < best_loss:
# #             best_loss = loss
# #             best_architecture = neurons
# #             best_optimizer = optimizer
# #             best_history = history

# # print(f'Best architecture: {best_architecture}, Best optimizer: {best_optimizer.get_config()["name"]}, Best loss: {best_loss}')

# # # Plot the training and validation loss
# # plt.plot(best_history.history['loss'], label='Training Loss')
# # plt.plot(best_history.history['val_loss'], label='Validation Loss')
# # plt.title('Model Loss')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()

# # # Create and train the best model
# # best_model = create_model(X_train.shape[1], len(best_architecture), best_architecture, best_optimizer)
# # best_model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2)

# # # Evaluate the best model
# # loss = best_model.evaluate(X_test, y_test)
# # print(f'Test loss of best model: {loss}')

# # # Make predictions with the best model
# # y_pred = best_model.predict(X_test)


# # # export_model(best_model, 'oldmodelkrnw.model')
