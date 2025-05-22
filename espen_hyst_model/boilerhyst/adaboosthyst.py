import csv
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltbis

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
import numpy as np

path = os.getcwd()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


sys.path.insert(0, '/Users/macbookn/activopmwkspc/edgedev/opm-common/python/opm/ml')

from ml_tools import export_model


def PreprocessData(Sh, data, wp):

    SandSmax = []

    krw = []
    krnw = []
    pc = []
    S2 = []

    with open(path+'/KrPcData_Swi0_06.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            S2.append(1-float(row[3]))
            krnw.append(float(row[6]))
            krw.append(float(row[5]))
            pc.append(float(row[4]))
            SandSmax.append([1-float(row[3]),1-float(row[2])])

    return krnw, krw, SandSmax

def trainNN(krnw, Smax):
    x = np.array(Smax)
    y = np.array(krnw)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Use AdaBoost with DecisionTreeRegressor as the base estimator
    base_estimator = DecisionTreeRegressor(max_depth=4)
    model = AdaBoostRegressor(estimator=base_estimator, n_estimators=1000, random_state=42)

    # Fit the model on the training dataset
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    loss = mean_squared_error(y_test, y_pred)
    print(f'Test loss: {loss}')

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

satspace = np.linspace(0.0, 1.0e0, 100)
satMaxspace = np.linspace(2.0e-1, 7.5e-1, 20)

for valsatmax in satMaxspace:
    for valsat in satspace:
        if valsatmax > valsat:
            xhat = np.array([[valsat, valsatmax]])
            yhatkrnw = modelnonwett.predict(xhat)
            pyplot.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=5, markeredgecolor="red")

plt.legend()
plt.savefig("output/Killoughbis.png")
plt.show()

#export_model(modelnonwett, 'oldmodelkrnw.model')

plt.savefig("newKillough.png")

