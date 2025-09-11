import csv
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras
from sklearn.utils import resample
import os, sys

sys.path.insert(0, '/Users/macbookn/activopmwkspc/edgedev/opm-common/python/opm/ml')

from ml_tools import export_model
from ml_tools import  MinMaxScalerLayer, MinMaxUnScalerLayer


keras.utils.set_random_seed(1234)

# Function to create a model
def create_model(input_dim, layers, neurons, optimizer):
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=input_dim, activation='tanh'))
    for i in range(1, layers):
        model.add(Dense(neurons[i], activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Load and preprocess data
def PreprocessData(file_path):
    SandSmax, krnw = [], []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            # SandSmax.append([1.0-float(row[3]),1.0-float(row[2]),float(row[7]),float(row[0])])
            SandSmax.append([1.0-float(row[3]),1.0-float(row[2])])
            krnw.append(float(row[6]))
    return np.array(SandSmax), np.array(krnw)

# Generate synthetic data
def generate_synthetic_data(x, y, num_samples):
    synthetic_x, synthetic_y = [], []
    for _ in range(num_samples):
        idx = np.random.randint(0, len(x))
        noise = np.random.normal(0, 0.01, x.shape[1])
        synthetic_x.append(x[idx] + noise)
        synthetic_y.append(y[idx] + noise[0])
    return np.array(synthetic_x), np.array(synthetic_y)

# Train the model
def trainNN(x, y):
    x_resampled, y_resampled = resample(x, y, n_samples=len(x) * 30, random_state=42)
    synthetic_x, synthetic_y = generate_synthetic_data(x_resampled, y_resampled, num_samples=len(x) * 30)
    x_combined = np.vstack((x, synthetic_x))
    y_combined = np.hstack((y, synthetic_y))
    X_train, X_test, y_train, y_test = train_test_split(x_combined, y_combined, test_size=0.3, random_state=42)
    X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    model = create_model(X_train.shape[1], 10, [19]*10, Adam(learning_rate=0.001))
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, start_from_epoch=100)
    history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
    return model, history, X_test, y_test, x_val, y_val

# Paths to data
train_file = 'data/curated_data/KrPcData_Combinedtot.csv'
# val_file =  'data/curated_data/KrPcData_Swi0_06-CORRECTED.csv'

# Load data
x_train, y_train = PreprocessData(train_file)
# x_val, y_val = PreprocessData(val_file)

# Train model
model, history, X_test, y_test, x_val, y_val = trainNN(x_train, y_train)
export_model(model, 'model/oldmodelkrnw.model')

# Predictions
y_pred_test = model.predict(X_test)
y_pred_val = model.predict(x_val)
y_pred_all = model.predict(x_train)

# Plotting
plt.figure(figsize=(30, 6))

# 1. Test Set
plt.subplot(1, 6, 1)
plt.scatter(y_test, y_pred_test, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Test Set: Actual vs Predicted')
plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.grid(True)

# 2. Loss
plt.subplot(1, 6, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

# 3. Drainage/Imbibition
# plt.subplot(1, 6, 3)
# satspace = np.linspace(0.1, 1.0, 100)
# satMaxspace = np.linspace(0.1, 0.91, 7)
# tag = 0
# imbibition_label_added = False
# drainage_label_added = False

# # for valsat in satspace:
# #     for valsatmax in satMaxspace:
# #         xhat = np.array([[valsat, valsatmax]])
# #         yhatkrnw = model.predict(xhat)
# #         if valsatmax > valsat:
# #             if not imbibition_label_added:
# #                 plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue", label="Imbibition")
# #                 imbibition_label_added = True
# #             else:
# #                 plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue")
# #         if valsatmax < valsat:
# #             if not drainage_label_added:
# #                 plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red", label="Drainage")
# #                 drainage_label_added = True
# #             else:
# #                 plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red")

# plt.plot(label='Drainage')
# plt.title('Drainage and Imbibition')
# plt.xlabel('$S_n$'); plt.ylabel('$k_{rn}$'); plt.grid(True)

# # 4. DNN vs Original Drainage
# plt.subplot(1, 6, 4)
# plt.scatter(x_val[:, 0], y_val, color='pink', marker='*', label='Original')
# plt.scatter(x_val[:, 0], y_pred_val, color='black', marker='*', label='DNN')
# plt.title('DNN vs Original Drainage')
# plt.xlabel('$S_w$'); plt.ylabel('$k_{rn}$'); plt.legend(); plt.grid(True)

# 5. DNN vs Original Imbibition
plt.subplot(1, 6, 5)
plt.scatter(x_val[:, 0], y_val, color='cyan', label='Original')
plt.scatter(x_val[:, 0], y_pred_val, color='black', marker='o', label='DNN')
plt.title('DNN vs Original Imbibition')
plt.xlabel('$S_n$'); plt.ylabel('$k_{rn}$'); plt.legend(); plt.grid(True)

# 6. Validation Set
plt.subplot(1, 6, 6)
plt.scatter(y_val, y_pred_val, alpha=0.6, color='purple')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.title('Validation Set: Actual vs Predicted')
plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.grid(True)

# Save
os.makedirs("output_figures", exist_ok=True)
plt.tight_layout()
plt.savefig("output_figures/final_model_plots_with_comparison.png")
