import csv
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras

# Ensure eager execution is enabled
tf.config.run_functions_eagerly(True)

# Set random seed for reproducibility
keras.utils.set_random_seed(1234)

# Import custom modules
sys.path.insert(0, '/Users/macbookn/activopmwkspc/edgedev/opm-common/python/opm/ml')
from ml_tools import export_model

# Function to create a model with a specified number of layers and neurons
def create_model(input_dim, layers, neurons, optimizer):
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=input_dim, activation='tanh'))
    for i in range(1, layers):
        model.add(Dense(neurons[i], activation='tanh'))
    model.add(Dense(1, activation='tanh'))  # Output layer for regression
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

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

def trainNN(krnw, Smax, n_splits=5):
    x = np.array(Smax)
    y = np.array(krnw)

    # Generate synthetic data
    synthetic_x, synthetic_y = generate_synthetic_data(x, y, num_samples=len(x) * 5)

    # Combine original and synthetic data
    x_combined = np.vstack((x, synthetic_x))
    y_combined = np.hstack((y, synthetic_y))

    # Define model parameters
    input_dim = x_combined.shape[1]
    layers = 12
    neurons = [4] * layers

    # K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for train_index, test_index in kf.split(x_combined):
        X_train, X_test = x_combined[train_index], x_combined[test_index]
        y_train, y_test = y_combined[train_index], y_combined[test_index]

        optimizer = Adam(learning_rate=0.01)

        model = create_model(input_dim, layers, neurons, optimizer)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, start_from_epoch=200)

        history = model.fit(X_train, y_train, epochs=350, batch_size=64,
                            validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

        loss = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test, verbose=0)
        r2 = r2_score(y_test, y_pred)
        fold_metrics.append((loss, r2))

    avg_loss = np.mean([m[0] for m in fold_metrics])
    avg_r2 = np.mean([m[1] for m in fold_metrics])

    print(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Average R2 Score: {avg_r2:.4f}")

    return model, history, X_test, y_test, avg_loss, avg_r2

# Main script
swl = 0.05
path = os.getcwd()
S = np.linspace(0.0, 1.0-swl, 700)
smax = 0.
S2 = 1.0 - S - swl

SS = []

for smax in S:
    SS.append(np.linspace(smax, 0, 100))

krnw, krw, SandSmax = PreprocessData(SS, "/Users/macbookn/activopmwkspc/pyDavid/pyopmnearwell/examples/hysteresis_models/Killough/CO2_KILLOUGH", "GW")

modelnonwett, history, X_test, y_test, avg_loss, avg_r2 = trainNN(krnw, SandSmax)

satspace = np.linspace(0.0e-1, 1.0e0, 1000)
satMaxspace = np.linspace(0.0e-1, 9.0e-1, 1)

export_model(modelnonwett, 'bloboldmodelkrnw.model')


# Ensure these variables are defined from your training process:
# modelnonwett, history, X_test, y_test

# Create an output folder if it doesn't exist
output_folder = "output_figures"
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(18, 6))

# Subplot 1: Actual vs Predicted Values
plt.subplot(1, 3, 1)
y_pred = modelnonwett.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.6, color='green', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)
plt.title('Final Model: Actual vs Predicted Values', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# R2 Score
r2 = r2_score(y_test, y_pred)
plt.text(0.05, 0.95, f'R2 Score: {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

# Subplot 2: Loss Function Plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss Function Over Epochs', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# Subplot 3: Solution Plot
plt.subplot(1, 3, 3)
# Define satspace and satMaxspace
satspace = np.linspace(0.0e-1, 1.0e0, 100)
satMaxspace = np.linspace(6.0e-1, 6.1e-1, 2)

# Flags to ensure labels are added only once
imbibition_label_added = False
drainage_label_added = False

for valsat in satspace:
    for valsatmax in satMaxspace:
        xhat = np.array([[valsat, valsatmax]])
        yhatkrnw = modelnonwett.predict(xhat)
        
        # Imbibition curves
        if valsatmax > valsat:
            if not imbibition_label_added:
                plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue", label="Imbibition")
                imbibition_label_added = True
            else:
                plt.plot(xhat.flat[0], yhatkrnw, marker="o", markersize=3, markeredgecolor="blue")
        
        # Drainage curves
        if valsatmax < valsat:
            if not drainage_label_added:
                plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red", label="Drainage")
                drainage_label_added = True
            else:
                plt.plot(xhat.flat[0], yhatkrnw, marker="*", markersize=3, markeredgecolor="red")

plt.xlabel(r"$S_w$", fontsize=14)
plt.ylabel(r"$k_{rn}$", fontsize=14)
plt.title('Drainage and Imbibition Curves Plot', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# Save the figure
output_path = os.path.join(output_folder, "final_model_plots.png")
plt.tight_layout()
plt.savefig(output_path)
# plt.show()

print(f"Figures saved in: {output_path}")
