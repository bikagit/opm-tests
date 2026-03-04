# -*- coding: utf-8 -*-
"""
Dense Random-Feature "Reservoir" for regression (ELM/Random Kitchen Sinks style)
- No Keras/TensorFlow required
- Random Dense layers with fixed weights (the "reservoir")
- Closed-form ridge readout training
- Train/val/test similar to your original pipeline
- Backward-compatible RMSE (no 'squared=' kwarg)
- Saves model to NPZ and produces parity/residual plots

Run:
    python dense_reservoir_krnw.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os, sys

# Reproducibility
np.random.seed(1234)

sys.path.insert(0, '/Users/macbookn/activopmwkspc/edgedev/opm-common/python/opm/ml')

from ml_tools import export_model
from ml_tools import  MinMaxScalerLayer, MinMaxUnScalerLayer
# -----------------------------
# Utilities
# -----------------------------
def rmse(y_true, y_pred):
    """
    Backward-compatible RMSE:
      - Uses squared=False when available (newer sklearn)
      - Falls back to sqrt(MSE) for older versions (avoids TypeError)
    """
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

# -----------------------------
# Data I/O and preprocessing
# -----------------------------
def PreprocessData(file_path):
    """
    Reads CSV and returns:
      X: [[1 - row[3], 1 - row[2]], ...]
      y: [row[6], ...]
    """
    SandSmax, krnw = [], []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # skip header
        for row in reader:
            # Matches your original feature engineering:
            # X = [1 - float(row[3]), 1 - float(row[2])]
            SandSmax.append([1.0 - float(row[3]), 1.0 - float(row[2])])
            krnw.append(float(row[6]))
    return np.array(SandSmax, dtype=np.float64), np.array(krnw, dtype=np.float64)

def generate_synthetic_data(x, y, num_samples):
    """
    Simple augmentation by jittering inputs and y slightly.
    """
    synthetic_x, synthetic_y = [], []
    for _ in range(num_samples):
        idx = np.random.randint(0, len(x))
        noise = np.random.normal(0, 0.01, x.shape[1])
        synthetic_x.append(x[idx] + noise)
        synthetic_y.append(y[idx] + noise[0])
    return np.array(synthetic_x), np.array(synthetic_y)

# -----------------------------
# Dense Reservoir Regressor
# -----------------------------
class DenseReservoirRegressor:
    """
    Feed-forward dense "reservoir" with fixed random weights, trained linear readout.

    - Layer stack: X -> Dense(random) -> act -> Dense(random) -> act -> ...
    - You can concatenate hidden activations across layers for a richer feature set.
    - Uses StandardScaler for X and y (optional).
    - Readout trained by ridge regression (closed-form).

    Parameters
    ----------
    input_dim: int
        Number of input features.
    layer_sizes: list[int]
        Sizes of dense random layers (e.g., [256, 256, 256]).
    activation: str
        One of {'tanh', 'relu', 'leaky_relu', 'sigmoid', 'softplus'}.
    concat_all_layers: bool
        If True, concatenates activations from all layers as features.
        If False, only the last layer's activations are used.
    include_input_in_features: bool
        If True, includes the original input X in the final feature vector.
    ridge_reg: float
        L2 regularization for readout.
    scale_data: bool
        If True, scales X and y using StandardScaler.
    random_state: int
        RNG seed.
    weight_scale: float or None
        Additional scaling factor for random weights; if None, uses activation-aware init.
    """

    def __init__(
        self,
        input_dim,
        layer_sizes=(256, 256, 256),
        activation='tanh',
        concat_all_layers=True,
        include_input_in_features=True,
        ridge_reg=1e-5,
        scale_data=True,
        random_state=1234,
        weight_scale=None,
        leaky_relu_slope=0.01,
    ):
        self.input_dim = input_dim
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.concat_all_layers = concat_all_layers
        self.include_input_in_features = include_input_in_features
        self.ridge_reg = ridge_reg
        self.scale_data = scale_data
        self.rng = np.random.default_rng(random_state)
        self.weight_scale = weight_scale
        self.leaky_relu_slope = leaky_relu_slope

        # Set in fit()
        self.weights = []  # list of (W, b)
        self.x_scaler = None
        self.y_scaler = None
        self.W_out = None  # readout weights

    # ---------- activations ----------
    def _act(self, Z):
        if self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'relu':
            return np.maximum(0.0, Z)
        elif self.activation == 'leaky_relu':
            return np.where(Z >= 0.0, Z, self.leaky_relu_slope * Z)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-Z))
        elif self.activation == 'softplus':
            # numerically stable softplus
            return np.log1p(np.exp(-np.abs(Z))) + np.maximum(Z, 0)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _fan_in_scale(self, fan_in):
        """Activation-aware weight scaling (Xavier/He-like)."""
        if self.weight_scale is not None:
            return self.weight_scale / np.sqrt(max(fan_in, 1))
        if self.activation in ('tanh', 'sigmoid', 'softplus'):
            return 1.0 / np.sqrt(max(fan_in, 1))  # Xavier-ish
        elif self.activation in ('relu', 'leaky_relu'):
            return np.sqrt(2.0 / max(fan_in, 1))  # He-ish
        else:
            return 1.0 / np.sqrt(max(fan_in, 1))

    def _init_random_layers(self):
        self.weights = []
        prev_dim = self.input_dim
        for size in self.layer_sizes:
            scale = self._fan_in_scale(prev_dim)
            W = self.rng.standard_normal((prev_dim, size)) * scale
            b = self.rng.standard_normal(size) * 0.01  # small biases
            self.weights.append((W, b))
            prev_dim = size

    # ---------- forward / features ----------
    def _forward_layers(self, Xs):
        """
        Forward pass through random dense stack.
        Returns list of hidden activations per layer: [H1, H2, ...].
        """
        Hs = []
        H = Xs
        for (W, b) in self.weights:
            Z = H @ W + b
            H = self._act(Z)
            Hs.append(H)
        return Hs

    def _design_matrix(self, Xs):
        """
        Build feature matrix Z for linear readout:
          - bias column
          - optional input X
          - hidden activations: last layer or concatenation of all
        """
        Hs = self._forward_layers(Xs)
        if self.concat_all_layers:
            hidden = np.concatenate(Hs, axis=1) if len(Hs) > 1 else Hs[0]
        else:
            hidden = Hs[-1]
        parts = [np.ones((Xs.shape[0], 1))]  # bias
        if self.include_input_in_features:
            parts.append(Xs)
        parts.append(hidden)
        Z = np.concatenate(parts, axis=1)
        return Z

    # ---------- fit / predict ----------
    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        if self.scale_data:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            Xs = self.x_scaler.fit_transform(X)
            ys = self.y_scaler.fit_transform(y)
        else:
            Xs, ys = X, y

        # Initialize random dense reservoir
        self._init_random_layers()

        # Build design matrix and train readout
        Z = self._design_matrix(Xs)
        regI = self.ridge_reg * np.eye(Z.shape[1])
        A = Z.T @ Z + regI
        B = Z.T @ ys
        self.W_out = np.linalg.solve(A, B)  # (features, 1)

        # Metrics
        y_pred = self.predict(X)
        train_r2 = r2_score(y, y_pred)
        train_rmse = rmse(y, y_pred)
        print(f"[DenseReservoir] Train R2={train_r2:.4f}, RMSE={train_rmse:.6f}")

        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            val_r2 = r2_score(y_val, y_val_pred)
            val_rmse = rmse(y_val, y_val_pred)
            print(f"[DenseReservoir]  Val  R2={val_r2:.4f}, RMSE={val_rmse:.6f}")

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xs = self.x_scaler.transform(X) if (self.scale_data and self.x_scaler) else X
        Z = self._design_matrix(Xs)
        y_scaled = Z @ self.W_out
        if self.scale_data and self.y_scaler:
            y = self.y_scaler.inverse_transform(y_scaled)
        else:
            y = y_scaled
        return y.ravel()

    # ---------- persistence ----------
    def save(self, filepath):
        """
        Save model to .npz (random weights, readout, scalers, hyperparams).
        Uses dtype=object for Ws/bs to avoid inhomogeneous array shape errors.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Pack weights into object arrays to support differing shapes across layers
        Ws = [W for (W, b) in self.weights]
        bs = [b for (W, b) in self.weights]
        Ws_arr = np.array(Ws, dtype=object)
        bs_arr = np.array(bs, dtype=object)

        # Scalers
        x_mean = self.x_scaler.mean_ if self.x_scaler else np.array([])
        x_scale = self.x_scaler.scale_ if self.x_scaler else np.array([])
        y_mean = self.y_scaler.mean_ if self.y_scaler else np.array([])
        y_scale = self.y_scaler.scale_ if self.y_scaler else np.array([])

        np.savez(
            filepath,
            layer_sizes=np.array(self.layer_sizes, dtype=np.int64),
            activation=np.array(self.activation),  # 0-d unicode array is fine
            concat_all_layers=np.array([int(self.concat_all_layers)]),
            include_input_in_features=np.array([int(self.include_input_in_features)]),
            ridge_reg=np.array([self.ridge_reg]),
            scale_data=np.array([int(self.scale_data)]),
            weight_scale=np.array([-1.0 if self.weight_scale is None else float(self.weight_scale)]),
            leaky_relu_slope=np.array([self.leaky_relu_slope]),
            W_out=self.W_out,
            Ws=Ws_arr,
            bs=bs_arr,
            x_scaler_mean=x_mean,
            x_scaler_scale=x_scale,
            y_scaler_mean=y_mean,
            y_scaler_scale=y_scale,
            input_dim=np.array([self.input_dim], dtype=np.int64),
        )
        print(f"[DenseReservoir] Model saved to {filepath}.npz")

    @staticmethod
    def load(filepath):
        """
        Load model from .npz created by save().
        """
        data = np.load(filepath + ".npz", allow_pickle=True)

        input_dim = int(data["input_dim"][0])
        layer_sizes = list(data["layer_sizes"].tolist())
        activation = str(data["activation"])
        concat_all_layers = bool(int(data["concat_all_layers"][0]))
        include_input_in_features = bool(int(data["include_input_in_features"][0]))
        ridge_reg = float(data["ridge_reg"][0])
        scale_data = bool(int(data["scale_data"][0]))
        weight_scale_val = float(data["weight_scale"][0])
        weight_scale = None if weight_scale_val < 0 else weight_scale_val
        leaky_relu_slope = float(data["leaky_relu_slope"][0])

        model = DenseReservoirRegressor(
            input_dim=input_dim,
            layer_sizes=layer_sizes,
            activation=activation,
            concat_all_layers=concat_all_layers,
            include_input_in_features=include_input_in_features,
            ridge_reg=ridge_reg,
            scale_data=scale_data,
            weight_scale=weight_scale,
            leaky_relu_slope=leaky_relu_slope,
        )

        # Restore weights (object arrays)
        Ws = list(data["Ws"])
        bs = list(data["bs"])
        model.weights = [(Ws[i], bs[i]) for i in range(len(Ws))]

        # Restore readout
        model.W_out = data["W_out"]

        # Restore scalers
        # if model.scale_data and data["x_scaler_mean"].size > 0:
        #     model.x_scaler = StandardScaler()
        #     model.x_scaler.mean_ = data["x_scaler_mean"]
        #     model.x_scaler.scale_ = data["x_scaler_scale"]
        #     model.x_scaler.var_ = model.x_scaler.scale_ ** 2
        #     model.x_scaler.n_features_in_ = model.x_scaler.mean_.shape[0]
        # if model.scale_data and data["y_scaler_mean"].size > 0:
        #     model.y_scaler = StandardScaler()
        #     model.y_scaler.mean_ = data["y_scaler_mean"]
        #     model.y_scaler.scale_ = data["y_scaler_scale"]
        #     model.y_scaler.var_ = model.y_scaler.scale_ ** 2
        #     model.y_scaler.n_features_in_ = 1

        print(f"[DenseReservoir] Model loaded from {filepath}.npz")
        return model

# -----------------------------
# Training pipeline using Dense Reservoir
# -----------------------------
def trainDenseReservoir(
    x, y,
    use_augmentation=True,
    aug_multiplier=30,
    random_state=42,
    layer_sizes=(256, 256, 256),
    activation='tanh',
    concat_all_layers=True,
    include_input_in_features=True,
    ridge_reg=1e-5
):
    """
    Mirrors your original trainNN() pipeline but with a DenseReservoirRegressor.
    """
    if use_augmentation:
        x_resampled, y_resampled = resample(
            x, y, n_samples=len(x) * aug_multiplier, random_state=random_state
        )
        synthetic_x, synthetic_y = generate_synthetic_data(
            x_resampled, y_resampled, num_samples=len(x) * aug_multiplier
        )
        x_combined = np.vstack((x, synthetic_x))
        y_combined = np.hstack((y, synthetic_y))
    else:
        x_combined, y_combined = x, y

    X_train, X_test, y_train, y_test = train_test_split(
        x_combined, y_combined, test_size=0.3, random_state=random_state
    )
    X_train, x_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=random_state
    )

    model = DenseReservoirRegressor(
        input_dim=X_train.shape[1],
        layer_sizes=layer_sizes,
        activation=activation,
        # concat_all_layers=concat_all_layers,
        # include_input_in_features=include_input_in_features,
        # ridge_reg=ridge_reg,
        # scale_data=True,
        random_state=1234
    )

    model.fit(X_train, y_train, X_val=x_val, y_val=y_val)
    return model, X_test, y_test, x_val, y_val

# -----------------------------
# Main: paths, train, evaluate, plot
# -----------------------------
if __name__ == "__main__":
    # Paths to data
    train_file = 'data/curated_data/KrPcData_Combinedtot.csv'

    # Load data
    x_train, y_train = PreprocessData(train_file)

    # Train Dense Reservoir (tune layer_sizes/activation as needed)
    model, X_test, y_test, x_val, y_val = trainDenseReservoir(
        x_train, y_train,
        # use_augmentation=True,
        # aug_multiplier=30,
        layer_sizes=(256, 256, 256),  # try (512,512) or (256,256,256,256)
        activation='tanh',             # try 'relu' or 'leaky_relu'
        # concat_all_layers=True,
        # include_input_in_features=True,
        # ridge_reg=1e-5
    )

    # Save model
    os.makedirs("model", exist_ok=True)
    model.save('model/dense_reservoir_krnw_model')
    export_model(model, 'model/oldmodelkrnw.model')

    # Predictions
    y_pred_test = model.predict(X_test)
    y_pred_val = model.predict(x_val)
    y_pred_all = model.predict(x_train)

    # Metrics
    test_r2 = r2_score(y_test, y_pred_test)
    val_r2 = r2_score(y_val, y_pred_val)
    test_rmse = rmse(y_test, y_pred_test)
    val_rmse = rmse(y_val, y_pred_val)
    print(f"[DenseReservoir] Test  R2={test_r2:.4f}, RMSE={test_rmse:.6f}")
    print(f"[DenseReservoir]  Val  R2={val_r2:.4f}, RMSE={val_rmse:.6f}")

    # -----------------------------
    # Plotting (mirrors your layout)
    # -----------------------------
    plt.figure(figsize=(30, 6))

    # 1. Test Set Parity
    plt.subplot(1, 4, 1)
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='green')
    lo, hi = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.title('Dense Reservoir – Test Set: Actual vs Predicted')
    plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.grid(True)

    # 2. Residuals on Test
    plt.subplot(1, 4, 2)
    residuals = y_pred_test - y_test
    plt.hist(residuals, bins=40, color='steelblue', alpha=0.8)
    plt.title('Dense Reservoir – Test Residuals')
    plt.xlabel('Prediction Error'); plt.ylabel('Count'); plt.grid(True)

    # 3. Model vs Original (Validation)
    plt.subplot(1, 4, 3)
    plt.scatter(x_val[:, 0], y_val, color='cyan', label='Original')
    plt.scatter(x_val[:, 0], y_pred_val, color='black', marker='o', label='Dense Reservoir')
    plt.title('Dense Reservoir vs Original (Validation)')
    plt.xlabel('$S_n$'); plt.ylabel('$k_{rn}$'); plt.legend(); plt.grid(True)

    # 4. Validation Set Parity
    plt.subplot(1, 4, 4)
    plt.scatter(y_val, y_pred_val, alpha=0.6, color='purple')
    lo, hi = min(y_val.min(), y_pred_val.min()), max(y_val.max(), y_pred_val.max())
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.title('Dense Reservoir – Validation: Actual vs Predicted')
    plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.grid(True)

    os.makedirs("output_figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig("output_figures/dense_reservoir_model_plots.png", dpi=200)
    plt.close()
