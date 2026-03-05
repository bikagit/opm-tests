# SPDX-FileCopyrightText: 2024 NORCE
# SPDX-License-Identifier: GPL-3.0

"""
Train a neural network to predict the best linear solver tolerance (bestTol)
from OPM Flow convergence data in bestpath.csv.

Based on the original working approach (9-layer sigmoid NN + bootstrap
resampling + MSE), with targeted improvements:

  1. Bootstrap n_samples: 51 → 400. At n=51 only ~15/20 classes are
     represented on average; n=400 reliably covers all 20 every run.

  2. Target scaling: y divided by y.max() before training so the sigmoid
     output [0,1] maps exactly onto the target range. Predictions are
     multiplied back at inference time. This removes the mismatch where
     sigmoid was implicitly predicting 10× the true range.

  3. Learning rate schedule: Adam with ReduceLROnPlateau halves lr when
     val_loss plateaus, squeezing more out of the same architecture.

  4. Early stopping: avoids wasting epochs after convergence; restores
     best weights automatically.

  5. Log-transform of residual features: cnvminmaxresid, cnvresidoil,
     cnvresidwater, cnvresidgas span many orders of magnitude — log10
     compression gives the network a much more learnable input.

  6. Plots saved to disk in addition to being shown.

Everything else — architecture, optimizer family, loss, export call — is
kept exactly as in the original.
"""

import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from numpy import asarray
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

keras.utils.set_random_seed(1234)
np.random.seed(1234)

sys.path.insert(0, '/Users/macbookn/hackatonwork/opm-common/python/opm/ml/')
from ml_tools import export_model

# ---------------------------------------------------------------------------
# Config — tweak here, not buried in code
# ---------------------------------------------------------------------------

CSV_PATH        = 'bestpath.csv'
OUTPUT_DIR      = Path('output')
TARGET_COL      = 'bestTol'
N_BOOTSTRAP     = 10000          # was 51; 400 reliably covers all 20 classes
EPOCHS          = 500
VALIDATION_SPLIT = 0.30
LOG_FEATURES    = ['cnvminmaxresid', 'cnvresidoil', 'cnvresidwater', 'cnvresidgas']

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(csv_path, n_bootstrap):
    df = pd.read_csv(csv_path)

    # Log10-transform heavy-tailed residual features
    for col in LOG_FEATURES:
        if col in df.columns:
            df[col] = np.log10(df[col].clip(lower=1e-12))

    # Bootstrap resample — stratified so all classes appear
    boot = resample(
        df,
        replace=True,
        n_samples=n_bootstrap,
        stratify=df[TARGET_COL],
        random_state=42,
    )

    x = boot.drop(TARGET_COL, axis=1).to_numpy()
    y = boot[TARGET_COL].to_numpy()

    n_features = x.shape[1]
    x = x.reshape((len(x), n_features))
    y = y.reshape((len(y), 1))

    # Scale y to [0, 1] so sigmoid output aligns with target range exactly.
    # Save y_max to invert predictions at inference time.
    y_max = float(df[TARGET_COL].max())
    y_scaled = y / y_max

    print(f"Bootstrap samples: {len(x)}  |  features: {n_features}")
    print(f"Target range: {y.min():.5f} – {y.max():.5f}  (scaled by 1/{y_max})")
    print(f"Classes covered: {len(np.unique(y))}/20")

    return x, y_scaled, y_max, n_features


# ---------------------------------------------------------------------------
# Model  — same architecture as original, input_dim set from data
# ---------------------------------------------------------------------------

def build_model(n_features):
    model = Sequential()
    model.add(Dense(8,  activation='relu',    input_dim=n_features))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1,  activation='sigmoid'))

    model.compile(
        optimizer=Adam(),
        loss='mean_squared_error',
        metrics=['mae'],     # mae is more interpretable than accuracy for regression
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, x, y_scaled, epochs, validation_split):
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=60,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=30,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    return model.fit(
        x, y_scaled,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss(history, out_dir):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'],     label='train')
    ax.plot(history.history['val_loss'], label='val')
    ax.set_title('Model loss')
    ax.set_ylabel('MSE loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    fig.savefig(out_dir / 'loss_curve.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_predictions(y_true_scaled, y_pred_scaled, y_max, out_dir):
    y_true = y_true_scaled * y_max
    y_pred = y_pred_scaled * y_max
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6, s=25, label='predictions')
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, 'r--', lw=1, label='perfect fit')
    ax.set_xlabel('Actual bestTol')
    ax.set_ylabel('Predicted bestTol')
    ax.set_title('Predicted vs Actual')
    ax.legend()
    fig.savefig(out_dir / 'predictions.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y_scaled, y_max, n_features = load_data(CSV_PATH, N_BOOTSTRAP)

    model = build_model(n_features)
    model.summary()

    history = train(model, x, y_scaled, EPOCHS, VALIDATION_SPLIT)

    # Evaluate on training set (unscaled)
    y_pred_scaled = model.predict(x, verbose=0)
    mse = mean_squared_error(y_pred_scaled * y_max, y_scaled * y_max)
    print(f'\nTrain MSE (original scale): {mse:.6f}')

    plot_loss(history, out_dir)
    plot_predictions(y_scaled, y_pred_scaled, y_max, out_dir)

    export_model(model, 'linredNN.model')
    print('Model exported → linredNN.model')


if __name__ == '__main__':
    main()