"""Módulo 3 — Arquitectura LSTM apilada para regresión de RUL."""
from __future__ import annotations

import keras
from keras import layers

from src.config import TrainConfig


def build_lstm_model(
    window: int,
    n_features: int,
    lstm_units: tuple = (64, 32),
    dense_units: tuple = (32, 16),
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    output_scale: float = 125.0,
) -> keras.Model:
    
    if not lstm_units:
        raise ValueError("Se requiere al menos una capa LSTM.")

    inputs = keras.Input(shape=(window, n_features), name="ventana_sensores")
    x = inputs
    for i, units in enumerate(lstm_units):
        last = i == len(lstm_units) - 1
        x = layers.LSTM(units, return_sequences=not last, name=f"lstm_{i + 1}")(x)
        x = layers.Dropout(dropout, name=f"dropout_lstm_{i + 1}")(x)
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        x = layers.Dropout(dropout, name=f"dropout_dense_{i + 1}")(x)
    x = layers.Dense(1, activation="linear", name="salida_lineal")(x)
    outputs = layers.Rescaling(scale=float(output_scale), name="rul_ciclos")(x)

    model = keras.Model(inputs, outputs, name="lstm_rul")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def build_from_config(cfg: TrainConfig, n_features: int) -> keras.Model:
    return build_lstm_model(
        window=cfg.window,
        n_features=n_features,
        lstm_units=cfg.lstm_units,
        dense_units=cfg.dense_units,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
        output_scale=cfg.rul_cap,
    )


def make_callbacks(cfg: TrainConfig) -> list:
    """ReduceLROnPlateau + parada temprana."""
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=cfg.lr_factor, patience=cfg.lr_patience,
            min_lr=cfg.min_lr, verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.early_stop_patience,
            restore_best_weights=True, verbose=0,
        ),
    ]