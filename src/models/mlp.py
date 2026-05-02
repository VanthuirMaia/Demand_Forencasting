import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers


def construir_mlp(
    input_shape: tuple[int, int],
    camadas: list[int] = [128, 64],
    dropout: float = 0.2,
) -> keras.Model:
    """Constrói e compila um MLP para regressão de séries temporais."""
    modelo = keras.Sequential()
    modelo.add(layers.Input(shape=input_shape))
    modelo.add(layers.Flatten())

    for unidades in camadas:
        modelo.add(layers.Dense(unidades, activation="relu"))
        modelo.add(layers.Dropout(dropout))

    modelo.add(layers.Dense(1))  # saída escalar sem ativação (regressão)

    modelo.compile(optimizer="adam", loss="mse")
    return modelo


def treinar_mlp(
    modelo: keras.Model,
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    epochs: int = 50,
    batch_size: int = 512,
    validation_split: float = 0.1,
) -> keras.callbacks.History:
    """Treina o MLP com EarlyStopping sobre val_loss."""
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    historico = modelo.fit(
        X_treino,
        y_treino,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1,
    )

    return historico


def prever_mlp(
    modelo: keras.Model,
    X_teste: np.ndarray,
    scaler_alvo: MinMaxScaler,
) -> np.ndarray:
    """Gera predições e inverte a normalização para a escala original."""
    predicoes = modelo.predict(X_teste, verbose=0)
    return scaler_alvo.inverse_transform(predicoes).flatten()
