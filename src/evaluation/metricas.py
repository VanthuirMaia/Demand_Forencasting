import numpy as np
import pandas as pd


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray, nome_modelo: str) -> dict:
    """Calcula MAE, RMSE e MAPE. Ignora amostras com y_real == 0 no MAPE."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_real - y_pred))
    rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))

    mascara = y_real != 0
    mape = np.mean(np.abs((y_real[mascara] - y_pred[mascara]) / y_real[mascara])) * 100

    return {
        "modelo": nome_modelo,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
    }


def comparar_modelos(lista_resultados: list[dict]) -> pd.DataFrame:
    """Monta tabela comparativa ordenada por MAE crescente."""
    return (
        pd.DataFrame(lista_resultados)[["modelo", "MAE", "RMSE", "MAPE"]]
        .sort_values("MAE")
        .reset_index(drop=True)
    )
