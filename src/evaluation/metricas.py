import numpy as np
import pandas as pd


def calcular_metricas(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    nome_modelo: str,
    mase: float | None = None,
) -> dict:
    """
    Calcula MAE, RMSE e MAPE. Ignora amostras com y_real == 0 no MAPE.
    mase: valor pré-computado por calcular_mase_agregado (None se não disponível).
    """
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
        "MASE": round(float(mase), 4) if mase is not None else None,
    }


def comparar_modelos(lista_resultados: list[dict]) -> pd.DataFrame:
    """Monta tabela comparativa ordenada por MAE crescente. Inclui MASE se disponível."""
    df = pd.DataFrame(lista_resultados)
    colunas = ["modelo", "MAE", "RMSE", "MAPE"]
    if "MASE" in df.columns and df["MASE"].notna().any():
        colunas.append("MASE")
    return df[colunas].sort_values("MAE").reset_index(drop=True)
