import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.evaluation.metricas import calcular_metricas  # noqa: F401  — reexportado para conveniência


def preparar_serie_arima(
    df: pd.DataFrame,
    coluna_data: str = "date",
    coluna_alvo: str = "sales",
    data_corte: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Agrega vendas por data e divide em treino/teste."""
    serie = df.groupby(coluna_data)[coluna_alvo].sum().sort_index()
    serie.index = pd.to_datetime(serie.index)

    if data_corte is None:
        data_corte = serie.index[int(len(serie) * 0.8)]

    data_corte = pd.Timestamp(data_corte)

    serie_treino = serie[serie.index < data_corte]
    serie_teste = serie[serie.index >= data_corte]

    print(f"Serie agregada: {len(serie)} dias")
    print(f"Treino: {len(serie_treino)} dias  |  Teste: {len(serie_teste)} dias")

    return serie_treino, serie_teste


def construir_e_treinar_arima(
    serie_treino: pd.Series,
    ordem: tuple[int, int, int] = (1, 1, 1),
):
    """Treina um modelo ARIMA na série de treino."""
    modelo = ARIMA(serie_treino, order=ordem)
    modelo_fitted = modelo.fit()
    print(modelo_fitted.summary().tables[0])
    return modelo_fitted


def prever_arima(modelo_fitted, serie_teste: pd.Series) -> np.ndarray:
    """Gera previsões para o horizonte do conjunto de teste."""
    previsao = modelo_fitted.forecast(steps=len(serie_teste))
    return previsao.values


