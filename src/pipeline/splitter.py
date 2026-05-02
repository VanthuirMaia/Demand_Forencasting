import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def dividir_treino_teste(
    df: pd.DataFrame,
    coluna_data: str = "date",
    data_corte: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o DataFrame por data de corte (divisão temporal, nunca aleatória).
    Se data_corte não for informada, usa os últimos 20% do período como teste.
    """
    if data_corte is None:
        datas = df[coluna_data].sort_values()
        data_corte = datas.quantile(0.8)

    data_corte = pd.Timestamp(data_corte)

    df_treino = df[df[coluna_data] < data_corte].reset_index(drop=True)
    df_teste = df[df[coluna_data] >= data_corte].reset_index(drop=True)

    print(f"Corte: {data_corte.date()}")
    print(f"Treino: {len(df_treino):,} linhas  ({df_treino[coluna_data].min().date()} a {df_treino[coluna_data].max().date()})")
    print(f"Teste:  {len(df_teste):,} linhas  ({df_teste[coluna_data].min().date()} a {df_teste[coluna_data].max().date()})")

    return df_treino, df_teste


def normalizar(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    colunas_features: list[str],
    coluna_alvo: str,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    """
    Aplica MinMaxScaler nas features e no alvo separadamente.
    Scaler fittado apenas no treino — sem data leakage.
    Retorna os scalers para inversão posterior das predições.
    """
    scaler_features = MinMaxScaler()
    scaler_alvo = MinMaxScaler()

    df_treino = df_treino.copy()
    df_teste = df_teste.copy()

    # Fit somente no treino
    df_treino[colunas_features] = scaler_features.fit_transform(df_treino[colunas_features])
    df_treino[[coluna_alvo]] = scaler_alvo.fit_transform(df_treino[[coluna_alvo]])

    # Transforma o teste com os scalers já fittados
    df_teste[colunas_features] = scaler_features.transform(df_teste[colunas_features])
    df_teste[[coluna_alvo]] = scaler_alvo.transform(df_teste[[coluna_alvo]])

    return df_treino, df_teste, scaler_features, scaler_alvo


def criar_janelas(
    df: pd.DataFrame,
    coluna_alvo: str,
    colunas_features: list[str],
    tamanho_janela: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cria arrays X e y no formato de janela deslizante.

    X: (amostras, tamanho_janela, n_features)
    y: (amostras,) — valor do alvo imediatamente após cada janela
    """
    X_vals = df[colunas_features].values
    y_vals = df[coluna_alvo].values

    n_amostras = len(df) - tamanho_janela
    n_features = len(colunas_features)

    X = np.empty((n_amostras, tamanho_janela, n_features), dtype=np.float32)
    y = np.empty(n_amostras, dtype=np.float32)

    for i in range(n_amostras):
        X[i] = X_vals[i : i + tamanho_janela]
        y[i] = y_vals[i + tamanho_janela]

    print(f"Janelas criadas — X: {X.shape}, y: {y.shape}")

    return X, y
