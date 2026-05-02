import pandas as pd


def criar_features_temporais(df: pd.DataFrame, coluna_data: str) -> pd.DataFrame:
    """Extrai features de calendário a partir da coluna de data."""
    dt = df[coluna_data].dt

    df["dia_mes"] = dt.day
    df["dia_semana"] = dt.dayofweek          # 0 = segunda, 6 = domingo
    df["semana_ano"] = dt.isocalendar().week.astype(int)
    df["mes"] = dt.month
    df["trimestre"] = dt.quarter
    df["ano"] = dt.year
    df["e_fim_de_semana"] = dt.dayofweek >= 5
    df["e_inicio_de_mes"] = dt.day <= 5
    df["e_fim_de_mes"] = dt.day >= 25

    return df


def criar_lags(
    df: pd.DataFrame,
    coluna_alvo: str,
    colunas_grupo: list[str],
    lags: list[int] = [1, 7, 30],
) -> pd.DataFrame:
    """Cria colunas de lag para a coluna alvo, respeitando os grupos."""
    serie = df.groupby(colunas_grupo)[coluna_alvo]

    for lag in lags:
        df[f"lag_{lag}"] = serie.shift(lag)

    return df


def criar_medias_moveis(
    df: pd.DataFrame,
    coluna_alvo: str,
    colunas_grupo: list[str],
    janelas: list[int] = [7, 30],
) -> pd.DataFrame:
    """Cria médias móveis para a coluna alvo, respeitando os grupos."""
    serie = df.groupby(colunas_grupo)[coluna_alvo]

    for janela in janelas:
        # shift(1) evita vazamento de dados — a média não inclui o valor atual
        # transform mantém o índice alinhado ao DataFrame original
        df[f"media_movel_{janela}"] = (
            df.groupby(colunas_grupo)[coluna_alvo]
            .transform(lambda x: x.shift(1).rolling(window=janela).mean())
        )

    return df


def preparar_dataset(
    df: pd.DataFrame,
    coluna_alvo: str,
    colunas_grupo: list[str],
    coluna_data: str = "date",
    lags: list[int] = [1, 7, 30],
    janelas: list[int] = [7, 30],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Pipeline completo de pré-processamento.

    Retorna o DataFrame processado e a lista de features geradas.
    NaNs introduzidos por lags e médias móveis são removidos ao final.
    """
    df = df.copy()

    df = criar_features_temporais(df, coluna_data)
    df = criar_lags(df, coluna_alvo, colunas_grupo, lags)
    df = criar_medias_moveis(df, coluna_alvo, colunas_grupo, janelas)

    features_temporais = [
        "dia_mes", "dia_semana", "semana_ano", "mes", "trimestre", "ano",
        "e_fim_de_semana", "e_inicio_de_mes", "e_fim_de_mes",
    ]
    features_lag = [f"lag_{l}" for l in lags]
    features_mm = [f"media_movel_{j}" for j in janelas]

    features = features_temporais + features_lag + features_mm

    # Remove linhas com NaN gerados pelos lags e médias móveis
    linhas_antes = len(df)
    df = df.dropna(subset=features_lag + features_mm).reset_index(drop=True)
    linhas_removidas = linhas_antes - len(df)

    print(f"Linhas removidas por NaN (lags/médias): {linhas_removidas}")
    print(f"Shape final: {df.shape}")
    print(f"Features geradas ({len(features)}): {features}")

    return df, features
