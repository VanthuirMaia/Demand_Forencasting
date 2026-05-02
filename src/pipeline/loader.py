import pandas as pd

# Nomes de coluna de data reconhecidos automaticamente
_COLUNAS_DATA = {"date", "data", "timestamp", "dt"}


def carregar_serie_temporal(caminho_csv: str) -> pd.DataFrame:
    """Carrega um CSV de série temporal, detecta a coluna de data e ordena por ela."""
    df = pd.read_csv(caminho_csv)

    coluna_data = _detectar_coluna_data(df)
    if coluna_data is None:
        raise ValueError(
            f"Nenhuma coluna de data encontrada. "
            f"Esperado um dos nomes: {_COLUNAS_DATA}. "
            f"Colunas encontradas: {list(df.columns)}"
        )

    df[coluna_data] = pd.to_datetime(df[coluna_data])
    df = df.sort_values(coluna_data).reset_index(drop=True)

    return df


def validar_serie_temporal(
    df: pd.DataFrame,
    colunas_grupo: list[str] | None = None,
) -> dict:
    """
    Valida um DataFrame de série temporal.

    Parâmetros
    ----------
    df : DataFrame já carregado
    colunas_grupo : colunas que identificam a série (ex: ['store', 'item']).
                    Se None, trata o dataset como série única.

    Retorna um dicionário com o resultado de cada verificação.
    """
    relatorio = {}

    # 1. Coluna de data presente
    coluna_data = _detectar_coluna_data(df)
    relatorio["coluna_data_encontrada"] = coluna_data is not None
    relatorio["coluna_data"] = coluna_data

    # 2. Ao menos uma coluna numérica (excluindo a própria coluna de data)
    colunas_numericas = df.select_dtypes(include="number").columns.tolist()
    relatorio["colunas_numericas"] = colunas_numericas
    relatorio["tem_coluna_numerica"] = len(colunas_numericas) > 0

    # 3. Datas duplicadas por série
    if coluna_data is not None:
        if colunas_grupo:
            duplicatas = df.duplicated(subset=colunas_grupo + [coluna_data]).sum()
        else:
            duplicatas = df.duplicated(subset=[coluna_data]).sum()

        relatorio["datas_duplicadas"] = int(duplicatas)
        relatorio["sem_duplicatas"] = duplicatas == 0

    relatorio["valido"] = all([
        relatorio.get("coluna_data_encontrada", False),
        relatorio.get("tem_coluna_numerica", False),
        relatorio.get("sem_duplicatas", True),
    ])

    return relatorio


def _detectar_coluna_data(df: pd.DataFrame) -> str | None:
    """Retorna o nome da primeira coluna cujo nome (em minúsculas) está em _COLUNAS_DATA."""
    for col in df.columns:
        if col.strip().lower() in _COLUNAS_DATA:
            return col
    return None
