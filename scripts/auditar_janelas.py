"""
Auditoria de janelas corrompidas — Condição A (bug) vs Condição B (atual correta).

Condição A: commit 1c864df — criar_janelas sem colunas_grupo, desliza sobre o
DataFrame inteiro. Chamada em c759802:main.py sem sort_values antes do call.

Cadeia de ordenação real (rastreada no git):
  1. loader.py (1c864df): sort_values("date").reset_index(drop=True)
  2. preprocessor.preparar_dataset: groupby().transform() preserva ordem de linha
  3. dividir_treino_teste: filtro booleano + reset_index(drop=True)
  4. normalizar: fit_transform/transform preserva ordem
  => DataFrame entrava no criar_janelas antigo ordenado por date,
     com desempate na ordem natural do CSV Kaggle (date, store, item).

Candidato principal da Condição A real: sort_values(["date", "store", "item"]).
Candidato alternativo (testado como fallback): sort_values(["store", "item", "date"]).

Uso:
    python scripts/auditar_janelas.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.pipeline.loader import carregar_serie_temporal
from src.pipeline.preprocessor import preparar_dataset
from src.pipeline.splitter import dividir_treino_teste, normalizar


def _contar_corrompidas_em_df(
    df_ord: pd.DataFrame,
    colunas_grupo: list[str],
    window: int,
) -> tuple[int, int]:
    """
    Aplica a Condição A (sem segmentação de grupo) sobre df_ord e conta
    janelas onde ao menos uma linha tem grupo diferente do alvo.
    Retorna (total_janelas, n_corrompidas).
    """
    grupos = list(zip(*[df_ord[c].values for c in colunas_grupo]))
    n = len(df_ord) - window
    corrompidas = 0
    for i in range(n):
        alvo = grupos[i + window]
        for g in grupos[i : i + window]:
            if g != alvo:
                corrompidas += 1
                break
    return n, corrompidas


def auditar_condicao_a(
    df: pd.DataFrame,
    colunas_grupo: list[str],
    coluna_alvo: str,
    colunas_features: list[str],
    window: int = 30,
    descricao: str = "",
) -> None:
    """
    Testa dois ordenamentos sobre o DataFrame fornecido e imprime o resultado.
    O candidato primário (date, store, item) é marcado como 'REAL' com base na
    evidência do git; o alternativo (store, item, date) serve de fallback.
    """
    if descricao:
        print(f"=== {descricao} ===")

    candidatos = [
        (
            ["date"] + colunas_grupo,
            "CANDIDATO REAL (git: loader faz sort_values('date'); CSV Kaggle = date,store,item)",
        ),
        (
            colunas_grupo + ["date"],
            "Alternativo (store, item, date)",
        ),
    ]

    for colunas_ord, rotulo in candidatos:
        df_ord = df.sort_values(colunas_ord).reset_index(drop=True)
        total, corrompidas = _contar_corrompidas_em_df(df_ord, colunas_grupo, window)
        pct = corrompidas / total * 100 if total > 0 else 0.0
        print(f"  [{rotulo}]")
        print(f"    Total de janelas : {total:,}")
        print(f"    Corrompidas       : {corrompidas:,}")
        print(f"    Percentual        : {pct:.2f}%")
        print()


if __name__ == "__main__":
    CAMINHO_DADOS = "data/raw/train.csv"
    COLUNA_ALVO = "sales"
    COLUNAS_GRUPO = ["store", "item"]
    WINDOW = 30

    print("Carregando dados...")
    df = carregar_serie_temporal(CAMINHO_DADOS)

    print("Pre-processando...")
    df_proc, features = preparar_dataset(df, COLUNA_ALVO, COLUNAS_GRUPO)

    print("Dividindo treino/teste...")
    df_treino, df_teste = dividir_treino_teste(df_proc)

    print("Normalizando...")
    df_tr_norm, df_te_norm, _, _ = normalizar(
        df_treino, df_teste, features, COLUNA_ALVO
    )

    print()
    auditar_condicao_a(
        df_tr_norm, COLUNAS_GRUPO, COLUNA_ALVO, features, WINDOW, "TREINO"
    )
    auditar_condicao_a(
        df_te_norm, COLUNAS_GRUPO, COLUNA_ALVO, features, WINDOW, "TESTE"
    )

    print("Auditoria concluida.")
    print()
    print("NOTA: O percentual do candidato REAL (date, store, item) e o numero")
    print("comparavel com o MAPE de 56% dos modelos para avaliar impacto do bug.")
