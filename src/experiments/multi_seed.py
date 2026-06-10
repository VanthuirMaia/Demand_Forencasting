"""
Experimentos com múltiplas seeds para MLP, LSTM e GRU.

Salva resultados incrementalmente em CSV após cada seed para sobreviver a
desconexões do Colab. clear_session() entre seeds evita acúmulo de memória
na GPU T4 e garante reinicialização completa dos pesos.

Uso típico (Colab):
    from src.experiments.multi_seed import executar_experimento, resumir_resultados
    from src.models.mlp import construir_mlp, treinar_mlp, prever_mlp

    df_mlp = executar_experimento(
        "MLP", construir_mlp, treinar_mlp, prever_mlp,
        X_tr, y_tr, X_te, y_te_real, sc_alvo, input_shape,
        df_treino, df_te_norm, COLUNAS_GRUPO, COLUNA_ALVO,
    )
    resumir_resultados(df_mlp)
"""

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from src.evaluation.metricas import calcular_metricas
from src.baselines import calcular_mase_agregado

SEEDS_PADRAO = [42, 123, 7, 2024, 99]
CAMINHO_CSV_PADRAO = "resultados/metricas_por_seed.csv"


def _fixar_seed(seed: int) -> None:
    """Fixa numpy, random e tensorflow para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def executar_experimento(
    nome: str,
    construir_fn,
    treinar_fn,
    prever_fn,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te_real: np.ndarray,
    sc_alvo,
    input_shape: tuple,
    df_treino: pd.DataFrame,
    df_te_norm: pd.DataFrame,
    colunas_grupo: list[str],
    coluna_alvo: str,
    seeds: list[int] = SEEDS_PADRAO,
    epochs: int = 50,
    batch_size: int = 512,
    tamanho_janela: int = 30,
    sazonalidade: int = 7,
    caminho_csv: str = CAMINHO_CSV_PADRAO,
) -> pd.DataFrame:
    """
    Treina o modelo uma vez por seed e coleta MAE, RMSE, MAPE e MASE.

    clear_session() antes de cada seed libera memória da GPU e reinicia o grafo
    TensorFlow, evitando vazamento entre execuções.

    O CSV é gravado incrementalmente (append) após cada seed. Se o arquivo já
    existir com resultados de seeds anteriores, seeds já presentes são puladas.
    """
    os.makedirs(os.path.dirname(caminho_csv) or ".", exist_ok=True)

    # Carrega seeds já concluídas para permitir retomada
    seeds_feitas: set[int] = set()
    if os.path.exists(caminho_csv):
        df_existente = pd.read_csv(caminho_csv)
        seeds_feitas = set(
            df_existente.loc[df_existente["modelo"] == nome, "seed"].astype(int)
        )
        if seeds_feitas:
            print(f"  Seeds já concluídas para {nome}: {sorted(seeds_feitas)} — pulando.")

    resultados = []

    for seed in seeds:
        if seed in seeds_feitas:
            continue

        print(f"\n  [{nome}] seed={seed}")

        # Libera memória e reinicia grafo — essencial na GPU T4 do Colab
        tf.keras.backend.clear_session()
        _fixar_seed(seed)

        modelo = construir_fn(input_shape=input_shape)
        treinar_fn(modelo, X_tr, y_tr, epochs=epochs, batch_size=batch_size)
        preds = prever_fn(modelo, X_te, sc_alvo)

        res = calcular_metricas(y_te_real, preds, nome)
        mase = calcular_mase_agregado(
            df_treino, df_te_norm, preds, y_te_real,
            colunas_grupo, coluna_alvo,
            sazonalidade=sazonalidade,
            tamanho_janela=tamanho_janela,
        )

        linha = {
            "modelo": nome,
            "seed": seed,
            "mae": res["MAE"],
            "rmse": res["RMSE"],
            "mape": res["MAPE"],
            "mase": round(mase, 4) if not np.isnan(mase) else None,
        }

        # Grava imediatamente — Colab pode desconectar a qualquer momento
        df_linha = pd.DataFrame([linha])
        cabecalho = not os.path.exists(caminho_csv) or os.path.getsize(caminho_csv) == 0
        df_linha.to_csv(caminho_csv, mode="a", header=cabecalho, index=False)

        print(f"    MAE={linha['mae']:.4f}  RMSE={linha['rmse']:.4f}  "
              f"MAPE={linha['mape']:.2f}%  MASE={linha['mase']}")
        resultados.append(linha)

    return pd.DataFrame(resultados)


def resumir_resultados(
    df_ou_caminho: "pd.DataFrame | str" = CAMINHO_CSV_PADRAO,
    modelos: list[str] | None = None,
) -> pd.DataFrame:
    """
    Agrega média ± desvio padrão por modelo a partir do CSV ou DataFrame.
    Imprime e retorna o resumo.
    """
    if isinstance(df_ou_caminho, str):
        df = pd.read_csv(df_ou_caminho)
    else:
        df = df_ou_caminho.copy()

    if modelos:
        df = df[df["modelo"].isin(modelos)]

    metricas_numericas = ["mae", "rmse", "mape", "mase"]
    metricas_presentes = [m for m in metricas_numericas if m in df.columns]

    agg = {}
    for col in metricas_presentes:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")

    resumo = (
        df.groupby("modelo")
        .agg(**agg)
        .round(4)
        .reset_index()
    )

    print("\nResumo por modelo (média ± desvio padrão):")
    for _, row in resumo.iterrows():
        partes = [f"  {row['modelo']}"]
        for m in metricas_presentes:
            partes.append(f"{m.upper()}={row[f'{m}_mean']:.4f}±{row[f'{m}_std']:.4f}")
        print("  ".join(partes))

    return resumo
