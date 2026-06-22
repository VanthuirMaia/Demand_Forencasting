"""
Experimento multi-seed para o TFT.

Salva resultados incrementalmente em CSV apos cada seed, permitindo retomada
em caso de interrupcao (mesma logica de multi_seed.py para MLP/LSTM/GRU).

Shim de importacao: pytorch-lightning 2.1.3 (ambiente GPU legado) expoe o
modulo como pytorch_lightning, mas tft.py importa lightning.pytorch. O bloco
try/except aplica o mapeamento antes de qualquer import de tft.py, tornando
este modulo compativel com ambos os ambientes sem alterar tft.py.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

# Shim: deve rodar ANTES de importar tft.py.
# pytorch-lightning 2.1.3 expoe pytorch_lightning; lightning >= 2.x usa lightning.pytorch.
try:
    import lightning.pytorch  # ambiente local ou GPU com lightning >= 2.x
except ImportError:
    import pytorch_lightning as _pl  # ambiente GPU legado com pytorch-lightning 2.1.3
    sys.modules.setdefault("lightning", _pl)
    sys.modules.setdefault("lightning.pytorch", _pl)

from src.evaluation.metricas import calcular_metricas
from src.models.tft import construir_tft, preparar_dataset_tft, prever_tft, treinar_tft

SEEDS_PADRAO = [42, 123, 7, 2024, 99]
CAMINHO_CSV_PADRAO = "resultados/metricas_tft_por_seed.csv"


def _mase_tft(
    df_treino: pd.DataFrame,
    preds: np.ndarray,
    y_real: np.ndarray,
    colunas_grupo: list[str],
    coluna_alvo: str,
    tamanho_predicao: int = 1,
    sazonalidade: int = 7,
    coluna_data: str = "date",
) -> float:
    """
    MASE medio para TFT: tamanho_predicao pontos por serie.

    Denominador = MAE sNaive in-sample no treino por serie (mesma logica de baselines.py).
    Diferenca de calcular_mase_agregado: aqui cada serie contribui tamanho_predicao pontos
    (nao janelas deslizantes), pois o TFT avalia so os ultimos tamanho_predicao dias.

    Assume que preds e y_real estao em ordem numerica crescente de (store, item),
    identica a extracao via sort_values(["date"] + colunas_grupo) sobre df_proc inteiro.
    sort=True abaixo garante a mesma ordenacao no agrupamento de df_treino.
    """
    mases = []
    offset = 0

    for _, g_treino in df_treino.groupby(colunas_grupo, sort=True):
        fim = offset + tamanho_predicao
        if fim > len(preds):
            break

        preds_i = preds[offset:fim]
        y_real_i = y_real[offset:fim]
        offset += tamanho_predicao

        treino_vals = g_treino.sort_values(coluna_data)[coluna_alvo].values
        if len(treino_vals) <= sazonalidade:
            continue

        denom = float(np.mean(
            np.abs(treino_vals[sazonalidade:] - treino_vals[:-sazonalidade])
        ))
        if denom == 0:
            continue

        mae_i = float(np.mean(np.abs(y_real_i - preds_i)))
        mases.append(mae_i / denom)

    return float(np.mean(mases)) if mases else float("nan")


def executar_tft_multiseed(
    df_proc: pd.DataFrame,
    df_treino: pd.DataFrame,
    coluna_alvo: str = "sales",
    colunas_grupo: list[str] | None = None,
    seeds: list[int] | None = None,
    max_epochs: int = 50,
    batch_size: int = 64,
    tamanho_encoder: int = 30,
    tamanho_predicao: int = 1,
    hidden_size: int = 128,
    checkpoint_dir: str = "checkpoints/tft_lightning",
    accelerator: str = "auto",
    sazonalidade: int = 7,
    caminho_csv: str = CAMINHO_CSV_PADRAO,
) -> pd.DataFrame:
    """
    Treina o TFT para cada seed e salva resultados incrementalmente em CSV.

    df_proc: DataFrame processado completo (todas as datas, store/item inteiros, escala original).
    df_treino: parcela de treino de df_proc na escala original, usada para o denominador do MASE.
    Retorna DataFrame com as linhas novas gravadas nesta chamada (vazio se todas ja existiam).
    """
    if colunas_grupo is None:
        colunas_grupo = ["store", "item"]
    if seeds is None:
        seeds = SEEDS_PADRAO

    os.makedirs(os.path.dirname(caminho_csv) or ".", exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if torch.cuda.is_available():
        print(f"  PyTorch/TFT: GPU disponivel — {torch.cuda.get_device_name(0)}")
    else:
        print("  PyTorch/TFT: nenhuma GPU detectada, usando CPU.")

    # Carrega seeds ja concluidas para retomada incremental
    seeds_feitas: set[int] = set()
    if os.path.exists(caminho_csv):
        df_existente = pd.read_csv(caminho_csv)
        seeds_feitas = set(
            df_existente.loc[df_existente["modelo"] == "TFT", "seed"].astype(int)
        )
        if seeds_feitas:
            print(f"  Seeds ja concluidas para TFT: {sorted(seeds_feitas)} — pulando.")

    # y_real: ultimos tamanho_predicao dias de cada serie, ordem numerica crescente
    data_inicio_val = df_proc["date"].max() - pd.Timedelta(days=tamanho_predicao - 1)
    y_real_tft = (
        df_proc[df_proc["date"] >= data_inicio_val]
        .sort_values(["date"] + colunas_grupo)[coluna_alvo]
        .values.astype(float)
    )

    # Estrutura do dataset e identica para todas as seeds — so os pesos mudam
    ds_treino, ds_val = preparar_dataset_tft(
        df_proc, coluna_alvo, colunas_grupo,
        tamanho_encoder=tamanho_encoder,
        tamanho_predicao=tamanho_predicao,
    )

    resultados = []

    for seed in seeds:
        if seed in seeds_feitas:
            continue

        print(f"\n  [TFT] seed={seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        modelo = construir_tft(ds_treino, hidden_size=hidden_size)
        _, melhor = treinar_tft(
            modelo, ds_treino, ds_val,
            max_epochs=max_epochs,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            accelerator=accelerator,
        )
        preds = prever_tft(melhor, ds_val, accelerator=accelerator)

        n = min(len(y_real_tft), len(preds))
        res = calcular_metricas(y_real_tft[:n], preds[:n], "TFT")

        mase = _mase_tft(
            df_treino, preds[:n], y_real_tft[:n],
            colunas_grupo, coluna_alvo,
            tamanho_predicao=tamanho_predicao,
            sazonalidade=sazonalidade,
        )

        linha = {
            "modelo": "TFT",
            "seed": seed,
            "mae": res["MAE"],
            "rmse": res["RMSE"],
            "mape": res["MAPE"],
            "mase": round(mase, 4) if not np.isnan(mase) else None,
        }

        # Grava imediatamente para sobreviver a interrupcoes
        df_linha = pd.DataFrame([linha])
        cabecalho = not os.path.exists(caminho_csv) or os.path.getsize(caminho_csv) == 0
        df_linha.to_csv(caminho_csv, mode="a", header=cabecalho, index=False)

        print(
            f"    MAE={linha['mae']:.4f}  RMSE={linha['rmse']:.4f}  "
            f"MAPE={linha['mape']:.2f}%  MASE={linha['mase']}"
        )
        resultados.append(linha)

    return pd.DataFrame(resultados)


def resumir_resultados_tft(caminho_csv: str = CAMINHO_CSV_PADRAO) -> pd.DataFrame:
    """Agrega media e desvio padrao das seeds do TFT a partir do CSV salvo."""
    df = pd.read_csv(caminho_csv)
    df = df[df["modelo"] == "TFT"]

    metricas = [c for c in ["mae", "rmse", "mape", "mase"] if c in df.columns]
    agg = {}
    for m in metricas:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")

    resumo = df.groupby("modelo").agg(**agg).round(4).reset_index()

    print("\nTFT — resumo multi-seed (media +- desvio padrao):")
    row = resumo.iloc[0]
    partes = ["  TFT"]
    for m in metricas:
        partes.append(f"{m.upper()}={row[f'{m}_mean']:.4f}+-{row[f'{m}_std']:.4f}")
    print("  ".join(partes))

    return resumo
