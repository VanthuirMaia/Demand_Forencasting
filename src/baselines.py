"""
Baseline Seasonal Naïve e métrica MASE por série.

Regras de cálculo:
- Denominador do MASE = MAE sNaïve in-sample no TREINO daquela série (nunca global).
- Numerador = MAE das predições do modelo no TESTE para a mesma série.
- Agregação = média do MASE_i entre as N séries.
- sNaïve no teste usa a cauda do treino para não perder os primeiros dias do teste
  na fronteira da divisão temporal.
"""

import numpy as np
import pandas as pd


# ── Seasonal Naïve ────────────────────────────────────────────────────────────

def snaive_por_serie(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    colunas_grupo: list[str],
    coluna_alvo: str,
    coluna_data: str = "date",
    sazonalidade: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Seasonal Naïve: pred(t) = y(t - sazonalidade), dentro de cada grupo.
    Usa os últimos `sazonalidade` dias do treino para cobrir o início do teste.

    Retorna (y_real, y_pred) concatenados na ordem dos grupos
    (groupby sort=False, mesma convenção de criar_janelas).
    """
    y_real_list, y_pred_list = [], []

    for grupo_vals, g_treino in df_treino.groupby(colunas_grupo, sort=False):
        g_teste = _filtrar_grupo(df_teste, colunas_grupo, grupo_vals)
        if len(g_teste) == 0:
            continue

        treino_ord = g_treino.sort_values(coluna_data)[coluna_alvo].values
        teste_ord = g_teste.sort_values(coluna_data)[coluna_alvo].values

        # cauda do treino garante a previsão dos primeiros `sazonalidade` dias do teste
        cauda = treino_ord[-sazonalidade:]
        combinado = np.concatenate([cauda, teste_ord])
        # shift de sazonalidade: combinado[:-s] alinhado com teste_ord
        preds = combinado[: len(teste_ord)]

        y_real_list.append(teste_ord)
        y_pred_list.append(preds)

    return np.concatenate(y_real_list), np.concatenate(y_pred_list)


# ── MASE ──────────────────────────────────────────────────────────────────────

def calcular_mase_agregado(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    preds_flat: np.ndarray,
    y_real_flat: np.ndarray,
    colunas_grupo: list[str],
    coluna_alvo: str,
    coluna_data: str = "date",
    sazonalidade: int = 7,
    tamanho_janela: int = 30,
) -> float:
    """
    MASE médio entre séries para modelos baseados em janelas deslizantes.

    Para cada série i:
        denominador_i = MAE sNaïve in-sample no TREINO de i
        MASE_i        = MAE(preds_i no teste) / denominador_i
    Retorna média de MASE_i.

    preds_flat e y_real_flat devem seguir a ordem de criar_janelas (Condição B):
    grupos em groupby(colunas_grupo, sort=False) sobre df_teste,
    dentro de cada grupo ordenado por data, janelas de tamanho_janela passos.

    df_treino deve estar na escala original (para o denominador sNaïve).
    df_teste é usado apenas para estrutura de grupos e contagem de janelas.
    """
    mases = []
    offset = 0

    for grupo_vals, g_treino in df_treino.groupby(colunas_grupo, sort=False):
        g_teste = _filtrar_grupo(df_teste, colunas_grupo, grupo_vals)
        n_janelas = max(0, len(g_teste) - tamanho_janela)
        if n_janelas <= 0:
            continue

        preds_i = preds_flat[offset: offset + n_janelas]
        y_real_i = y_real_flat[offset: offset + n_janelas]
        offset += n_janelas

        # Denominador: MAE sNaïve in-sample no treino (nunca no teste, nunca global)
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


def calcular_mase_snaive(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    colunas_grupo: list[str],
    coluna_alvo: str,
    coluna_data: str = "date",
    sazonalidade: int = 7,
) -> float:
    """
    MASE do próprio sNaïve no teste, denominador in-sample no treino por série.
    Usa o conjunto de teste completo (sem janelamento).
    """
    mases = []

    for grupo_vals, g_treino in df_treino.groupby(colunas_grupo, sort=False):
        g_teste = _filtrar_grupo(df_teste, colunas_grupo, grupo_vals)
        if len(g_teste) == 0:
            continue

        treino_vals = g_treino.sort_values(coluna_data)[coluna_alvo].values
        teste_vals = g_teste.sort_values(coluna_data)[coluna_alvo].values

        if len(treino_vals) <= sazonalidade:
            continue

        denom = float(np.mean(
            np.abs(treino_vals[sazonalidade:] - treino_vals[:-sazonalidade])
        ))
        if denom == 0:
            continue

        cauda = treino_vals[-sazonalidade:]
        combinado = np.concatenate([cauda, teste_vals])
        preds = combinado[: len(teste_vals)]

        mae_i = float(np.mean(np.abs(teste_vals - preds)))
        mases.append(mae_i / denom)

    return float(np.mean(mases)) if mases else float("nan")


# ── Utilitário interno ────────────────────────────────────────────────────────

def _filtrar_grupo(
    df: pd.DataFrame,
    colunas_grupo: list[str],
    grupo_vals,
) -> pd.DataFrame:
    """Filtra o DataFrame para um grupo específico."""
    if not isinstance(grupo_vals, tuple):
        grupo_vals = (grupo_vals,)
    mask = np.ones(len(df), dtype=bool)
    for col, val in zip(colunas_grupo, grupo_vals):
        mask &= df[col].values == val
    return df[mask]
