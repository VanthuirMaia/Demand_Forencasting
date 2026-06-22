"""
Valida _mase_tft sem re-treinar nem usar checkpoint de GPU.

Abordagem:
  1. Reconstroi y_real do jeito identico a executar_tft_multiseed.
  2. Verifica que groupby(sort=True) sobre df_treino da a mesma sequencia
     que sort_values(colunas_grupo) sobre df_proc — essa e a hipotese de
     ordenacao assumida em _mase_tft.
  3. Calcula o denominador sNaive medio das 500 series no treino.
  4. Testa _mase_tft com preds=y_real (MASE esperado = 0).
  5. Testa com preds=y_real + offset fixo e compara com MASE analitico.
  6. Imprime o MASE esperado para MAE=28.13 e confronta com ref ~3.64.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.multi_seed_tft import _mase_tft
from src.pipeline.loader import carregar_serie_temporal
from src.pipeline.preprocessor import preparar_dataset
from src.pipeline.splitter import dividir_treino_teste

COLUNA_ALVO = "sales"
COLUNAS_GRUPO = ["store", "item"]
TAMANHO_PREDICAO = 1
SAZONALIDADE = 7

print("Carregando e pre-processando dados...")
df = carregar_serie_temporal("data/raw/train.csv")
df_proc, features = preparar_dataset(df, COLUNA_ALVO, COLUNAS_GRUPO)
df_treino, df_teste = dividir_treino_teste(df_proc, data_corte=None)

# y_real extraido da mesma forma que executar_tft_multiseed faz internamente
data_inicio_val = df_proc["date"].max() - pd.Timedelta(days=TAMANHO_PREDICAO - 1)
y_real = (
    df_proc[df_proc["date"] >= data_inicio_val]
    .sort_values(["date"] + COLUNAS_GRUPO)[COLUNA_ALVO]
    .values.astype(float)
)
print(f"\ny_real: {len(y_real)} series  min={y_real.min():.1f}  max={y_real.max():.1f}  media={y_real.mean():.1f}")

# --- Validacao de ordenacao ---
# _mase_tft usa groupby(sort=True) sobre df_treino (colunas inteiras).
# y_real foi extraido por sort_values(["date","store","item"]) sobre df_proc.
# Como df_proc tem store/item como inteiros, sort_values da ordem numerica crescente.
# groupby(sort=True) com inteiros tambem da ordem numerica crescente.
# Os dois devem coincidir.
grupos_sort_true = [
    k for k, _ in df_treino.groupby(COLUNAS_GRUPO, sort=True)
]
grupos_sort_values = list(
    df_proc[df_proc["date"] >= data_inicio_val]
    .sort_values(COLUNAS_GRUPO)
    .groupby(COLUNAS_GRUPO, sort=False)
    .groups.keys()
)

print("\nValidando ordenacao...")
print(f"  Primeiros 5 grupos (groupby sort=True):  {grupos_sort_true[:5]}")
print(f"  Primeiros 5 grupos (sort_values):        {grupos_sort_values[:5]}")
if grupos_sort_true == grupos_sort_values:
    print("  Ordenacao OK: groupby(sort=True) == sort_values numerico.")
else:
    # Mostra onde diverge
    divergencias = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(grupos_sort_true, grupos_sort_values))
        if a != b
    ]
    print(f"  ATENCAO: ordenacao diverge em {len(divergencias)} posicoes.")
    print(f"  Primeiras divergencias: {divergencias[:5]}")

# --- Denominador sNaive medio por serie ---
denoms = []
for _, g in df_treino.groupby(COLUNAS_GRUPO, sort=True):
    vals = g.sort_values("date")[COLUNA_ALVO].values
    if len(vals) > SAZONALIDADE:
        d = float(np.mean(np.abs(vals[SAZONALIDADE:] - vals[:-SAZONALIDADE])))
        if d > 0:
            denoms.append(d)

denom_medio = float(np.mean(denoms))
print(f"\nDenominador sNaive medio (500 series): {denom_medio:.4f} unid/dia")
print(f"  MASE esperado para MAE=28.13: {28.13 / denom_medio:.4f}  (ref seed 42: ~3.64)")
print(f"  MASE esperado para MAE=28.16: {28.16 / denom_medio:.4f}  (ref media 5 seeds: ~3.66)")

# --- Teste 1: preds = y_real (MASE deve ser 0) ---
mase_zero = _mase_tft(
    df_treino, y_real.copy(), y_real.copy(),
    COLUNAS_GRUPO, COLUNA_ALVO, TAMANHO_PREDICAO, SAZONALIDADE,
)
print(f"\nTeste 1 (preds=y_real):  MASE = {mase_zero:.6f}  (esperado: 0.000000)")

# --- Teste 2: preds = y_real + offset uniforme ---
# MAE por serie = offset (constante), MASE analitico = offset / denom_medio
OFFSET = 28.13
preds_mock = y_real + OFFSET
mase_mock = _mase_tft(
    df_treino, preds_mock, y_real,
    COLUNAS_GRUPO, COLUNA_ALVO, TAMANHO_PREDICAO, SAZONALIDADE,
)
mase_analitico = OFFSET / denom_medio
print(f"Teste 2 (preds=y_real+{OFFSET}): MASE = {mase_mock:.4f}  analitico = {mase_analitico:.4f}")

# --- Resultado final ---
print("\nResumo:")
print(f"  Denominador medio:          {denom_medio:.4f} unid/dia")
print(f"  MASE analitico (MAE=28.13): {28.13/denom_medio:.4f}  ref seed 42:    ~3.64")
print(f"  MASE analitico (MAE=28.16): {28.16/denom_medio:.4f}  ref media seeds: ~3.66")
delta = abs(28.13 / denom_medio - 3.64)
if delta < 0.10:
    print("  Denominador consistente com os valores de referencia.")
else:
    print(f"  ATENCAO: denominador inconsistente (delta={delta:.4f}).")
