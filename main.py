import os
import json
import random

import joblib
import numpy as np
import tensorflow as tf
import torch

# ── Reprodutibilidade ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

from src.pipeline.loader import carregar_serie_temporal, validar_serie_temporal
from src.pipeline.preprocessor import preparar_dataset
from src.pipeline.splitter import criar_janelas, dividir_treino_teste, normalizar

from src.models.arima import (
    calcular_metricas,
    construir_e_treinar_arima,
    preparar_serie_arima,
    prever_arima,
)
from src.models.gru import construir_gru, prever_gru, treinar_gru
from src.models.lstm import construir_lstm, prever_lstm, treinar_lstm
from src.models.mlp import construir_mlp, prever_mlp, treinar_mlp
from src.models.tft import (
    construir_tft,
    preparar_dataset_tft,
    prever_tft,
    treinar_tft,
)

from src.evaluation.metricas import comparar_modelos
from src.evaluation.comparativo import (
    gerar_relatorio,
    plotar_comparativo_metricas,
    plotar_predicoes,
)

# ── Configurações ─────────────────────────────────────────────────────────────
CAMINHO_DADOS = "data/raw/train.csv"
COLUNA_ALVO = "sales"
COLUNAS_GRUPO = ["store", "item"]
TAMANHO_JANELA = 30
DATA_CORTE = None  # None = automático (80/20 temporal)

EPOCHS_NEURAIS = 50
BATCH_NEURAL = 512
EPOCHS_TFT = 50
BATCH_TFT = 64
# Horizonte 1 alinha TFT com MLP/LSTM/GRU; use 7 para reproduzir experimento anterior
TFT_HORIZONTE = 1

CHECKPOINT_DIR = "checkpoints"
TFT_CKPT_DIR = os.path.join(CHECKPOINT_DIR, "tft_lightning")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TFT_CKPT_DIR, exist_ok=True)

# ── Detecção de GPU ───────────────────────────────────────────────────────────
def _detectar_acelerador_torch() -> str:
    if torch.cuda.is_available():
        print(f"  PyTorch/TFT: GPU detectada — {torch.cuda.get_device_name(0)}")
        return "gpu"
    print("  PyTorch/TFT: nenhuma GPU detectada, usando CPU.")
    return "cpu"

def _verificar_gpu_tensorflow():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  TensorFlow: {len(gpus)} GPU(s) disponível(is)")
        for g in gpus:
            print(f"    {g.name}")
    else:
        print("  TensorFlow: nenhuma GPU detectada, usando CPU.")

print("Verificando aceleradores disponíveis...")
_verificar_gpu_tensorflow()
ACELERADOR_TORCH = _detectar_acelerador_torch()

# ── Helpers de checkpoint ─────────────────────────────────────────────────────
def _ckpt_result_path(nome: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{nome}_result.json")

def _ckpt_preds_path(nome: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{nome}_preds.npy")

def _resultado_salvo(nome: str) -> bool:
    return os.path.exists(_ckpt_result_path(nome))

def _salvar_resultado(nome: str, res: dict, preds: np.ndarray | None = None):
    safe = {k: float(v) if hasattr(v, "item") else v for k, v in res.items()}
    with open(_ckpt_result_path(nome), "w") as f:
        json.dump(safe, f, indent=2)
    if preds is not None:
        np.save(_ckpt_preds_path(nome), preds)
    print(f"  Checkpoint salvo: {nome}")

def _carregar_resultado(nome: str) -> tuple[dict, np.ndarray | None]:
    with open(_ckpt_result_path(nome)) as f:
        res = json.load(f)
    preds_path = _ckpt_preds_path(nome)
    preds = np.load(preds_path) if os.path.exists(preds_path) else None
    return res, preds

# ── [1/6] Carregando e validando dados ────────────────────────────────────────
print("\n[1/6] Carregando dados...")
df = carregar_serie_temporal(CAMINHO_DADOS)
relatorio = validar_serie_temporal(df, COLUNAS_GRUPO)
if not relatorio["valido"]:
    raise RuntimeError(f"Validacao falhou: {relatorio}")
print(f"  {len(df):,} linhas carregadas. Validacao OK.")

# ── [2/6] Pré-processamento ───────────────────────────────────────────────────
print("\n[2/6] Pre-processando...")
df_proc, features = preparar_dataset(df, COLUNA_ALVO, COLUNAS_GRUPO)

# ── [3/6] Divisão, normalização e janelas ────────────────────────────────────
print("\n[3/6] Dividindo e normalizando...")
df_treino, df_teste = dividir_treino_teste(df_proc, data_corte=DATA_CORTE)
df_tr_norm, df_te_norm, sc_feat, sc_alvo = normalizar(
    df_treino, df_teste, features, COLUNA_ALVO
)
X_tr, y_tr = criar_janelas(df_tr_norm, COLUNA_ALVO, features, TAMANHO_JANELA, COLUNAS_GRUPO)
X_te, y_te = criar_janelas(df_te_norm, COLUNA_ALVO, features, TAMANHO_JANELA, COLUNAS_GRUPO)

# y na escala original para calcular métricas dos modelos neurais
y_te_real = sc_alvo.inverse_transform(y_te.reshape(-1, 1)).flatten()

# Salva scaler para eventual recarga dos modelos TF
joblib.dump(sc_alvo, os.path.join(CHECKPOINT_DIR, "sc_alvo.pkl"))

resultados = []
predicoes_por_modelo = {}

# ── [4/6] ARIMA baseline ──────────────────────────────────────────────────────
# ARIMA opera sobre a série agregada (soma de todas as lojas/itens por dia).
# As métricas estão em escala de vendas totais diárias (~20k), enquanto os
# modelos neurais operam em escala de vendas por série (~50). O comparativo
# numérico deve ser interpretado com essa diferença em mente.
print("\n[4/6] Treinando ARIMA...")
if _resultado_salvo("ARIMA"):
    print("  Checkpoint encontrado — pulando treino ARIMA.")
    res_arima, _ = _carregar_resultado("ARIMA")
else:
    data_corte_arima = df_treino["date"].max()
    serie_tr_arima, serie_te_arima = preparar_serie_arima(
        df, data_corte=data_corte_arima
    )
    modelo_arima = construir_e_treinar_arima(serie_tr_arima)
    preds_arima = prever_arima(modelo_arima, serie_te_arima)
    res_arima = calcular_metricas(serie_te_arima.values, preds_arima, "ARIMA")
    _salvar_resultado("ARIMA", res_arima)

resultados.append(res_arima)
print(f"  ARIMA — MAE: {res_arima['MAE']:.2f}  RMSE: {res_arima['RMSE']:.2f}  MAPE: {res_arima['MAPE']:.2f}%")

# ── [5/6] Modelos neurais: MLP, LSTM, GRU ────────────────────────────────────
input_shape = (TAMANHO_JANELA, len(features))

for nome, construir, treinar, prever in [
    ("MLP",  construir_mlp,  treinar_mlp,  prever_mlp),
    ("LSTM", construir_lstm, treinar_lstm, prever_lstm),
    ("GRU",  construir_gru,  treinar_gru,  prever_gru),
]:
    print(f"\n[5/6] Treinando {nome}...")
    if _resultado_salvo(nome):
        print(f"  Checkpoint encontrado — pulando treino {nome}.")
        res, preds = _carregar_resultado(nome)
    else:
        modelo = construir(input_shape=input_shape)
        treinar(modelo, X_tr, y_tr, epochs=EPOCHS_NEURAIS, batch_size=BATCH_NEURAL)
        preds = prever(modelo, X_te, sc_alvo)
        res = calcular_metricas(y_te_real, preds, nome)
        _salvar_resultado(nome, res, preds)

    resultados.append(res)
    predicoes_por_modelo[nome] = preds
    print(f"  {nome} — MAE: {res['MAE']:.2f}  RMSE: {res['RMSE']:.2f}  MAPE: {res['MAPE']:.2f}%")

# ── [5/6] TFT ────────────────────────────────────────────────────────────────
# Subconjunto com as 10 primeiras lojas e 10 primeiros itens (100 séries),
# reduzindo ~876k para ~175k amostras e tornando o treino viável em CPU.
print("\n[5/6] Treinando TFT...")
if _resultado_salvo("TFT"):
    print("  Checkpoint encontrado — pulando treino TFT.")
    res_tft, preds_tft = _carregar_resultado("TFT")
else:
    df_proc_tft = df_proc
    ds_treino_tft, ds_val_tft = preparar_dataset_tft(
        df_proc_tft, COLUNA_ALVO, COLUNAS_GRUPO,
        tamanho_predicao=TFT_HORIZONTE,
    )
    modelo_tft = construir_tft(ds_treino_tft)

    # Retoma treino do último checkpoint Lightning, se existir
    tft_last_ckpt = os.path.join(TFT_CKPT_DIR, "last.ckpt")
    ckpt_path = tft_last_ckpt if os.path.exists(tft_last_ckpt) else None
    if ckpt_path:
        print(f"  Retomando treino TFT de: {ckpt_path}")

    trainer_tft, melhor_tft = treinar_tft(
        modelo_tft, ds_treino_tft, ds_val_tft,
        max_epochs=EPOCHS_TFT, batch_size=BATCH_TFT,
        checkpoint_dir=TFT_CKPT_DIR,
        accelerator=ACELERADOR_TORCH,
        ckpt_path=ckpt_path,
    )
    preds_tft = prever_tft(melhor_tft, ds_val_tft, accelerator=ACELERADOR_TORCH)

    # y_real do TFT: últimos TFT_HORIZONTE dias de cada série no subconjunto filtrado
    data_inicio_val = df_proc_tft["date"].max() - np.timedelta64(TFT_HORIZONTE - 1, "D")
    y_real_tft = (
        df_proc_tft[df_proc_tft["date"] >= data_inicio_val]
        .sort_values(["date", "store", "item"])[COLUNA_ALVO]
        .values.astype(float)
    )
    min_len = min(len(preds_tft), len(y_real_tft))
    res_tft = calcular_metricas(y_real_tft[:min_len], preds_tft[:min_len], "TFT")
    _salvar_resultado("TFT", res_tft, preds_tft)

resultados.append(res_tft)
predicoes_por_modelo["TFT"] = preds_tft
print(f"  TFT — MAE: {res_tft['MAE']:.2f}  RMSE: {res_tft['RMSE']:.2f}  MAPE: {res_tft['MAPE']:.2f}%")

# ── [6/6] Comparativo final ───────────────────────────────────────────────────
print("\n[6/6] Gerando comparativo...")
df_comparativo = comparar_modelos(resultados)

gerar_relatorio(df_comparativo)
plotar_comparativo_metricas(df_comparativo)

# Visualização das predições dos modelos neurais (escala por série)
plotar_predicoes(
    y_real=y_te_real,
    predicoes_por_modelo=predicoes_por_modelo,
    titulo="Previsao de Vendas por Serie: Real vs Modelos Neurais",
)

print("\nPipeline concluido. Resultados em outputs/")
