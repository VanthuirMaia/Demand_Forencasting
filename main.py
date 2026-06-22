import argparse
import os
import json

import joblib
import numpy as np

import tensorflow as tf

from src.pipeline.loader import carregar_serie_temporal, validar_serie_temporal
from src.pipeline.preprocessor import preparar_dataset
from src.pipeline.splitter import criar_janelas, dividir_treino_teste, normalizar

from src.models.arima import (
    calcular_metricas,
    construir_e_treinar_arima,
    preparar_serie_arima,
    prever_arima,
)
from src.models.mlp import construir_mlp, treinar_mlp, prever_mlp
from src.models.lstm import construir_lstm, treinar_lstm, prever_lstm
from src.models.gru import construir_gru, treinar_gru, prever_gru
from src.experiments.multi_seed import executar_experimento
from src.baselines import snaive_por_serie, calcular_mase_snaive

from src.evaluation.metricas import comparar_modelos
from src.evaluation.comparativo import (
    gerar_relatorio,
    plotar_comparativo_metricas,
    plotar_predicoes,
)

# ── Argparse ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Demand Forecasting Pipeline")
parser.add_argument(
    "--modelo",
    choices=["arima", "mlp", "lstm", "gru", "tft", "todos"],
    default="todos",
    help="Modelo(s) a treinar (default: todos)",
)
parser.add_argument(
    "--drive_path",
    type=str,
    default=None,
    help="Caminho alternativo para o CSV de métricas por seed (ex.: Google Drive no Colab)",
)
args = parser.parse_args()

RODAR = args.modelo

def _deve_rodar(nome: str) -> bool:
    return RODAR == "todos" or RODAR == nome.lower()

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

SEEDS = [42, 123, 7, 2024, 99]
# --drive_path sobrescreve o caminho padrão local (útil no Colab com Google Drive)
CAMINHO_CSV_SEEDS = args.drive_path if args.drive_path else "checkpoints/metricas_por_seed.csv"

CHECKPOINT_DIR = "checkpoints"
TFT_CKPT_DIR = os.path.join(CHECKPOINT_DIR, "tft_lightning")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TFT_CKPT_DIR, exist_ok=True)

# ── Detecção de GPU (TensorFlow) ─────────────────────────────────────────────
def _verificar_gpu_tensorflow():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  TensorFlow: {len(gpus)} GPU(s) disponível(is)")
        for g in gpus:
            print(f"    {g.name}")
    else:
        print("  TensorFlow: nenhuma GPU detectada, usando CPU.")

print("Verificando aceleradores TensorFlow...")
_verificar_gpu_tensorflow()

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

# sNaïve baseline — calculado logo após a divisão treino/teste
print("  Calculando baseline sNaïve (sazonalidade 7 dias)...")
df_treino_tail7 = df_treino.groupby(COLUNAS_GRUPO).tail(7)
y_real_snaive, y_pred_snaive = snaive_por_serie(
    df_treino_tail7, df_teste, COLUNAS_GRUPO, COLUNA_ALVO
)
res_snaive = calcular_metricas(y_real_snaive, y_pred_snaive, "sNaive")
denominador_mase = calcular_mase_snaive(df_treino, df_teste, COLUNAS_GRUPO, COLUNA_ALVO)
print(f"  sNaïve — MAE: {res_snaive['MAE']:.2f}  RMSE: {res_snaive['RMSE']:.2f}  MAPE: {res_snaive['MAPE']:.2f}%")

resultados = [res_snaive]
predicoes_por_modelo = {}
stats_multi_seed = {}

# ── [4/6] ARIMA baseline ──────────────────────────────────────────────────────
# ARIMA opera sobre a série agregada (soma de todas as lojas/itens por dia).
# As métricas estão em escala de vendas totais diárias (~20k), enquanto os
# modelos neurais operam em escala de vendas por série (~50). O comparativo
# numérico deve ser interpretado com essa diferença em mente.
print("\n[4/6] Treinando ARIMA...")
if not _deve_rodar("arima"):
    print(f"  ARIMA pulado (--modelo={RODAR}).")
    if _resultado_salvo("ARIMA"):
        res_arima, _ = _carregar_resultado("ARIMA")
        resultados.append(res_arima)
        print(f"  ARIMA — MAE: {res_arima['MAE']:.2f}  RMSE: {res_arima['RMSE']:.2f}  MAPE: {res_arima['MAPE']:.2f}%")
elif _resultado_salvo("ARIMA"):
    print("  Checkpoint encontrado — pulando treino ARIMA.")
    res_arima, _ = _carregar_resultado("ARIMA")
    resultados.append(res_arima)
    print(f"  ARIMA — MAE: {res_arima['MAE']:.2f}  RMSE: {res_arima['RMSE']:.2f}  MAPE: {res_arima['MAPE']:.2f}%")
else:
    data_corte_arima = df_treino["date"].max()
    serie_tr_arima, serie_te_arima = preparar_serie_arima(
        df, data_corte=data_corte_arima
    )
    modelo_arima = construir_e_treinar_arima(serie_tr_arima)
    preds_arima = prever_arima(modelo_arima, serie_te_arima)
    res_arima = calcular_metricas(serie_te_arima.values, preds_arima, "ARIMA", mase=None)
    _salvar_resultado("ARIMA", res_arima)
    resultados.append(res_arima)
    print(f"  ARIMA — MAE: {res_arima['MAE']:.2f}  RMSE: {res_arima['RMSE']:.2f}  MAPE: {res_arima['MAPE']:.2f}%")

# ── [5/6] Modelos neurais: MLP, LSTM, GRU (multi-seed) ───────────────────────
# executar_experimento gerencia o loop de seeds, retomada via CSV e MASE interno.
# Não há skip por _resultado_salvo aqui: a retomada é responsabilidade do experimento.
input_shape = (TAMANHO_JANELA, len(features))

_construir = {"MLP": construir_mlp, "LSTM": construir_lstm, "GRU": construir_gru}
_treinar   = {"MLP": treinar_mlp,   "LSTM": treinar_lstm,   "GRU": treinar_gru}
_prever    = {"MLP": prever_mlp,    "LSTM": prever_lstm,    "GRU": prever_gru}

for nome in ["MLP", "LSTM", "GRU"]:
    if not _deve_rodar(nome.lower()):
        print(f"\n[5/6] {nome} pulado (--modelo={RODAR}).")
        if _resultado_salvo(nome):
            res, preds = _carregar_resultado(nome)
            resultados.append(res)
            predicoes_por_modelo[nome] = preds
            print("  Checkpoint seed 42 carregado para comparativo.")
        continue

    print(f"\n[5/6] Treinando {nome} (seeds={SEEDS})...")
    resultado_exp = executar_experimento(
        nome,
        _construir[nome], _treinar[nome], _prever[nome],
        X_tr, y_tr, X_te, y_te_real, sc_alvo, input_shape,
        df_treino=df_treino,
        df_te_norm=df_te_norm,
        colunas_grupo=COLUNAS_GRUPO,
        coluna_alvo=COLUNA_ALVO,
        seeds=SEEDS,
        epochs=EPOCHS_NEURAIS,
        batch_size=BATCH_NEURAL,
        caminho_csv=CAMINHO_CSV_SEEDS,
    )

    df_exp = resultado_exp  # DataFrame [modelo, seed, mae, rmse, mape, mase]
    if df_exp.empty:
        # Todas as seeds já estavam concluídas — carrega do CSV gravado anteriormente
        import pandas as _pd
        df_exp = _pd.read_csv(CAMINHO_CSV_SEEDS)
        df_exp = df_exp[df_exp["modelo"] == nome]
    mask = (df_exp["seed"] == 42)
    if mask.sum() == 0:
        import pandas as _pd
        df_csv = _pd.read_csv(CAMINHO_CSV_SEEDS)
        mask = (df_csv["modelo"] == nome) & (df_csv["seed"] == 42)
        row_seed42 = df_csv[mask].iloc[0]
    else:
        row_seed42 = df_exp[mask].iloc[0]
    res_seed42 = {
        "modelo": nome,
        "MAE": row_seed42["mae"],
        "RMSE": row_seed42["rmse"],
        "MAPE": row_seed42["mape"],
        "MASE": row_seed42["mase"],
    }
    preds_seed42 = None  # executar_experimento não retorna predições
    stats_multi_seed[nome] = {
        "MAE_mean": df_exp["mae"].mean(), "MAE_std": df_exp["mae"].std(),
        "RMSE_mean": df_exp["rmse"].mean(), "RMSE_std": df_exp["rmse"].std(),
        "MAPE_mean": df_exp["mape"].mean(), "MAPE_std": df_exp["mape"].std(),
    }

    _salvar_resultado(nome, res_seed42, preds_seed42)

    resultados.append(res_seed42)
    if preds_seed42 is not None:
        predicoes_por_modelo[nome] = preds_seed42

    print(
        f"  {nome} (seed 42) — MAE: {res_seed42['MAE']:.2f}  "
        f"RMSE: {res_seed42['RMSE']:.2f}  MAPE: {res_seed42['MAPE']:.2f}%"
    )
    stats = stats_multi_seed[nome]
    if stats:
        print(
            f"  {nome} (média±std) — "
            f"MAE: {stats['MAE_mean']:.2f}±{stats['MAE_std']:.2f}  "
            f"RMSE: {stats['RMSE_mean']:.2f}±{stats['RMSE_std']:.2f}  "
            f"MAPE: {stats['MAPE_mean']:.2f}±{stats['MAPE_std']:.2f}%"
        )

# ── [5/6] TFT (multi-seed) ──────────────────────────────────────────────────
CAMINHO_CSV_TFT = "resultados/metricas_tft_por_seed.csv"

print("\n[5/6] Treinando TFT (multi-seed)...")
if not _deve_rodar("tft"):
    print(f"  TFT pulado (--modelo={RODAR}).")
else:
    from src.experiments.multi_seed_tft import executar_tft_multiseed
    executar_tft_multiseed(
        df_proc, df_treino,
        coluna_alvo=COLUNA_ALVO,
        colunas_grupo=COLUNAS_GRUPO,
        seeds=SEEDS,
        max_epochs=EPOCHS_TFT,
        batch_size=BATCH_TFT,
        tamanho_predicao=TFT_HORIZONTE,
        hidden_size=128,
        checkpoint_dir=TFT_CKPT_DIR,
        caminho_csv=CAMINHO_CSV_TFT,
    )

# Agrega resultados TFT do CSV (funciona mesmo se o bloco acima foi pulado)
import pandas as _pd_tft
if os.path.exists(CAMINHO_CSV_TFT):
    _df_tft = _pd_tft.read_csv(CAMINHO_CSV_TFT)
    _df_tft = _df_tft[_df_tft["modelo"] == "TFT"]
    if not _df_tft.empty:
        _row42 = _df_tft[_df_tft["seed"] == 42]
        if not _row42.empty:
            _r = _row42.iloc[0]
            _mase_val = _r.get("mase")
            res_tft = {
                "modelo": "TFT",
                "MAE": float(_r["mae"]),
                "RMSE": float(_r["rmse"]),
                "MAPE": float(_r["mape"]),
                "MASE": float(_mase_val) if _pd_tft.notna(_mase_val) else None,
            }
            resultados.append(res_tft)
            print(
                f"  TFT (seed=42)  MAE: {res_tft['MAE']:.2f}"
                f"  RMSE: {res_tft['RMSE']:.2f}"
                f"  MAPE: {res_tft['MAPE']:.2f}%"
            )
        stats_multi_seed["TFT"] = {
            "MAE_mean": float(_df_tft["mae"].mean()),
            "MAE_std": float(_df_tft["mae"].std()),
            "RMSE_mean": float(_df_tft["rmse"].mean()),
            "RMSE_std": float(_df_tft["rmse"].std()),
            "MAPE_mean": float(_df_tft["mape"].mean()),
            "MAPE_std": float(_df_tft["mape"].std()),
        }

# ── [6/6] Comparativo final ───────────────────────────────────────────────────
print("\n[6/6] Gerando comparativo...")
df_comparativo = comparar_modelos(resultados)

gerar_relatorio(df_comparativo)

if stats_multi_seed:
    print("\n  Estabilidade multi-seed (média ± std):")
    for nome, stats in stats_multi_seed.items():
        print(
            f"    {nome}: MAE {stats['MAE_mean']:.2f}±{stats['MAE_std']:.2f}  "
            f"RMSE {stats['RMSE_mean']:.2f}±{stats['RMSE_std']:.2f}  "
            f"MAPE {stats['MAPE_mean']:.2f}±{stats['MAPE_std']:.2f}%"
        )

plotar_comparativo_metricas(df_comparativo)

if predicoes_por_modelo:
    plotar_predicoes(
        y_real=y_te_real,
        predicoes_por_modelo=predicoes_por_modelo,
        titulo="Previsao de Vendas por Serie: Real vs Modelos Neurais",
    )

print("\nPipeline concluido. Resultados em outputs/")
