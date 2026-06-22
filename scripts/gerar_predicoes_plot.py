"""
Reconstroi y_real e gera outputs/figures/predicoes_comparativo.png.

Usa Condicao B: criar_janelas com colunas_grupo=['store','item'],
identica ao pipeline que gerou os *_preds.npy no experimento multi-seed.

Fluxo:
  1. Reconstroi y_real sem normalizar (df_teste pre-normalizacao)
  2. Carrega MLP_preds.npy, LSTM_preds.npy, GRU_preds.npy de checkpoints/
  3. Valida alinhamento: MAPE de cada modelo (ref: MLP~16%, LSTM~12%, GRU~12%)
  4. Se OK, plota primeiros 90 dias da serie (store=1, item=1)
  5. Salva em outputs/figures/predicoes_comparativo.png
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.comparativo import plotar_predicoes
from src.evaluation.metricas import calcular_metricas
from src.pipeline.loader import carregar_serie_temporal
from src.pipeline.preprocessor import preparar_dataset
from src.pipeline.splitter import criar_janelas, dividir_treino_teste

COLUNA_ALVO = "sales"
COLUNAS_GRUPO = ["store", "item"]
TAMANHO_JANELA = 30
CAMINHO_CSV = "data/raw/train.csv"
CHECKPOINT_DIR = "checkpoints"
DIAS_PLOT = 90
# Tolerancia em pontos percentuais para aceitar o alinhamento
TOLERANCIA_MAPE = 3.0
# Referencias dos experimentos multi-seed (media de 5 seeds)
REFERENCIAS_MAPE = {"MLP": 15.82, "LSTM": 12.21, "GRU": 12.18}


def reconstruir_y_real() -> np.ndarray:
    """Reconstroi y_real aplicando o pipeline identico ao main.py, sem normalizar."""
    print("Carregando CSV...")
    df = carregar_serie_temporal(CAMINHO_CSV)

    print("Pre-processando...")
    df_proc, features = preparar_dataset(df, COLUNA_ALVO, COLUNAS_GRUPO)

    print("Dividindo treino/teste...")
    df_treino, df_teste = dividir_treino_teste(df_proc, data_corte=None)

    # Janelas no df_teste PRE-normalizacao: y retorna na escala original de vendas
    print("Criando janelas (Condicao B: com colunas_grupo)...")
    _, y_real = criar_janelas(
        df_teste, COLUNA_ALVO, features, TAMANHO_JANELA, COLUNAS_GRUPO
    )

    return y_real, df_teste, features


def encontrar_offset_serie(df_teste: "pd.DataFrame", store: int, item: int) -> tuple[int, int]:
    """
    Calcula o offset e o numero de janelas da serie (store, item) no array global.
    Percorre os grupos na mesma ordem que criar_janelas usaria (groupby sort=False).
    """
    offset = 0
    n_janelas_alvo = 0

    for (s, i), grupo in df_teste.groupby(COLUNAS_GRUPO, sort=False):
        n_janelas = max(0, len(grupo) - TAMANHO_JANELA)
        if s == store and i == item:
            n_janelas_alvo = n_janelas
            break
        offset += n_janelas

    return offset, n_janelas_alvo


def main() -> None:
    # 1. Reconstroi y_real
    y_real, df_teste, features = reconstruir_y_real()
    print(
        f"\ny_real: {len(y_real):,} pontos  "
        f"min={y_real.min():.1f}  max={y_real.max():.1f}  media={y_real.mean():.1f}"
    )

    # 2. Carrega predicoes salvas
    print("\nCarregando predicoes...")
    preds: dict[str, np.ndarray] = {}
    arquivos = {
        "MLP": "MLP_preds.npy",
        "LSTM": "LSTM_preds.npy",
        "GRU": "GRU_preds.npy",
    }
    for nome, arq in arquivos.items():
        caminho = os.path.join(CHECKPOINT_DIR, arq)
        if not os.path.exists(caminho):
            print(f"  ERRO: {caminho} nao encontrado.")
            print(f"  Coloque {arq} em {CHECKPOINT_DIR}/ e rode novamente.")
            sys.exit(1)
        preds[nome] = np.load(caminho).astype(float)
        print(f"  {nome}: {len(preds[nome]):,} pontos de {arq}")

    # 3. Valida alinhamento via MAPE em todos os pontos
    print("\nValidando alinhamento (MAPE sobre pontos totais)...")
    alinhamento_ok = True
    for nome, pred in preds.items():
        n = min(len(y_real), len(pred))
        res = calcular_metricas(y_real[:n], pred[:n], nome)
        ref = REFERENCIAS_MAPE[nome]
        diff = abs(res["MAPE"] - ref)
        status = "OK" if diff <= TOLERANCIA_MAPE else "FALHOU"
        print(
            f"  {nome}: MAE={res['MAE']:.2f}  RMSE={res['RMSE']:.2f}  "
            f"MAPE={res['MAPE']:.2f}%  (ref ~{ref:.1f}%)  [{status}]"
        )
        if status == "FALHOU":
            alinhamento_ok = False

    if not alinhamento_ok:
        print(
            "\nALINHAMENTO FALHOU: y_real desalinhado com as predicoes."
            " Figura NAO gerada."
        )
        sys.exit(1)

    print("Alinhamento confirmado.")

    # 4. Localiza serie (store=1, item=1) nos arrays globais
    offset, n_janelas_serie = encontrar_offset_serie(df_teste, store=1, item=1)
    print(
        f"\nSerie (store=1, item=1): "
        f"offset={offset}  janelas={n_janelas_serie}  "
        f"plotando primeiros {DIAS_PLOT} dias"
    )

    if n_janelas_serie == 0:
        print("ERRO: serie (store=1, item=1) nao encontrada em df_teste.")
        sys.exit(1)

    # Fatia a serie e limita a DIAS_PLOT
    fim = min(offset + n_janelas_serie, offset + DIAS_PLOT)
    y_plot = y_real[offset:fim]
    preds_plot = {nome: arr[offset:fim] for nome, arr in preds.items()}

    # 5. Gera e salva figura
    plotar_predicoes(
        y_real=y_plot,
        predicoes_por_modelo=preds_plot,
        titulo="Previsao de Vendas: Real vs Modelos Neurais (store=1, item=1)",
        dias=DIAS_PLOT,
    )


if __name__ == "__main__":
    main()
