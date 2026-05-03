# Demand Forecasting com Deep Learning

Projeto acadêmico (UPE — Redes Neurais) de previsão de demanda comparando modelos de deep learning (MLP, LSTM, GRU, TFT) contra baseline estatístico (ARIMA).

## Resultados finais

### Modelos neurais — escala por série (~50 unid/dia)

| Modelo | MAE   | RMSE  | MAPE    |
|--------|-------|-------|---------|
| TFT    | 22.99 | 28.93 | 85.47%  |
| GRU    | 23.57 | 28.99 | 56.72%  |
| LSTM   | 23.60 | 29.01 | 56.90%  |
| MLP    | 23.64 | 29.46 | 54.59%  |

### Baseline — série agregada (~20.000 unid/dia)

| Modelo | MAE     | RMSE      | MAPE   |
|--------|---------|-----------|--------|
| ARIMA  | 9234.06 | 10961.15  | 28.43% |

> MAPE é a métrica comparável entre os dois grupos (escalas diferentes). Os modelos neurais operam por série individual; o ARIMA opera na soma diária de todas as lojas/itens.

## Dataset

- **Fonte:** Kaggle — Store Item Demand Forecasting
- **Arquivo:** `data/raw/train.csv`
- **Colunas:** `date`, `store`, `item`, `sales`
- **Escala:** 913.000 linhas brutas, 500 séries (10 lojas × 50 itens), período 2013–2017
- **Após pré-processamento:** 898.000 linhas, 14 features geradas

## Como executar

```bash
# Ativar ambiente e rodar pipeline completo
.venv/Scripts/python main.py
```

O pipeline executa 6 etapas com progresso impresso: carregar → pré-processar → dividir/normalizar → ARIMA → MLP/LSTM/GRU → TFT → comparativo.

Saídas geradas em `outputs/`:
- `comparativo_resultados.csv`
- `figures/comparativo_metricas.png`
- `figures/predicoes_comparativo.png`

## Estrutura

```
src/
  pipeline/   — loader, preprocessor, splitter
  models/     — mlp, lstm, gru, arima, tft
  evaluation/ — metricas, comparativo
data/raw/     — train.csv
outputs/      — figuras e CSV de resultados
article/      — artigo.md (artigo acadêmico)
main.py       — pipeline completo
```

## Ambiente

- Python 3.12.1 · TensorFlow 2.21.0 · PyTorch 2.11.0+cpu · pytorch-forecasting 1.7.0
- Treino em CPU (TensorFlow ≥ 2.11 sem suporte a GPU nativa no Windows)
- Reprodutibilidade garantida via `SEED = 42` (random, numpy, TensorFlow, PyTorch)

