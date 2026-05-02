# Projeto RNA — Demand Forecasting

Projeto acadêmico (UPE — Redes Neurais) de previsão de demanda comparando modelos neurais e baseline estatístico.

## Dataset

- **Arquivo:** `data/raw/train.csv`
- **Colunas:** `date`, `store`, `item`, `sales`
- **Escala:** 913.000 linhas brutas, 500 séries (10 lojas × 50 itens), período 2013-01-01 a 2017-12-31
- **Após pré-processamento:** 898.000 linhas, 18 colunas (14 features + date/store/item/sales)
- **Divisão temporal:** treino < 2017-01-06 (718k), teste >= 2017-01-06 (180k)

## Ambiente

- Python 3.12.1, `.venv` local
- Executar com `.venv/Scripts/python` (Windows)
- **Nunca** usar divisão aleatória em séries temporais

## Versões das bibliotecas principais

| Biblioteca        | Versão      |
|-------------------|-------------|
| tensorflow        | 2.21.0      |
| torch (PyTorch)   | 2.11.0+cpu  |
| pytorch_forecasting | 1.7.0     |
| statsmodels       | 0.14.6      |
| pandas            | 3.0.2       |
| numpy             | 2.4.4       |
| scikit-learn      | 1.8.0       |
| matplotlib        | 3.10.9      |
| seaborn           | 0.13.2      |

> TensorFlow >= 2.11 não suporta GPU nativa no Windows. Treino é em CPU. Alternativas: WSL2 ou tensorflow-directml.

## Estrutura de arquivos

```
src/
  pipeline/
    loader.py        — ingestão e validação do CSV
    preprocessor.py  — features temporais, lags, médias móveis
    splitter.py      — divisão treino/teste, normalização, janelas numpy
  models/
    mlp.py           — MLP (TensorFlow/Keras)
    lstm.py          — LSTM (TensorFlow/Keras)
    gru.py           — GRU (TensorFlow/Keras)
    arima.py         — ARIMA baseline (statsmodels)
    tft.py           — (pendente) Temporal Fusion Transformer
  evaluation/
    metricas.py      — calcular_metricas, comparar_modelos
data/
  raw/train.csv
main.py              — entry point (vazio)
```

## Pipeline de dados

```
carregar_serie_temporal(csv)
  └─ preparar_dataset(df, coluna_alvo='sales', colunas_grupo=['store','item'])
       └─ criar_features_temporais   → 9 features de calendário
       └─ criar_lags                 → lag_1, lag_7, lag_30
       └─ criar_medias_moveis        → media_movel_7, media_movel_30
       └─ dropna → 898k linhas, 14 features
  └─ dividir_treino_teste(df, data_corte='2017-01-06')
  └─ normalizar(treino, teste, features, 'sales')  → MinMaxScaler fit só no treino
  └─ criar_janelas(df, 'sales', features, tamanho_janela=30)
       └─ X: (amostras, 30, 14)  |  y: (amostras,)
```

## Modelos neurais — interface comum

Todos em `src/models/` seguem a mesma assinatura:

```python
construir_<modelo>(input_shape=(30,14), ...)  → modelo compilado (Adam + MSE)
treinar_<modelo>(modelo, X_tr, y_tr, epochs=50, batch_size=512, validation_split=0.1)
prever_<modelo>(modelo, X_te, scaler_alvo)    → array 1D na escala original
```

EarlyStopping: `patience=10`, `monitor='val_loss'`, `restore_best_weights=True`.

## Modelos — parâmetros padrão e tamanho

| Modelo | Arquitetura padrão      | Params  |
|--------|-------------------------|---------|
| MLP    | Flatten → 128 → 64 → 1  | 62.209  |
| LSTM   | LSTM(64) → LSTM(32) → 1 | 32.673  |
| GRU    | GRU(64) → GRU(32) → 1  | 24.801  |

## ARIMA baseline

Opera na série **agregada** (soma diária de todas as lojas/itens) — 1.826 dias.
Ordem padrão: `(1, 1, 1)`. Resultado atual no conjunto de teste:

| MAE      | RMSE      | MAPE   |
|----------|-----------|--------|
| 9290.28  | 11007.98  | 28.6%  |

## Avaliação

```python
from src.evaluation.metricas import calcular_metricas, comparar_modelos

resultado = calcular_metricas(y_real, y_pred, nome_modelo='LSTM')
tabela    = comparar_modelos([res_arima, res_mlp, res_lstm, res_gru])
# → DataFrame ordenado por MAE com colunas: modelo, MAE, RMSE, MAPE
```

MAPE ignora amostras onde `y_real == 0`.

## Próximos passos

- [x] `src/models/tft.py` — Temporal Fusion Transformer (pytorch-forecasting)
- [ ] `main.py` — orquestrar pipeline completo e gerar tabela comparativa
- [ ] Visualizações (curvas de treino, predito vs real, comparativo de modelos)
