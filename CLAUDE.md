# Projeto RNA — Demand Forecasting

Projeto acadêmico (UPE — Redes Neurais) de previsão de demanda comparando modelos neurais e baseline estatístico.

## Dataset

- **Arquivo:** `data/raw/train.csv`
- **Colunas:** `date`, `store`, `item`, `sales`
- **Escala:** 913.000 linhas brutas, 500 séries (10 lojas × 50 itens), período 2013-01-01 a 2017-12-31
- **Após pré-processamento:** 898.000 linhas, 18 colunas (14 features + date/store/item/sales)
- **Divisão temporal:** treino < 2017-01-06 (718k linhas), teste >= 2017-01-06 (180k linhas)

## Ambiente

- Python 3.12.1, `.venv` local
- Executar com `.venv/Scripts/python` (Windows)
- TensorFlow >= 2.11 não suporta GPU nativa no Windows — treino em CPU
- **Nunca** usar divisão aleatória em séries temporais

## Versões das bibliotecas principais

| Biblioteca          | Versão      |
|---------------------|-------------|
| tensorflow          | 2.21.0      |
| torch (PyTorch)     | 2.11.0+cpu  |
| pytorch_forecasting | 1.7.0       |
| lightning.pytorch   | 2.6.1       |
| statsmodels         | 0.14.6      |
| pandas              | 3.0.2       |
| numpy               | 2.4.4       |
| scikit-learn        | 1.8.0       |
| matplotlib          | 3.10.9      |
| seaborn             | 0.13.2      |

## Estrutura de arquivos

```
src/
  pipeline/
    loader.py        — carregar_serie_temporal, validar_serie_temporal
    preprocessor.py  — criar_features_temporais, criar_lags, criar_medias_moveis, preparar_dataset
    splitter.py      — dividir_treino_teste, normalizar, criar_janelas
  models/
    mlp.py           — construir_mlp, treinar_mlp, prever_mlp  (TensorFlow/Keras)
    lstm.py          — construir_lstm, treinar_lstm, prever_lstm (TensorFlow/Keras)
    gru.py           — construir_gru, treinar_gru, prever_gru   (TensorFlow/Keras)
    arima.py         — preparar_serie_arima, construir_e_treinar_arima, prever_arima (statsmodels)
                       + reexporta calcular_metricas de evaluation/metricas.py
    tft.py           — preparar_dataset_tft, construir_tft, treinar_tft, prever_tft (pytorch-forecasting)
  evaluation/
    metricas.py      — calcular_metricas, comparar_modelos
    comparativo.py   — plotar_comparativo_metricas, plotar_predicoes, gerar_relatorio
data/
  raw/train.csv
outputs/
  figures/           — comparativo_metricas.png, predicoes_comparativo.png
  comparativo_resultados.csv
article/
  artigo.md          — esqueleto completo do artigo acadêmico
main.py              — pipeline completo (pronto para execução)
```

## Pipeline de dados

```
carregar_serie_temporal(csv)
  └─ validar_serie_temporal(df, colunas_grupo)   → dict {valido, coluna_data, ...}
  └─ preparar_dataset(df, coluna_alvo, colunas_grupo)
       └─ criar_features_temporais   → 9 features de calendário
       └─ criar_lags                 → lag_1, lag_7, lag_30      (por grupo)
       └─ criar_medias_moveis        → media_movel_7, media_movel_30 (por grupo)
       └─ dropna → 898k linhas, 14 features
  └─ dividir_treino_teste(df, data_corte=None)   → corte automático 80/20
  └─ normalizar(treino, teste, features, alvo)   → MinMaxScaler fit só no treino
  └─ criar_janelas(df, alvo, features, janela=30)
       └─ X: (amostras, 30, 14)  |  y: (amostras,)  [numpy float32]
```

## Modelos neurais — interface comum (MLP / LSTM / GRU)

```python
construir_<modelo>(input_shape=(30, 14), ...)  → modelo compilado (Adam + MSE)
treinar_<modelo>(modelo, X_tr, y_tr, epochs=50, batch_size=512, validation_split=0.1)
prever_<modelo>(modelo, X_te, scaler_alvo)     → array 1D na escala original
```

EarlyStopping: `patience=10`, `monitor='val_loss'`, `restore_best_weights=True`.

| Modelo | Arquitetura padrão      | Params  |
|--------|-------------------------|---------|
| MLP    | Flatten → 128 → 64 → 1  | 62.209  |
| LSTM   | LSTM(64) → LSTM(32) → 1 | 32.673  |
| GRU    | GRU(64) → GRU(32) → 1  | 24.801  |

## TFT — interface própria

```python
# Recebe df_proc (DataFrame processado, antes da normalização manual)
# Em main.py: subconjunto com 10 lojas × 10 itens (100 séries) para viabilizar CPU
ds_treino, ds_val = preparar_dataset_tft(df_proc, coluna_alvo, colunas_grupo,
                                          tamanho_encoder=30, tamanho_predicao=7)
modelo = construir_tft(ds_treino, hidden_size=32, dropout=0.1, learning_rate=0.001)
trainer, melhor = treinar_tft(modelo, ds_treino, ds_val, max_epochs=20, batch_size=64)
preds = prever_tft(melhor, ds_val)  → array 1D, escala original (GroupNormalizer interno)
```

- Parâmetros: 118.565 | EarlyStopping patience=5 | QuantileLoss | lightning.pytorch
- `preparar_dataset_tft` cria `time_idx` (dias desde data mínima), `serie_id` ("store_item")
- Booleanos e coluna alvo convertidos para float antes de criar o TimeSeriesDataSet
- Validação = últimos 7 dias de cada série via `from_dataset(..., predict=True)`
- MAPE alto (85.47%) apesar do melhor MAE/RMSE — possível distorção do GroupNormalizer em séries com valores próximos de zero

## ARIMA baseline

Opera na série **agregada** (soma diária de todas as lojas/itens) — 1.826 dias.  
Ordem padrão: `(1, 1, 1)`. Resultado no conjunto de teste:

| MAE      | RMSE      | MAPE   |
|----------|-----------|--------|
| 9234.06  | 10961.15  | 28.43% |

> Escala ~20.000 unidades/dia (agregado). Modelos neurais operam em ~50 unidades/dia (por série). MAPE é a métrica comparável entre os dois grupos.

## Avaliação

```python
from src.evaluation.metricas import calcular_metricas, comparar_modelos
from src.evaluation.comparativo import plotar_comparativo_metricas, plotar_predicoes, gerar_relatorio

res = calcular_metricas(y_real, y_pred, nome_modelo='LSTM')
# → {'modelo': 'LSTM', 'MAE': ..., 'RMSE': ..., 'MAPE': ...}

df = comparar_modelos([res_arima, res_mlp, res_lstm, res_gru, res_tft])
# → DataFrame ordenado por MAE

gerar_relatorio(df)                          # → CSV + print tabela
plotar_comparativo_metricas(df)              # → outputs/figures/comparativo_metricas.png
plotar_predicoes(y_real, {nome: preds}, ...) # → outputs/figures/predicoes_comparativo.png
```

MAPE ignora amostras onde `y_real == 0`.

## Executar o pipeline completo

```bash
.venv/Scripts/python main.py
```

`main.py` executa em 6 etapas com prints de progresso `[1/6]...[6/6]`:  
carregar → pré-processar → dividir/normalizar → ARIMA → MLP/LSTM/GRU → TFT → comparativo.

## Artigo acadêmico

`article/artigo.md` — esqueleto completo (~8 páginas formatadas):
- Seções preenchidas: Introdução, Referencial Teórico, Metodologia (dataset, pipeline, tabela de hiperparâmetros, protocolo de avaliação), Limitações, Generalização, Conclusão parcial, Referências
- Seções com `[PREENCHER]`: Resumo, Tabela de resultados, Análise por modelo, Síntese da conclusão

## Reprodutibilidade

`SEED = 42` fixado no topo de `main.py` para `random`, `numpy`, `tensorflow` e `torch`.

## Resultados finais (pipeline executado)

### Modelos neurais — escala por série (~50 unid/dia)

| Modelo | MAE   | RMSE  | MAPE   |
|--------|-------|-------|--------|
| TFT    | 22.99 | 28.93 | 85.47% |
| GRU    | 23.57 | 28.99 | 56.72% |
| LSTM   | 23.60 | 29.01 | 56.90% |
| MLP    | 23.64 | 29.46 | 54.59% |

### ARIMA — série agregada (~20.000 unid/dia)

| MAE     | RMSE     | MAPE   |
|---------|----------|--------|
| 9234.06 | 10961.15 | 28.43% |

## Pendências

- [x] Executar `main.py` com treino completo
- [ ] Preencher Resumo e seções `[PREENCHER]` do `article/artigo.md` com os resultados acima
- [ ] Revisar nota metodológica sobre janela deslizante global (cruza fronteiras de séries)
- [ ] Discutir no artigo o MAPE elevado do TFT vs MAE/RMSE baixo
