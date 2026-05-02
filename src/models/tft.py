import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


def preparar_dataset_tft(
    df: pd.DataFrame,
    coluna_alvo: str = "sales",
    colunas_grupo: list[str] = ["store", "item"],
    tamanho_encoder: int = 30,
    tamanho_predicao: int = 7,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Prepara TimeSeriesDataSet de treino e validação para o TFT."""
    df = df.copy()

    # time_idx: inteiro sequencial global (dias desde a primeira data do DataFrame)
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days

    # Identificador de série como string — exigência do TFT para group_ids
    df["serie_id"] = df[colunas_grupo[0]].astype(str) + "_" + df[colunas_grupo[1]].astype(str)
    for col in colunas_grupo:
        df[col] = df[col].astype(str)

    # GroupNormalizer com softplus exige float; converte alvo e booleanos
    df[coluna_alvo] = df[coluna_alvo].astype(float)
    for col in ["e_fim_de_semana", "e_inicio_de_mes", "e_fim_de_mes"]:
        df[col] = df[col].astype(float)

    # Calendário é conhecido no futuro; lags/médias dependem de valores passados
    known_reals = [
        "time_idx", "dia_mes", "dia_semana", "semana_ano",
        "mes", "trimestre", "ano",
        "e_fim_de_semana", "e_inicio_de_mes", "e_fim_de_mes",
    ]
    unknown_reals = [
        coluna_alvo,
        "lag_1", "lag_7", "lag_30",
        "media_movel_7", "media_movel_30",
    ]

    corte_treino = df["time_idx"].max() - tamanho_predicao

    dataset_treino = TimeSeriesDataSet(
        df[df["time_idx"] <= corte_treino],
        time_idx="time_idx",
        target=coluna_alvo,
        group_ids=["serie_id"],
        max_encoder_length=tamanho_encoder,
        max_prediction_length=tamanho_predicao,
        static_categoricals=["serie_id"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        # GroupNormalizer normaliza por série — TFT não precisa de scaler externo
        target_normalizer=GroupNormalizer(groups=["serie_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Validação: últimos tamanho_predicao steps de cada série
    dataset_val = TimeSeriesDataSet.from_dataset(
        dataset_treino, df, predict=True, stop_randomization=True
    )

    print(f"Dataset treino: {len(dataset_treino):,} amostras")
    print(f"Dataset validacao: {len(dataset_val):,} amostras")

    return dataset_treino, dataset_val


def construir_tft(
    dataset_treino: TimeSeriesDataSet,
    hidden_size: int = 32,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
) -> TemporalFusionTransformer:
    """Instancia o TFT a partir do dataset de treino."""
    modelo = TemporalFusionTransformer.from_dataset(
        dataset_treino,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=4,       # hidden_size deve ser divisível por 4
        dropout=dropout,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    print(f"Parametros TFT: {sum(p.numel() for p in modelo.parameters()):,}")
    return modelo


def treinar_tft(
    modelo_tft: TemporalFusionTransformer,
    dataset_treino: TimeSeriesDataSet,
    dataset_val: TimeSeriesDataSet,
    max_epochs: int = 20,
    batch_size: int = 64,
) -> tuple:
    """Treina o TFT com EarlyStopping. Retorna (trainer, melhor_modelo)."""
    train_loader = dataset_treino.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_loader = dataset_val.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=False,
    )

    trainer.fit(modelo_tft, train_loader, val_loader)

    # Carrega pesos do melhor epoch (menor val_loss)
    if checkpoint.best_model_path:
        melhor_modelo = TemporalFusionTransformer.load_from_checkpoint(
            checkpoint.best_model_path
        )
    else:
        melhor_modelo = modelo_tft

    return trainer, melhor_modelo


def prever_tft(
    modelo_tft: TemporalFusionTransformer,
    dataset_val: TimeSeriesDataSet,
) -> np.ndarray:
    """
    Gera predições na escala original.
    O GroupNormalizer reverte a normalização internamente — sem scaler externo.
    """
    predicoes = modelo_tft.predict(
        dataset_val,
        return_y=False,
        batch_size=128,
        num_workers=0,
        trainer_kwargs={"accelerator": "cpu"},
    )
    # predict() retorna tensor (amostras, tamanho_predicao) — flatten para 1D
    if isinstance(predicoes, torch.Tensor):
        return predicoes.numpy().flatten()
    return np.array(predicoes).flatten()
