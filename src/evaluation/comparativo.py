import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Diretório de saída compartilhado
_DIR_FIGURES = "outputs/figures"
_DIR_OUTPUTS = "outputs"


def plotar_comparativo_metricas(df_comparativo: pd.DataFrame) -> None:
    """Gráfico de barras agrupadas com MAE, RMSE e MAPE por modelo."""
    os.makedirs(_DIR_FIGURES, exist_ok=True)

    metricas = ["MAE", "RMSE", "MAPE"]
    modelos = df_comparativo["modelo"].tolist()
    n_modelos = len(modelos)
    n_metricas = len(metricas)

    x = np.arange(n_modelos)
    largura = 0.25
    offsets = np.linspace(-(n_metricas - 1) / 2, (n_metricas - 1) / 2, n_metricas) * largura

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    cores = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, metrica, cor, offset in zip(axes, metricas, cores, offsets):
        valores = df_comparativo[metrica].tolist()
        barras = ax.bar(x, valores, width=largura * n_metricas * 0.8, color=cor, alpha=0.85)

        # Rótulo de valor em cima de cada barra
        for barra, val in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() * 1.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_title(metrica, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=15, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle("Comparativo de Métricas por Modelo", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    caminho = os.path.join(_DIR_FIGURES, "comparativo_metricas.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {caminho}")


def plotar_predicoes(
    y_real: np.ndarray,
    predicoes_por_modelo: dict[str, np.ndarray],
    titulo: str = "Predições vs Real",
    dias: int = 90,
) -> None:
    """Série real vs predições de cada modelo — limitado aos primeiros `dias` dias."""
    os.makedirs(_DIR_FIGURES, exist_ok=True)

    y_real = np.asarray(y_real)[:dias]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(y_real, label="Real", color="black", linewidth=1.8, zorder=5)

    cores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    estilos = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for (nome, preds), cor, estilo in zip(predicoes_por_modelo.items(), cores, estilos):
        preds = np.asarray(preds)[:dias]
        ax.plot(preds, label=nome, color=cor, linestyle=estilo, linewidth=1.2, alpha=0.85)

    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Dias (primeiros {dias})", fontsize=10)
    ax.set_ylabel("Vendas", fontsize=10)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    plt.tight_layout()

    caminho = os.path.join(_DIR_FIGURES, "predicoes_comparativo.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {caminho}")


def gerar_relatorio(
    df_comparativo: pd.DataFrame,
    caminho_saida: str = "outputs/comparativo_resultados.csv",
) -> None:
    """Salva o comparativo em CSV e imprime tabela formatada no terminal."""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    df_comparativo.to_csv(caminho_saida, index=False)
    print(f"CSV salvo em: {caminho_saida}")
    print()
    print(df_comparativo.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
