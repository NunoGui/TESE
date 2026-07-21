"""
compare_all_models.py

Junta os resultados fixed_metrics_*.csv de TODOS os modelos - os 4 originais
(main\\results) e as 2 variantes visuais (resnet\\output) - numa única tabela,
e gera dois gráficos comparativos: barras e linhas.

Não modifica nenhum ficheiro existente; só lê e gera ficheiros novos,
isolados em resnet\\output.

Corre a partir de main\\resnet:
    python compare_all_models.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

PASTA_SCRIPT  = os.path.dirname(os.path.abspath(__file__))   # main\resnet
PASTA_MAIN    = os.path.dirname(PASTA_SCRIPT)                # main
PASTA_RESULTS = os.path.join(PASTA_MAIN, "results")           # main\results (modelos originais)
PASTA_OUTPUT  = os.path.join(PASTA_SCRIPT, "output")          # main\resnet\output (modelos visuais + outputs)
os.makedirs(PASTA_OUTPUT, exist_ok=True)

# Modelo -> (pasta onde está o fixed_metrics, nome do ficheiro)
FICHEIROS = {
    "LightGCN":           (PASTA_RESULTS, "fixed_metrics_lightgcn.csv"),
    "KNN":                (PASTA_RESULTS, "fixed_metrics_knn.csv"),
    "HeteroGAT":          (PASTA_RESULTS, "fixed_metrics_heterogat.csv"),
    "HeteroGATv2":        (PASTA_RESULTS, "fixed_metrics_heterogatv2.csv"),
    "HeteroGAT-Visual":   (PASTA_OUTPUT,  "fixed_metrics_heterogat_visual.csv"),
    "HeteroGATv2-Visual": (PASTA_OUTPUT,  "fixed_metrics_heterogatv2_visual.csv"),
}

METRICAS_GRAFICO = ["Precision@1", "NDCG@10", "MRR@10"]

OUTPUT_CSV           = os.path.join(PASTA_OUTPUT, "comparacao_todos_modelos.csv")
OUTPUT_GRAFICO_BARRAS = os.path.join(PASTA_OUTPUT, "comparacao_todos_modelos_barras.png")
OUTPUT_GRAFICO_LINHAS = os.path.join(PASTA_OUTPUT, "comparacao_todos_modelos_linhas.png")

# ---------------------------------------------------------------------------
# LÓGICA
# ---------------------------------------------------------------------------

def carregar_resultados(ficheiros):
    linhas = []
    for modelo, (pasta, nome_ficheiro) in ficheiros.items():
        caminho = os.path.join(pasta, nome_ficheiro)
        if not os.path.exists(caminho):
            print(f"Aviso: '{caminho}' não encontrado - a saltar o modelo {modelo}.")
            continue
        df = pd.read_csv(caminho)
        linha = df.iloc[0].to_dict()
        linha["Modelo"] = modelo
        if "model" in linha:
            del linha["model"]
        linhas.append(linha)

    resultado = pd.DataFrame(linhas)
    colunas = ["Modelo"] + [c for c in resultado.columns if c != "Modelo"]
    resultado = resultado[colunas]
    resultado = resultado.sort_values("NDCG@10", ascending=False).reset_index(drop=True)
    return resultado


def gerar_grafico_barras(df, metricas, caminho_output):
    metricas_presentes = [m for m in metricas if m in df.columns]
    modelos = df["Modelo"].tolist()
    n_modelos = len(modelos)
    n_metricas = len(metricas_presentes)

    x = range(n_modelos)
    largura_barra = 0.8 / n_metricas

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, metrica in enumerate(metricas_presentes):
        valores = df[metrica].tolist()
        posicoes = [xi + i * largura_barra for xi in x]
        barras = ax.bar(posicoes, valores, width=largura_barra, label=metrica)
        for barra, valor in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.01,
                f"{valor:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    centro_grupo = [xi + (n_metricas - 1) * largura_barra / 2 for xi in x]
    ax.set_xticks(centro_grupo)
    ax.set_xticklabels(modelos, rotation=15, ha="right")
    ax.set_ylabel("Valor da métrica")
    ax.set_title("Comparação de todos os modelos - Barras (pool-20, kfold5)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(caminho_output, dpi=150)
    plt.close(fig)
    print(f"Guardado: {caminho_output}")


def gerar_grafico_linhas(df, metricas, caminho_output):
    """
    Gráfico de linhas: eixo X = métricas, eixo Y = valor, uma linha por modelo.
    Permite ver o perfil de cada modelo ao longo das 3 métricas.
    """
    metricas_presentes = [m for m in metricas if m in df.columns]
    x = range(len(metricas_presentes))

    fig, ax = plt.subplots(figsize=(9, 6))

    marcadores = ["o", "s", "^", "D", "v", "P"]

    for i, (_, row) in enumerate(df.iterrows()):
        valores = [row[m] for m in metricas_presentes]
        ax.plot(
            x, valores,
            marker=marcadores[i % len(marcadores)],
            label=row["Modelo"],
            linewidth=2,
            markersize=7,
        )
        for xi, valor in zip(x, valores):
            ax.annotate(
                f"{valor:.3f}",
                (xi, valor),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(metricas_presentes)
    ax.set_ylabel("Valor da métrica")
    ax.set_xlabel("Métrica")
    ax.set_title("Comparação de todos os modelos - Linhas (pool-20, kfold5)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(caminho_output, dpi=150)
    plt.close(fig)
    print(f"Guardado: {caminho_output}")


if __name__ == "__main__":
    df_final = carregar_resultados(FICHEIROS)

    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Guardado: {OUTPUT_CSV}\n")
    print(df_final.to_string(index=False))

    gerar_grafico_barras(df_final, METRICAS_GRAFICO, OUTPUT_GRAFICO_BARRAS)
    gerar_grafico_linhas(df_final, METRICAS_GRAFICO, OUTPUT_GRAFICO_LINHAS)