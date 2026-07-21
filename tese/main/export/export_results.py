"""
export_results.py

Junta os ficheiros "fixed_metrics_*.csv" (um por modelo, cada um com uma
única linha com todas as métricas) numa única tabela comparativa, pronta a
enviar à orientadora. Gera dois CSVs: um para o protocolo pool-20 e outro
para o protocolo full-catalog (se os ficheiros existirem).

Corre a partir de qualquer sítio - os caminhos resolvem-se sempre
relativamente à pasta onde este script está guardado (main\\export).
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # não precisa de janela gráfica, só gravar ficheiros
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

# Pasta onde estão os ficheiros fixed_metrics_*.csv, relativa a este script.
# O script está em main\export, os resultados estão em main\results,
# por isso subimos um nível e entramos em "results".
PASTA_RESULTADOS = os.path.join("..", "results")

# Ficheiros do protocolo principal (pool-20, kfold5)
FICHEIROS_POOL20 = {
    "KNN":         "fixed_metrics_knn.csv",
    "LightGCN":    "fixed_metrics_lightgcn.csv",
    "HeteroGAT":   "fixed_metrics_heterogat.csv",
    "HeteroGATv2": "fixed_metrics_heterogatv2.csv",
}

# Ficheiros do protocolo full-catalog (ignora os que não existirem)
FICHEIROS_FULL = {
    "KNN":         "fixed_metrics_knn_full.csv",
    "LightGCN":    "fixed_metrics_lightgcn_full.csv",
    "HeteroGAT":   "fixed_metrics_heterogat_full.csv",
    "HeteroGATv2": "fixed_metrics_heterogatv2_full.csv",
}

OUTPUT_POOL20 = "resultados_pool20.csv"
OUTPUT_FULL = "resultados_full_catalog.csv"

GRAFICO_POOL20 = "grafico_pool20.png"
GRAFICO_FULL = "grafico_full_catalog.png"

# Métricas a incluir nos gráficos de barras (por esta ordem)
METRICAS_GRAFICO = ["Precision@1", "NDCG@10", "MRR@10"]

# ---------------------------------------------------------------------------
# LÓGICA
# ---------------------------------------------------------------------------

def juntar_resultados(pasta, ficheiros, nome_metrica_ordenacao="NDCG@10", obrigatorio=True):
    """
    Lê cada ficheiro fixed_metrics_*.csv (uma linha por modelo) e junta tudo
    numa única tabela, ordenada pela métrica principal (NDCG@10).

    Se obrigatorio=False, ficheiros em falta são simplesmente ignorados
    (útil para o full-catalog, que pode não existir para todos os modelos).
    """
    linhas = []

    for modelo, nome_ficheiro in ficheiros.items():
        caminho = os.path.join(pasta, nome_ficheiro)

        if not os.path.exists(caminho):
            if obrigatorio:
                raise FileNotFoundError(f"[{modelo}] Ficheiro não encontrado: {caminho}")
            else:
                print(f"Aviso: '{caminho}' não encontrado, a saltar o modelo {modelo}.")
                continue

        df = pd.read_csv(caminho)

        if len(df) != 1:
            print(f"Aviso: '{caminho}' tem {len(df)} linhas (esperava 1). A usar a primeira.")

        linha = df.iloc[0].to_dict()
        linha["Modelo"] = modelo  # garantir nome consistente, independente da coluna "model" original
        linhas.append(linha)

    if not linhas:
        return None

    resultado = pd.DataFrame(linhas)

    # remover a coluna original "model" se existir, para não duplicar com "Modelo"
    if "model" in resultado.columns:
        resultado = resultado.drop(columns=["model"])

    # reordenar para "Modelo" ficar primeiro
    colunas = ["Modelo"] + [c for c in resultado.columns if c != "Modelo"]
    resultado = resultado[colunas]

    if nome_metrica_ordenacao in resultado.columns:
        resultado = resultado.sort_values(nome_metrica_ordenacao, ascending=False).reset_index(drop=True)

    return resultado


def gerar_grafico(df, metricas, titulo, caminho_output):
    """
    Gera um gráfico de barras agrupadas comparando os modelos nas métricas
    indicadas, e guarda como PNG.
    """
    metricas_presentes = [m for m in metricas if m in df.columns]
    if not metricas_presentes:
        print(f"Aviso: nenhuma das métricas {metricas} encontrada, gráfico não gerado.")
        return

    modelos = df["Modelo"].tolist()
    n_modelos = len(modelos)
    n_metricas = len(metricas_presentes)

    x = range(n_modelos)
    largura_barra = 0.8 / n_metricas

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, metrica in enumerate(metricas_presentes):
        valores = df[metrica].tolist()
        posicoes = [xi + i * largura_barra for xi in x]
        barras = ax.bar(posicoes, valores, width=largura_barra, label=metrica)
        # anotar o valor em cima de cada barra
        for barra, valor in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.01,
                f"{valor:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    centro_grupo = [xi + (n_metricas - 1) * largura_barra / 2 for xi in x]
    ax.set_xticks(centro_grupo)
    ax.set_xticklabels(modelos)
    ax.set_ylabel("Valor da métrica")
    ax.set_title(titulo)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(caminho_output, dpi=150)
    plt.close(fig)
    print(f"Guardado: {caminho_output}")


if __name__ == "__main__":
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    pasta_resultados_abs = os.path.join(pasta_script, PASTA_RESULTADOS)

    # --- Pool-20 (protocolo principal) ---
    df_pool20 = juntar_resultados(pasta_resultados_abs, FICHEIROS_POOL20, obrigatorio=True)
    output_pool20_abs = os.path.join(pasta_script, OUTPUT_POOL20)
    df_pool20.to_csv(output_pool20_abs, index=False, encoding="utf-8-sig")
    print(f"\nGuardado: {output_pool20_abs}")
    print(df_pool20.to_string(index=False))

    grafico_pool20_abs = os.path.join(pasta_script, GRAFICO_POOL20)
    gerar_grafico(df_pool20, METRICAS_GRAFICO, "Comparação de Modelos - Pool-20 (kfold5)", grafico_pool20_abs)

    # --- Full-catalog (se existir) ---
    df_full = juntar_resultados(pasta_resultados_abs, FICHEIROS_FULL, obrigatorio=False)
    if df_full is not None and len(df_full) > 0:
        output_full_abs = os.path.join(pasta_script, OUTPUT_FULL)
        df_full.to_csv(output_full_abs, index=False, encoding="utf-8-sig")
        print(f"\nGuardado: {output_full_abs}")
        print(df_full.to_string(index=False))

        grafico_full_abs = os.path.join(pasta_script, GRAFICO_FULL)
        gerar_grafico(df_full, METRICAS_GRAFICO, "Comparação de Modelos - Full-Catalog", grafico_full_abs)
    else:
        print("\nNenhum ficheiro full-catalog encontrado - a saltar esse CSV e gráfico.")