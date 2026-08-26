"""
generate_result_charts.py

Gera os gráficos de comparação de resultados (pool-20, full-catalog,
produto-escalar vs cosseno, curvas @k) a partir dos CSVs já guardados
em ../results/, e grava cada imagem separadamente nesta pasta (export/).

Correr a partir de dentro da pasta export:
    python generate_result_charts.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Ancorar os caminhos à localização deste ficheiro, não à pasta onde o
# terminal está posicionado — assim funciona sempre da mesma forma,
# independentemente de onde chamares "python generate_result_charts.py".
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
OUT_DIR     = SCRIPT_DIR  # guardar as imagens sempre na pasta onde este script está

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

MODELS = ["KNN", "LightGCN", "HeteroGAT", "HeteroGATv2"]
COLORS = {"KNN": "#A1C9F4", "LightGCN": "#FFB482", "HeteroGAT": "#8DE5A1", "HeteroGATv2": "#FF9F9B"}
FILE_KEY = {"KNN": "knn", "LightGCN": "lightgcn", "HeteroGAT": "heterogat", "HeteroGATv2": "heterogatv2"}
METRICS = ["Precision@1", "Recall@10", "HitRate@10", "NDCG@10", "MRR@10"]


def load_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  [aviso] ficheiro não encontrado, a saltar: {path}")
        return None
    return pd.read_csv(path)


def build_master_table(suffix=""):
    """suffix: '' para pool-20, '_full' para full-catalog, '_full_cosine' para cosseno."""
    rows = []
    for name, key in FILE_KEY.items():
        df = load_csv(f"fixed_metrics_{key}{suffix}.csv")
        if df is None:
            continue
        row = df.iloc[0].to_dict()
        row["Model"] = name
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("Model")[METRICS]


def chart_01_pool20_bar():
    print("\n[1/5] Pool-20 bar comparison...")
    pool20 = build_master_table("")
    if pool20 is None:
        print("  Sem dados suficientes, gráfico saltado.")
        return

    x = np.arange(len(METRICS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = []
    for i, m in enumerate(pool20.index):
        vals = pool20.loc[m, METRICS].values
        b = ax.bar(x + (i - (len(pool20.index) - 1) / 2) * width, vals, width,
                   label=m, color=COLORS.get(m, None))
        bars.append(b)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pool-20 Protocol: Fixed Metrics Comparison")
    fig.legend(bars, list(pool20.index), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(OUT_DIR, "01_pool20_bar_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {out}")


def chart_02_cross_protocol():
    print("\n[2/5] Cross-protocol validation (pool-20 vs full-catalog)...")
    pool20 = build_master_table("")
    full = build_master_table("_full")
    if pool20 is None or full is None:
        print("  Sem dados suficientes, gráfico saltado.")
        return

    common_models = [m for m in pool20.index if m in full.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    bar_handles = None
    for ax, metric in zip(axes, ["NDCG@10", "Precision@1"]):
        xx = np.arange(len(common_models))
        p20_vals = [pool20.loc[m, metric] for m in common_models]
        full_vals = [full.loc[m, metric] for m in common_models]
        w = 0.32
        b1 = ax.bar(xx - w / 2, p20_vals, w, label="Pool-20", color="#A1C9F4")
        b2 = ax.bar(xx + w / 2, full_vals, w, label="Full-Catalog", color="#FFB482")
        bar_handles = [b1, b2]
        ax.set_xticks(xx)
        ax.set_xticklabels(common_models, rotation=15)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.0)
        ax.set_title(metric)
        for j, (p, f) in enumerate(zip(p20_vals, full_vals)):
            if p > 0:
                drop = (1 - f / p) * 100
                ax.annotate(f"-{drop:.0f}%", xy=(j, max(p, f) + 0.03),
                            ha="center", fontsize=9, color="#555")

    fig.legend(bar_handles, ["Pool-20", "Full-Catalog"], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Cross-Protocol Validation: Pool-20 vs Full-Catalog\n(percentages show relative drop)", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(OUT_DIR, "02_cross_protocol_validation.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {out}")


def chart_03_dotproduct_vs_cosine():
    print("\n[3/5] Dot product vs Cosine similarity (HeteroGAT / HeteroGATv2)...")
    cos_models = ["HeteroGAT", "HeteroGATv2"]
    full = build_master_table("_full")
    cos_full = build_master_table("_full_cosine")
    if full is None or cos_full is None:
        print("  Sem dados suficientes, gráfico saltado.")
        return

    available = [m for m in cos_models if m in full.index and m in cos_full.index]
    if not available:
        print("  Nenhum modelo com dados de cosseno disponíveis, gráfico saltado.")
        return

    fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 5))
    bar_handles = None
    for ax, metric in zip(axes, METRICS):
        xx = np.arange(len(available))
        dot_vals = [full.loc[m, metric] for m in available]
        cos_vals = [cos_full.loc[m, metric] for m in available]
        w = 0.32
        b1 = ax.bar(xx - w / 2, dot_vals, w, label="Dot product", color="#A1C9F4")
        b2 = ax.bar(xx + w / 2, cos_vals, w, label="Cosine sim.", color="#FFB482")
        bar_handles = [b1, b2]
        ax.set_xticks(xx)
        ax.set_xticklabels(available, rotation=15)
        ax.set_title(metric)

    fig.legend(bar_handles, ["Dot product", "Cosine sim."], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Full-Catalog Scoring: Dot Product vs Cosine Similarity", fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(OUT_DIR, "03_dotproduct_vs_cosine.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {out}")


def chart_04_metric_curves():
    print("\n[4/5] Curvas @k combinadas (grelha 2x2) — Pool-20...")
    curve_metrics = {"precision": "Precision@k", "recall": "Recall@k", "f1": "F1@k", "mrr": "MRR@k"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    any_plotted = False
    for ax, (key, label) in zip(axes, curve_metrics.items()):
        for m in MODELS:
            df = load_csv(f"{key}_{FILE_KEY[m]}.csv")
            if df is None:
                continue
            vals = df.iloc[0][[f"top{i}" for i in range(1, 11)]].astype(float).values
            ax.plot(range(1, 11), vals, marker="o", markersize=4, label=m,
                     color=COLORS.get(m, None), linewidth=1.8)
            any_plotted = True
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(range(1, 11))

    if not any_plotted:
        print("  Sem dados de curvas disponíveis, gráfico saltado.")
        plt.close()
        return

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Ranking Quality Across k=1 to 10 (Pool-20 Protocol)", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(OUT_DIR, "04_metric_curves_grid.png")
    plt.savefig(out, dpi=300, pad_inches=0.15)
    plt.close()
    print(f"  Guardado: {out}")


def chart_05_individual_curves():
    print("\n[5/5] Curvas @k individuais (Precision, Recall, F1, MRR) — Pool-20...")
    curve_metrics = {
        "precision": ("Precision@k", "05_precision_curve.png"),
        "recall":    ("Recall@k",    "06_recall_curve.png"),
        "f1":        ("F1@k",        "07_f1_curve.png"),
        "mrr":       ("MRR@k",       "08_mrr_curve.png"),
    }

    for key, (label, filename) in curve_metrics.items():
        fig, ax = plt.subplots(figsize=(7, 5.5))
        any_plotted = False
        for m in MODELS:
            df = load_csv(f"{key}_{FILE_KEY[m]}.csv")
            if df is None:
                continue
            vals = df.iloc[0][[f"top{i}" for i in range(1, 11)]].astype(float).values
            ax.plot(range(1, 11), vals, marker="o", markersize=5, label=m,
                     color=COLORS.get(m, None), linewidth=2)
            any_plotted = True

        if not any_plotted:
            print(f"  [aviso] sem dados para {label}, gráfico saltado.")
            plt.close()
            continue

        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(f"{label} — Pool-20 Protocol")
        ax.set_xticks(range(1, 11))
        ax.legend(frameon=False)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, filename)
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"  Guardado: {out}")


if __name__ == "__main__":
    print(f"A ler resultados de: {os.path.abspath(RESULTS_DIR)}")
    print(f"A guardar imagens em: {os.path.abspath(OUT_DIR)}")

    chart_01_pool20_bar()
    chart_02_cross_protocol()
    chart_03_dotproduct_vs_cosine()
    chart_04_metric_curves()
    chart_05_individual_curves()

    print("\nConcluído.")