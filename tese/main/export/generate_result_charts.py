"""
generate_result_charts.py

Gera os gráficos de comparação de resultados (pool-20, full-catalog,
produto-escalar vs cosseno, curvas @k, e emoção vs visual vs fusão)
a partir dos CSVs já guardados em ../results/ e ../resnet/output/,
e grava cada imagem separadamente nesta pasta (export/).

IMPORTANTE: para o HeteroGAT e o HeteroGATv2, este script lê sempre os
ficheiros com sufixo "_grouped" (codificação demográfica final, 24 dims),
não os ficheiros originais de 47 dims. O KNN e o LightGCN não usam
demografia, por isso não têm variante "_grouped" e são lidos tal como
sempre foram.

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
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "..", "results")
VISUAL_DIR   = os.path.join(SCRIPT_DIR, "..", "resnet", "output")
OUT_DIR      = SCRIPT_DIR  # guardar as imagens sempre na pasta onde este script está

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

# Modelos que usam demografia e por isso têm a variante final "_grouped"
# (24 dims). KNN e LightGCN nunca usam demografia, por isso não entram aqui.
GROUPED_MODELS = {"HeteroGAT", "HeteroGATv2"}


def load_csv(name, base_dir=RESULTS_DIR):
    path = os.path.join(base_dir, name)
    if not os.path.exists(path):
        print(f"  [aviso] ficheiro não encontrado, a saltar: {path}")
        return None
    return pd.read_csv(path)


def fixed_metrics_filename(model, suffix):
    """suffix: '' para pool-20, '_full' para full-catalog, '_full_cosine' para cosseno.
    Para HeteroGAT/HeteroGATv2, usa sempre a versão '_grouped' (codificação final)."""
    key = FILE_KEY[model]
    base = f"fixed_metrics_{key}{suffix}"
    if model in GROUPED_MODELS:
        base += "_grouped"
    return base + ".csv"


def curve_filename(model, metric_key):
    """metric_key: 'precision', 'recall', 'f1' ou 'mrr'. Só usado para pool-20."""
    key = FILE_KEY[model]
    base = f"{metric_key}_{key}"
    if model in GROUPED_MODELS:
        base += "_grouped"
    return base + ".csv"


def build_master_table(suffix=""):
    """suffix: '' para pool-20, '_full' para full-catalog, '_full_cosine' para cosseno."""
    rows = []
    for name in FILE_KEY:
        df = load_csv(fixed_metrics_filename(name, suffix))
        if df is None:
            continue
        row = df.iloc[0].to_dict()
        row["Model"] = name
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("Model")[METRICS]


def chart_01_pool20_bar():
    print("\n[1/9] Pool-20 bar comparison...")
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
    ax.set_title("Pool-20 Protocol — Fixed Metrics Comparison")
    fig.legend(bars, list(pool20.index), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(OUT_DIR, "01_pool20_bar_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {out}")


def chart_02_cross_protocol():
    print("\n[2/9] Cross-protocol validation (pool-20 vs full-catalog)...")
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
    print("\n[3/9] Dot product vs Cosine similarity (HeteroGAT / HeteroGATv2)...")
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
    print("\n[4/9] Curvas @k combinadas (grelha 2x2) — Pool-20...")
    curve_metrics = {"precision": "Precision@k", "recall": "Recall@k", "f1": "F1@k", "mrr": "MRR@k"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    any_plotted = False
    for ax, (key, label) in zip(axes, curve_metrics.items()):
        for m in MODELS:
            df = load_csv(curve_filename(m, key))
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
    print("\n[5/9] Curvas @k individuais (Precision, Recall, F1, MRR) — Pool-20...")
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
            df = load_csv(curve_filename(m, key))
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


def chart_06_image_feature_comparison():
    """Compara as três configurações do nó Image (emoção / visual / fusão)
    para o HeteroGAT e o HeteroGATv2, sob o pool-20 protocol."""
    print("\n[6/9] Emotion vs Visual vs Fusion (image node features)...")

    configs = {
        "Emotion only\n(10d)":  ("results", lambda base: fixed_metrics_filename(base, "")),
        "Visual only\n(64d)":   ("visual",  lambda base: f"fixed_metrics_{FILE_KEY[base]}_visual.csv"),
        "Fusion\n(74d)":        ("visual",  lambda base: f"fixed_metrics_{FILE_KEY[base]}_visual_fusion.csv"),
    }

    models = ["HeteroGAT", "HeteroGATv2"]
    data = {m: {} for m in models}

    for m in models:
        for label, (source, fname_fn) in configs.items():
            base_dir = RESULTS_DIR if source == "results" else VISUAL_DIR
            df = load_csv(fname_fn(m), base_dir=base_dir)
            if df is not None:
                data[m][label] = df.iloc[0]["NDCG@10"]

    if not any(data[m] for m in models):
        print("  Sem dados suficientes, gráfico saltado.")
        return

    config_labels = list(configs.keys())
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(config_labels))
    width = 0.32
    bars = []
    for i, m in enumerate(models):
        vals = [data[m].get(lbl, np.nan) for lbl in config_labels]
        b = ax.bar(x + (i - 0.5) * width, vals, width, label=m, color=COLORS.get(m, None))
        bars.append(b)
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.4f}", xy=(x[j] + (i - 0.5) * width, v + 0.005),
                            ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0.80, 0.90)
    ax.set_title("Image Node Feature Configuration — Pool-20 NDCG@10")
    fig.legend(bars, models, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(OUT_DIR, "09_emotion_visual_fusion_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {out}")


def chart_07_leaderboard():
    """Ranking de todas as configurações testadas, por NDCG@10 (pool-20)."""
    print("\n[7/9] Leaderboard — todas as configurações (pool-20)...")

    entries = []

    base = build_master_table("")
    if base is not None:
        for m in base.index:
            entries.append((m, base.loc[m, "NDCG@10"], "Baseline" if m in ("KNN", "LightGCN") else "Heterogeneous (emotion)"))

    visual_configs = [
        ("HeteroGAT-Visual",        "heterogat",   "_visual",        "Heterogeneous (visual)"),
        ("HeteroGATv2-Visual",      "heterogatv2", "_visual",        "Heterogeneous (visual)"),
        ("HeteroGAT-Fusion",        "heterogat",   "_visual_fusion", "Heterogeneous (fusion)"),
        ("HeteroGATv2-Fusion",      "heterogatv2", "_visual_fusion", "Heterogeneous (fusion)"),
    ]
    for label, key, suffix, group in visual_configs:
        df = load_csv(f"fixed_metrics_{key}{suffix}.csv", base_dir=VISUAL_DIR)
        if df is not None:
            entries.append((label, df.iloc[0]["NDCG@10"], group))

    if not entries:
        print("  Sem dados suficientes, gráfico saltado.")
        return

    entries.sort(key=lambda e: e[1], reverse=True)
    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    groups = [e[2] for e in entries]

    group_colors = {
        "Baseline": "#A1C9F4",
        "Heterogeneous (emotion)": "#8DE5A1",
        "Heterogeneous (visual)": "#FF9F9B",
        "Heterogeneous (fusion)": "#D0BBFF",
    }
    bar_colors = [group_colors.get(g, "#CCCCCC") for g in groups]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(labels))
    ax.barh(y, values, color=bar_colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("NDCG@10")
    ax.set_xlim(min(values) - 0.02, max(values) + 0.02)
    ax.set_title("Leaderboard — All Configurations (Pool-20, NDCG@10)")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.4f}", xy=(v + 0.001, i), va="center", fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in group_colors.values()]
    fig.legend(handles, group_colors.keys(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(OUT_DIR, "10_leaderboard.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {out}")


def chart_08_degree_norm_scatter():
    """Scatter grau de treino vs norma do embedding, para HeteroGAT e HeteroGATv2."""
    print("\n[8/9] Training degree vs embedding norm (diagnostic)...")

    files = {
        "HeteroGAT":   "diagnostic_heterogat_full_grouped.csv",
        "HeteroGATv2": "diagnostic_heterogatv2_full_grouped.csv",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    any_plotted = False
    for ax, (model, fname) in zip(axes, files.items()):
        df = load_csv(fname)
        if df is None:
            ax.set_title(f"{model} (sem dados)")
            continue
        ax.scatter(df["train_degree"], df["embedding_norm"], s=10, alpha=0.15,
                   color=COLORS.get(model, "#888888"))
        # Média por grau, para tornar a tendência visível por cima dos pontos
        means = df.groupby("train_degree")["embedding_norm"].mean()
        ax.plot(means.index, means.values, color="black", linewidth=2, label="Mean")
        ax.set_title(model)
        ax.set_xlabel("Training degree (times seen in training)")
        any_plotted = True

    if not any_plotted:
        print("  Sem dados suficientes, gráfico saltado.")
        plt.close()
        return

    axes[0].set_ylabel("Embedding norm (L2)")
    fig.suptitle("Embedding Norm Collapse for Under-Trained Images", fontsize=13)
    axes[0].legend(frameon=False, loc="lower right")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "11_degree_norm_scatter.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {out}")


def chart_09_distance_to_best():
    """Distância percentual ao melhor modelo (LightGCN), pool-20 vs full-catalog."""
    print("\n[9/9] Distance to best model (LightGCN)...")

    pool20 = build_master_table("")
    full = build_master_table("_full")
    if pool20 is None or full is None:
        print("  Sem dados suficientes, gráfico saltado.")
        return

    best_pool20 = pool20["NDCG@10"].max()
    best_full = full["NDCG@10"].max()

    models = [m for m in pool20.index if m != "LightGCN"]
    gap_pool20 = [(1 - pool20.loc[m, "NDCG@10"] / best_pool20) * 100 for m in models]
    gap_full = [(1 - full.loc[m, "NDCG@10"] / best_full) * 100 if m in full.index else np.nan for m in models]

    y = np.arange(len(models))
    height = 0.32
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y - height / 2, gap_pool20, height, label="Pool-20", color="#A1C9F4")
    ax.barh(y + height / 2, gap_full, height, label="Full-Catalog", color="#FFB482")
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("% gap to best model (LightGCN) — NDCG@10")
    ax.set_title("Competitiveness Relative to the Best Model")
    ax.legend(frameon=False, loc="lower right")
    for i, (p, f) in enumerate(zip(gap_pool20, gap_full)):
        ax.annotate(f"{p:.1f}%", xy=(p + 0.5, i - height / 2), va="center", fontsize=8)
        if not np.isnan(f):
            ax.annotate(f"{f:.1f}%", xy=(f + 0.5, i + height / 2), va="center", fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "12_distance_to_best.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {out}")


if __name__ == "__main__":
    print(f"A ler resultados de: {os.path.abspath(RESULTS_DIR)}")
    print(f"A ler resultados visuais de: {os.path.abspath(VISUAL_DIR)}")
    print(f"A guardar imagens em: {os.path.abspath(OUT_DIR)}")
    print(f"Modelos com codificação demográfica agrupada (24d, final): {sorted(GROUPED_MODELS)}")

    chart_01_pool20_bar()
    chart_02_cross_protocol()
    chart_03_dotproduct_vs_cosine()
    chart_04_metric_curves()
    chart_05_individual_curves()
    chart_06_image_feature_comparison()
    chart_07_leaderboard()
    chart_08_degree_norm_scatter()
    chart_09_distance_to_best()

    print("\nConcluído.")