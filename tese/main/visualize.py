import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ── Diretório de resultados
RESULTS_DIR = "results"

# ── Mapeamento de nomes de modelos para labels legíveis
name_mapping = {
    "KNN_All":      "KNN-All",
    "LightGCN":     "LightGCN (NDCG)",
    "LightGCN_MRR": "LightGCN (MRR)",
    "HeteroGAT":    "HeteroGAT",
    "HeteroGATv2":  "HeteroGATv2",
}

# ── Estilo por modelo
model_styles = {
    "KNN_All":      {"color": "steelblue",   "marker": "o"},
    "LightGCN":     {"color": "darkorange",  "marker": "s"},
    "LightGCN_MRR": {"color": "gold",        "marker": "D"},
    "HeteroGAT":    {"color": "darkgreen",   "marker": "^"},
    "HeteroGATv2":  {"color": "crimson",     "marker": "P"},
}

# ──────────────────────────────────────────────
# 1. Gráficos de curvas (Precision, Recall, F1, MRR)
# ──────────────────────────────────────────────
metrics = [
    ("precision", "Precision", "upper right"),
    ("recall",    "Recall",    "lower right"),
    ("f1",        "F1 Score",  "upper right"),
    ("mrr",       "MRR",       "lower right"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for ax, (metric_key, ylabel, legend_loc) in zip(axes, metrics):
    pattern = f"{metric_key}_"
    files   = [f for f in os.listdir(RESULTS_DIR) if f.startswith(pattern) and f.endswith(".csv")]

    for fname in sorted(files):
        df = pd.read_csv(os.path.join(RESULTS_DIR, fname))
        model_col  = df.columns[0]
        value_cols = df.columns[1:]

        for _, row in df.iterrows():
            model_name = str(row[model_col]).strip()
            label      = name_mapping.get(model_name, model_name)
            style      = model_styles.get(model_name, {"color": None, "marker": "o"})
            y_values   = row[value_cols].astype(float).values

            ax.plot(
                range(1, len(value_cols) + 1),
                y_values,
                marker=style["marker"],
                label=label,
                color=style["color"],
                markersize=6,
                linewidth=1.5
            )

    ax.set_xlabel("Top@k", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc=legend_loc, frameon=True, framealpha=0.8,
              borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

plt.suptitle("EmoRecSys — Comparação de Modelos", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison_plot.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico de curvas guardado em results/comparison_plot.png")

# ──────────────────────────────────────────────
# 2. Gráfico de barras — métricas fixas
# ──────────────────────────────────────────────
fixed_files = glob.glob(os.path.join(RESULTS_DIR, "fixed_metrics_*.csv"))
if fixed_files:
    fixed_dfs = [pd.read_csv(f) for f in sorted(fixed_files)]
    fixed_df  = pd.concat(fixed_dfs, ignore_index=True)

    available = [m for m in ["Precision@1", "NDCG@10", "MRR@10", "HitRate@10"] if m in fixed_df.columns]
    n_models  = len(fixed_df)
    n_metrics = len(available)
    x         = np.arange(n_metrics)
    width     = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (_, row) in enumerate(fixed_df.iterrows()):
        model  = row['model']
        style  = model_styles.get(model, {"color": None})
        label  = name_mapping.get(model, model)
        values = [row[m] for m in available]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width, label=label, color=style["color"])

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel("Métrica", fontsize=12)
    ax.set_ylabel("Valor", fontsize=12)
    ax.set_title("EmoRecSys — Comparação de Métricas Fixas", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(available, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fixed_metrics_comparison.png"), dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico de barras guardado em results/fixed_metrics_comparison.png")

# ──────────────────────────────────────────────
# 3. Tabela comparativa de métricas fixas
# ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("TABELA COMPARATIVA — MÉTRICAS FIXAS")
    print("="*65)

    display_df = fixed_df.copy()
    display_df['model'] = display_df['model'].map(lambda x: name_mapping.get(x, x))

    metric_cols = [m for m in ["Precision@1", "NDCG@10", "MRR@10", "HitRate@10", "Recall@10"] if m in display_df.columns]
    display_df  = display_df[['model'] + metric_cols]

    if 'NDCG@10' in display_df.columns:
        display_df = display_df.sort_values("NDCG@10", ascending=False)

    display_df = display_df.rename(columns={"model": "Modelo"})
    print(display_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*65)

    display_df.to_csv(os.path.join(RESULTS_DIR, "tabela_comparativa.csv"), index=False)
    print("Tabela guardada em results/tabela_comparativa.csv")
else:
    print("Nenhum ficheiro fixed_metrics_*.csv encontrado.")