import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
 
# ── Ficheiros de resultados
RESULTS_DIR = "results"
 
metrics = [
    ("precision", "Precision", "upper right"),
    ("recall",    "Recall",    "lower right"),
    ("f1",        "F1 Score",  "upper right"),
    ("mrr",       "MRR",       "lower right"),
]
 
# Mapeamento de nomes de modelos para labels legíveis
name_mapping = {
    "KNN_All":          "KNN-All",
    "LightGCN":         "LightGCN",
    "HeteroLightGCN":   "HeteroLightGCN",
}
 
# Estilo por modelo
model_styles = {
    "KNN_All":        {"color": "steelblue",  "marker": "o"},
    "LightGCN":       {"color": "darkorange", "marker": "s"},
    "HeteroLightGCN": {"color": "darkgreen",  "marker": "^"},
}
 
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
 
for ax, (metric_key, ylabel, legend_loc) in zip(axes, metrics):
    # Carregar todos os CSVs desta métrica
    pattern = f"{metric_key}_"
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith(pattern) and f.endswith(".csv")]
 
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
                markersize=7
            )
 
    ax.set_xlabel("Top@k", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(1, len(value_cols) + 1))
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True)
    ax.legend(
        fontsize=9, loc=legend_loc, frameon=True, framealpha=0.8,
        borderpad=0.3, labelspacing=0.3, handletextpad=0.4
    )
 
plt.suptitle("EmoRecSys — Comparação de Modelos", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("results/comparison_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado em results/comparison_plot.png")
 
# ──────────────────────────────────────────────
# Gráfico de barras — métricas fixas
# ──────────────────────────────────────────────
import glob
 
fixed_files = glob.glob("results/fixed_metrics_*.csv")
if fixed_files:
    fixed_dfs = [pd.read_csv(f) for f in sorted(fixed_files)]
    fixed_df  = pd.concat(fixed_dfs, ignore_index=True)
 
    metrics_fixed = ["Precision@1", "MRR@10", "NDCG@10", "HitRate@10"]
    # MRR@10 não está no CSV directamente — usar top10 do mrr como proxy
    # Só plotar as que existem
    available = [m for m in metrics_fixed if m in fixed_df.columns]
 
    x     = np.arange(len(available))
    width = 0.35
 
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(fixed_df.iterrows()):
        model  = row['model']
        style  = model_styles.get(model, {"color": None})
        values = [row[m] for m in available]
        ax.bar(x + i * width, values, width, label=name_mapping.get(model, model),
               color=style["color"])
 
    ax.set_xlabel("Métrica", fontsize=12)
    ax.set_ylabel("Valor", fontsize=12)
    ax.set_title("Comparação de Métricas Fixas", fontsize=13)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(available, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.5)
 
    plt.tight_layout()
    plt.savefig("results/fixed_metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico de barras guardado em results/fixed_metrics_comparison.png")