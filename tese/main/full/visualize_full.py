import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

RESULTS_DIR = "../results"

name_mapping = {
    "KNN_All_FullCatalog":        "KNN-All",
    "LightGCN_FullCatalog":       "LightGCN (NDCG)",
    "HeteroGATv2_FullCatalog":    "HeteroGATv2",
}

model_styles = {
    "KNN_All_FullCatalog":        {"color": "steelblue",  "marker": "o"},
    "LightGCN_FullCatalog":       {"color": "darkorange", "marker": "s"},
    "HeteroGATv2_FullCatalog":    {"color": "crimson",    "marker": "P"},
}

# ──────────────────────────────────────────────
# 1. Tabela comparativa
# ──────────────────────────────────────────────
fixed_files = glob.glob(os.path.join(RESULTS_DIR, "fixed_metrics_*_full.csv"))

if not fixed_files:
    print("Nenhum ficheiro fixed_metrics_*_full.csv encontrado em results/")
else:
    fixed_dfs = [pd.read_csv(f) for f in sorted(fixed_files)]
    fixed_df  = pd.concat(fixed_dfs, ignore_index=True)

    print("\n" + "="*65)
    print("TABELA COMPARATIVA — CATÁLOGO COMPLETO (kfold5)")
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

    display_df.to_csv(os.path.join(RESULTS_DIR, "tabela_full_catalog.csv"), index=False)
    print("Tabela guardada em results/tabela_full_catalog.csv")

    # ──────────────────────────────────────────────
    # 2. Gráfico de barras
    # ──────────────────────────────────────────────
    available = [m for m in ["Precision@1", "NDCG@10", "MRR@10", "HitRate@10"] if m in fixed_df.columns]
    n_models  = len(fixed_df)
    x         = np.arange(len(available))
    width     = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (_, row) in enumerate(fixed_df.iterrows()):
        model  = row['model']
        style  = model_styles.get(model, {"color": None})
        label  = name_mapping.get(model, model)
        values = [row[m] for m in available]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width, label=label, color=style["color"])

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xlabel("Métrica", fontsize=12)
    ax.set_ylabel("Valor", fontsize=12)
    ax.set_title("EmoRecSys — Avaliação Catálogo Completo (~1285 candidatos)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(available, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "full_catalog_comparison.png"), dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico guardado em results/full_catalog_comparison.png")

    # ──────────────────────────────────────────────
    # 3. Comparação pool 20 vs catálogo completo
    # ──────────────────────────────────────────────
    # Carregar métricas do pool de 20
    pool_files = [f for f in glob.glob(os.path.join(RESULTS_DIR, "fixed_metrics_*.csv"))
                  if "_full" not in f]

    if pool_files:
        pool_dfs = [pd.read_csv(f) for f in sorted(pool_files)]
        pool_df  = pd.concat(pool_dfs, ignore_index=True)

        # Mapear modelos equivalentes
        model_map = {
            "KNN_All":     "KNN_All_FullCatalog",
            "LightGCN":    "LightGCN_FullCatalog",
            "HeteroGATv2": "HeteroGATv2_FullCatalog",
        }

        print("\n" + "="*75)
        print("COMPARAÇÃO: Pool de 20 vs Catálogo Completo")
        print("="*75)
        print(f"{'Modelo':<20} {'Métrica':<15} {'Pool 20':>10} {'Catálogo':>12} {'Diferença':>12}")
        print("-"*75)

        for pool_model, full_model in model_map.items():
            pool_row = pool_df[pool_df['model'] == pool_model]
            full_row = fixed_df[fixed_df['model'] == full_model]
            if pool_row.empty or full_row.empty:
                continue
            label = name_mapping.get(full_model, full_model)
            for metric in ["Precision@1", "NDCG@10", "MRR@10"]:
                if metric in pool_row.columns and metric in full_row.columns:
                    v_pool = pool_row[metric].values[0]
                    v_full = full_row[metric].values[0]
                    diff   = v_full - v_pool
                    print(f"{label:<20} {metric:<15} {v_pool:>10.4f} {v_full:>12.4f} {diff:>+12.4f}")
            print()

print("\nDone!")