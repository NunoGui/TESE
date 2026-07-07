import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

import split
import evaluation

# ── Configuração
PATH_RATINGS = "../data/ratings_full.csv"
K            = 20    # número de vizinhos
TOP_K        = 10    # tamanho da lista de recomendações
K_FOLDS      = 5
TOTAL_ITEMS  = 3084
MIN_TEST     = 20
BASE_SEED    = 42

np.random.seed(BASE_SEED)

FEATURE_COLS = ['rating', 'anger', 'fear', 'disgust', 'sadness',
                'happiness', 'surprise', 'neutral', 'valence', 'arousal', 'dominance']

# ──────────────────────────────────────────────
# Funções de similaridade (estilo Ana)
# ──────────────────────────────────────────────
def cosine_sim_vector(df):
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_pivot = df.pivot_table(
        index='user', columns='item', values=feature_cols, fill_value=0
    )
    df_pivot.columns = [f"{item}_{feat}" for feat, item in df_pivot.columns]
    df_pivot = df_pivot.fillna(0)
    sim_matrix = cosine_similarity(df_pivot)
    return pd.DataFrame(sim_matrix, index=df_pivot.index, columns=df_pivot.index)

def get_k_most_sim_users(sim_df, user, k):
    return sim_df.loc[user].drop(user).nlargest(k)

def get_scores_full(all_candidate_items, similar_users, train_df):
    """Score para todos os candidatos (universo completo não visto)."""
    user_item_matrix = train_df.pivot_table(
        index='user', columns='item', values='rating', fill_value=0
    )
    neighbor_ids = similar_users.index
    scores = {}
    for item in all_candidate_items:
        if item in user_item_matrix.columns:
            neighbor_ratings = user_item_matrix.loc[
                user_item_matrix.index.isin(neighbor_ids), item
            ]
            scores[item] = neighbor_ratings.mean()
        else:
            scores[item] = 0.0
    return pd.Series(scores).sort_values(ascending=False)

# ──────────────────────────────────────────────
# 1. Carregar dados
# ──────────────────────────────────────────────
print("A carregar dados...")
ratings = pd.read_csv(PATH_RATINGS).fillna(0)
ratings = ratings.rename(columns={"user_id": "user", "image_id": "item"})
ratings = split.remove_degenerate_users(ratings)

all_items = set(ratings['item'].unique())
print(f"  Users: {ratings['user'].nunique()} | Items no catálogo: {len(all_items)}")

# ──────────────────────────────────────────────
# 2. Loop kfold — avaliação sobre catálogo completo
# ──────────────────────────────────────────────
cols = [f'top{i}' for i in range(1, TOP_K + 1)]

acc_prec = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_rec  = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_f1   = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_mrr  = pd.DataFrame(0.0, index=range(1), columns=cols)

precision1_all, recall10_all, hit10_all, ndcg10_all, mrr10_all = [], [], [], [], []

print(f"\nAvaliação sobre catálogo completo — kfold ({K_FOLDS} folds)...")

for k, train_df, test_df in split.kfold_split(ratings, K_FOLDS, TOTAL_ITEMS, MIN_TEST, BASE_SEED):
    print(f"  Fold {k+1}/{K_FOLDS}...", end=" ", flush=True)

    sim_df     = cosine_sim_vector(train_df)
    train_seen = train_df.groupby("user")["item"].apply(set).to_dict()

    precision_list, recall_list, f1_list, mrr_list = [], [], [], []
    precision1_list, recall10_list, hit10_list, ndcg10_list, mrr10_list = [], [], [], [], []

    for user_id in test_df["user"].unique():
        user_test = test_df[test_df["user"] == user_id]
        relevant  = user_test[user_test["rating"] == 1]["item"].tolist()
        if len(relevant) == 0:
            continue
        if user_id not in sim_df.index:
            continue

        # Candidatos = todas as imagens não vistas pelo utilizador
        seen = train_seen.get(user_id, set())
        candidates = list(all_items - seen)

        if len(candidates) == 0:
            continue

        k_most_sim = get_k_most_sim_users(sim_df, user_id, K)
        scores     = get_scores_full(candidates, k_most_sim, train_df)
        recs       = scores.index.tolist()[:TOP_K]

        def pad(lst, k):
            return (lst + [lst[-1]] * k)[:k] if lst else [0.0] * k

        precision_list.append(pad(evaluation.precision_curve(recs, relevant), TOP_K))
        recall_list.append(pad(evaluation.recall_curve(recs, relevant), TOP_K))
        f1_list.append(pad(evaluation.f1_curve(recs, relevant), TOP_K))
        mrr_list.append(pad(evaluation.mrr_curve(recs, relevant), TOP_K))

        precision1_list.append(evaluation.precision_at_k(recs, relevant, 1))
        recall10_list.append(evaluation.recall_at_k(recs, relevant, 10))
        hit10_list.append(evaluation.hit_rate_at_k(recs, relevant, 10))
        ndcg10_list.append(evaluation.ndcg_at_k(recs, relevant, 10))
        mrr_curve_vals = evaluation.mrr_curve(recs, relevant)
        mrr10_list.append(mrr_curve_vals[9] if len(mrr_curve_vals) >= 10 else mrr_curve_vals[-1])

    acc_prec += pd.DataFrame([np.mean(precision_list, axis=0)], columns=cols)
    acc_rec  += pd.DataFrame([np.mean(recall_list,    axis=0)], columns=cols)
    acc_f1   += pd.DataFrame([np.mean(f1_list,        axis=0)], columns=cols)
    acc_mrr  += pd.DataFrame([np.mean(mrr_list,       axis=0)], columns=cols)

    precision1_all.extend(precision1_list)
    recall10_all.extend(recall10_list)
    hit10_all.extend(hit10_list)
    ndcg10_all.extend(ndcg10_list)
    mrr10_all.extend(mrr10_list)

    print(f"Precision@1={round(np.mean(precision1_list),4)} | NDCG@10={round(np.mean(ndcg10_list),4)} | MRR@10={round(np.mean(mrr10_list),4)}")

# ──────────────────────────────────────────────
# 3. Resultados finais
# ──────────────────────────────────────────────
mean_prec = acc_prec / K_FOLDS
mean_rec  = acc_rec  / K_FOLDS
mean_f1   = acc_f1   / K_FOLDS
mean_mrr  = acc_mrr  / K_FOLDS

print("\n" + "="*60)
print("KNN — Avaliação Catálogo Completo (kfold5)")
print("="*60)
print("\nPrecision@k:") ; print(mean_prec.to_string(index=False))
print("\nRecall@k:")    ; print(mean_rec.to_string(index=False))
print("\nF1@k:")        ; print(mean_f1.to_string(index=False))
print("\nMRR@k:")       ; print(mean_mrr.to_string(index=False))

print("\n── Métricas fixas (média dos folds) ──")
print(f"  Precision@1:  {round(np.mean(precision1_all), 4)}")
print(f"  Recall@10:    {round(np.mean(recall10_all), 4)}")
print(f"  HitRate@10:   {round(np.mean(hit10_all), 4)}")
print(f"  NDCG@10:      {round(np.mean(ndcg10_all), 4)}")
print(f"  MRR@10:       {round(np.mean(mrr10_all), 4)}")
print(f"  N_users:      {len(precision1_all)}")

# Guardar
os.makedirs("../results", exist_ok=True)
fixed = pd.DataFrame([{
    "model":        "KNN_All_FullCatalog",
    "Precision@1":  round(np.mean(precision1_all), 4),
    "Recall@10":    round(np.mean(recall10_all), 4),
    "HitRate@10":   round(np.mean(hit10_all), 4),
    "NDCG@10":      round(np.mean(ndcg10_all), 4),
    "MRR@10":       round(np.mean(mrr10_all), 4)
}])
fixed.to_csv("../results/fixed_metrics_knn_full.csv", index=False)
print("\nResultados guardados em results/fixed_metrics_knn_full.csv")
print("Done!")