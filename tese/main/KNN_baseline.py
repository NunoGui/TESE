import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import split
import evaluation

# ── Configuração
PATH_RATINGS = "data/ratings_full.csv"
K          = 20   # número de vizinhos (igual à Ana com n=20)
TOP_K      = 10   # tamanho da lista de recomendações
K_FOLDS    = 5
TOTAL_ITEMS = 3084
MIN_TEST   = 20
BASE_SEED  = 42

np.random.seed(BASE_SEED)

# Features usadas para similaridade (igual ao CF-KNN-All da Ana)
FEATURE_COLS = ['rating', 'anger', 'fear', 'disgust', 'sadness',
                'happiness', 'surprise', 'neutral', 'valence', 'arousal', 'dominance']

# ──────────────────────────────────────────────
# Funções de similaridade (estilo Ana)
# ──────────────────────────────────────────────
def cosine_sim_vector(df):
    """
    Constrói matriz user x (item x feature) e calcula cosine similarity.
    Equivalente ao cosine_sim_vector da Ana.
    """
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_pivot = df.pivot_table(
        index='user',
        columns='item',
        values=feature_cols,
        fill_value=0
    )
    df_pivot.columns = [f"{item}_{feat}" for feat, item in df_pivot.columns]
    df_pivot = df_pivot.fillna(0)

    sim_matrix = cosine_similarity(df_pivot)
    return pd.DataFrame(sim_matrix, index=df_pivot.index, columns=df_pivot.index)


def get_k_most_sim_users(sim_df, user, k):
    """Top K utilizadores mais similares (excluindo o próprio)."""
    sim_scores = sim_df.loc[user].drop(user)
    return sim_scores.nlargest(k)


def get_scores(test_items, similar_users, train_df):
    """
    Score de cada item = média dos ratings dos vizinhos para esse item.
    Equivalente ao get_scores da Ana.
    """
    user_item_matrix = train_df.pivot_table(
        index='user',
        columns='item',
        values='rating',
        fill_value=0
    )
    neighbor_ids = similar_users.index
    scores = {}
    for item in test_items:
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

print(f"  Users: {ratings['user'].nunique()} | Items: {ratings['item'].nunique()}")

# ──────────────────────────────────────────────
# 2. Loop kfold
# ──────────────────────────────────────────────
cols = [f'top{i}' for i in range(1, TOP_K + 1)]

acc_prec = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_rec  = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_f1   = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_mrr  = pd.DataFrame(0.0, index=range(1), columns=cols)

precision1_all, recall10_all, hit10_all, ndcg10_all, mrr10_all = [], [], [], [], []

print(f"\nA iniciar kfold ({K_FOLDS} folds)...")

all_recommendations = []  

for k, train_df, test_df in split.kfold_split(ratings, K_FOLDS, TOTAL_ITEMS, MIN_TEST, BASE_SEED):
    print(f"  Fold {k+1}/{K_FOLDS}...", end=" ", flush=True)

    # Calcular similaridade (estilo Ana)
    sim_df = cosine_sim_vector(train_df)

    # Avaliação
    precision_list, recall_list, f1_list, mrr_list = [], [], [], []
    precision1_list, recall10_list, hit10_list, ndcg10_list, mrr10_list = [], [], [], [], []

    for user_id in test_df["user"].unique():
        user_test  = test_df[test_df["user"] == user_id]
        test_items = user_test["item"].tolist()
        relevant   = user_test[user_test["rating"] == 1]["item"].tolist()
        if len(relevant) == 0:
            continue
        if user_id not in sim_df.index:
            continue

        # Obter K vizinhos mais similares
        k_most_sim = get_k_most_sim_users(sim_df, user_id, K)

        # Calcular scores (média dos ratings dos vizinhos)
        scores = get_scores(test_items, k_most_sim, train_df)
        recs   = scores.index.tolist()[:TOP_K]
        
        for rank, item_id in enumerate(recs, start=1):
            all_recommendations.append({
                "fold": k,
                "user": user_id,
                "rank": rank,
                "item": item_id,
                "score": scores.loc[item_id]
            })

        precision_list.append(evaluation.precision_curve(recs, relevant)[:TOP_K])
        recall_list.append(evaluation.recall_curve(recs, relevant)[:TOP_K])
        f1_list.append(evaluation.f1_curve(recs, relevant)[:TOP_K])
        mrr_list.append(evaluation.mrr_curve(recs, relevant)[:TOP_K])

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

    print(f"Precision@1={round(np.mean(precision1_list), 4)} | NDCG@10={round(np.mean(ndcg10_list), 4)}")

# ──────────────────────────────────────────────
# 3. Resultados finais
# ──────────────────────────────────────────────
mean_prec = acc_prec / K_FOLDS
mean_rec  = acc_rec  / K_FOLDS
mean_f1   = acc_f1   / K_FOLDS
mean_mrr  = acc_mrr  / K_FOLDS

print("\n" + "="*50)
print("CF-KNN-All — Resultados Finais (kfold5, n=20)")
print("="*50)
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

# Guardar resultados
import os
os.makedirs("results", exist_ok=True)
mean_prec_s = mean_prec.copy(); mean_prec_s.insert(0, 'model', 'KNN_All')
mean_rec_s  = mean_rec.copy();  mean_rec_s.insert(0,  'model', 'KNN_All')
mean_f1_s   = mean_f1.copy();   mean_f1_s.insert(0,   'model', 'KNN_All')
mean_mrr_s  = mean_mrr.copy();  mean_mrr_s.insert(0,  'model', 'KNN_All')

mean_prec_s.to_csv("results/precision_knn.csv", index=False)
mean_rec_s.to_csv("results/recall_knn.csv",     index=False)
mean_f1_s.to_csv("results/f1_knn.csv",          index=False)
mean_mrr_s.to_csv("results/mrr_knn.csv",        index=False)

recs_df = pd.DataFrame(all_recommendations)
recs_df.to_csv("results/recommendations_knn.csv", index=False)
print(f"Recomendações completas guardadas: {len(recs_df)} linhas")

fixed_metrics = pd.DataFrame([{
    "model":        "KNN_All",
    "Precision@1":  round(np.mean(precision1_all), 4),
    "Recall@10":    round(np.mean(recall10_all), 4),
    "HitRate@10":   round(np.mean(hit10_all), 4),
    "NDCG@10":      round(np.mean(ndcg10_all), 4),
    "MRR@10":       round(np.mean(mrr10_all), 4)
}])
fixed_metrics.to_csv("results/fixed_metrics_knn.csv", index=False)

print("\nResultados guardados em results/")
print("Done!")