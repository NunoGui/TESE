import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
 
import split
import evaluation
 
PATH_RATINGS = "data/ratings_full.csv"
K          = 10   
TOP_K      = 10   
K_FOLDS    = 5
TOTAL_ITEMS = 1300
MIN_TEST   = 20
BASE_SEED  = 42
 
np.random.seed(BASE_SEED)
 
FEATURE_COLS = [
    "rating",
    "valence", "arousal", "dominance",
    "happiness", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
]
 
# ──────────────────────────────────────────────
# 1. Carregar dados
# ──────────────────────────────────────────────
print("A carregar dados...")
ratings = pd.read_csv(PATH_RATINGS).fillna(0)
ratings = ratings.rename(columns={"user_id": "user", "image_id": "item"})
ratings = split.remove_degenerate_users(ratings)
 
print(f"  Users: {ratings['user'].nunique()} | Items: {ratings['item'].nunique()}")
 
# ──────────────────────────────────────────────
# 2. Função de recomendação KNN
# ──────────────────────────────────────────────
def recommend_knn(user_id, sim_df, train_seen, train_df, test_items, k=10, top_k=10):
    if user_id not in sim_df.index:
        return test_items[:top_k]
 
    neighbors = sim_df[user_id].drop(index=user_id).nlargest(k).index.tolist()
    scores = {item: 0.0 for item in test_items}
    for neighbor in neighbors:
        sim_weight = sim_df.loc[user_id, neighbor]
        for _, row in train_df[train_df["user"] == neighbor].iterrows():
            if row["item"] in scores:
                scores[row["item"]] += row["rating"] * sim_weight
 
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [img for img, _ in ranked[:top_k]]
 
# ──────────────────────────────────────────────
# 3. Loop kfold
# ──────────────────────────────────────────────
cols = [f'top{i}' for i in range(1, TOP_K + 1)]
 
# Acumuladores (estilo Ana)
acc_prec = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_rec  = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_f1   = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_mrr  = pd.DataFrame(0.0, index=range(1), columns=cols)
 
precision1_all, recall10_all, hit10_all, ndcg10_all = [], [], [], []
 
print(f"\nA iniciar kfold ({K_FOLDS} folds)...")
 
for k, train_df, test_df in split.kfold_split(ratings, K_FOLDS, TOTAL_ITEMS, MIN_TEST, BASE_SEED):
    print(f"  Fold {k+1}/{K_FOLDS}...", end=" ", flush=True)
 
    # Perfil de user e similaridade
    user_profiles = train_df.groupby("user")[FEATURE_COLS].mean()
    scaler = StandardScaler()
    user_profiles_scaled = pd.DataFrame(
        scaler.fit_transform(user_profiles),
        index=user_profiles.index,
        columns=user_profiles.columns
    )
 
    sim_matrix = cosine_similarity(user_profiles_scaled)
    sim_df = pd.DataFrame(
        sim_matrix,
        index=user_profiles_scaled.index,
        columns=user_profiles_scaled.index
    )
    train_seen = train_df.groupby("user")["item"].apply(set).to_dict()
 
    # Avaliação
    precision_list, recall_list, f1_list, mrr_list = [], [], [], []
    precision1_list, recall10_list, hit10_list, ndcg10_list = [], [], [], []
 
    for user_id in test_df["user"].unique():
        user_test  = test_df[test_df["user"] == user_id]
        test_items = user_test["item"].tolist()
        relevant   = user_test[user_test["rating"] == 1]["item"].tolist()
        if len(relevant) == 0:
            continue
 
        recs = recommend_knn(user_id, sim_df, train_seen, train_df, test_items, k=K, top_k=TOP_K)
 
        precision_list.append(evaluation.precision_curve(recs, relevant)[:TOP_K])
        recall_list.append(evaluation.recall_curve(recs, relevant)[:TOP_K])
        f1_list.append(evaluation.f1_curve(recs, relevant)[:TOP_K])
        mrr_list.append(evaluation.mrr_curve(recs, relevant)[:TOP_K])
 
        precision1_list.append(evaluation.precision_at_k(recs, relevant, 1))
        recall10_list.append(evaluation.recall_at_k(recs, relevant, 10))
        hit10_list.append(evaluation.hit_rate_at_k(recs, relevant, 10))
        ndcg10_list.append(evaluation.ndcg_at_k(recs, relevant, 10))
 
    # Acumular resultados do fold
    acc_prec += pd.DataFrame([np.mean(precision_list, axis=0)], columns=cols)
    acc_rec  += pd.DataFrame([np.mean(recall_list,    axis=0)], columns=cols)
    acc_f1   += pd.DataFrame([np.mean(f1_list,        axis=0)], columns=cols)
    acc_mrr  += pd.DataFrame([np.mean(mrr_list,       axis=0)], columns=cols)
 
    precision1_all.extend(precision1_list)
    recall10_all.extend(recall10_list)
    hit10_all.extend(hit10_list)
    ndcg10_all.extend(ndcg10_list)
 
    print(f"Precision@1={round(np.mean(precision1_list), 4)} | NDCG@10={round(np.mean(ndcg10_list), 4)}")
 
# ──────────────────────────────────────────────
# 4. Média dos folds
# ──────────────────────────────────────────────
mean_prec = acc_prec / K_FOLDS
mean_rec  = acc_rec  / K_FOLDS
mean_f1   = acc_f1   / K_FOLDS
mean_mrr  = acc_mrr  / K_FOLDS
 
print("\n" + "="*50)
print("KNN User-Based — Resultados Finais (kfold5)")
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
 
# Guardar resultados
import os
os.makedirs("results", exist_ok=True)
mean_prec.insert(0, 'model', 'KNN_All')
mean_rec.insert(0,  'model', 'KNN_All')
mean_f1.insert(0,   'model', 'KNN_All')
mean_mrr.insert(0,  'model', 'KNN_All')
 
mean_prec.to_csv("results/precision_knn.csv", index=False)
mean_rec.to_csv("results/recall_knn.csv",     index=False)
mean_f1.to_csv("results/f1_knn.csv",          index=False)
mean_mrr.to_csv("results/mrr_knn.csv",        index=False)
 
print("\nResultados guardados em results/")
print("Done!")
 
# Guardar métricas fixas
fixed_metrics = pd.DataFrame([{
    "model":        "KNN_All",
    "Precision@1":  round(np.mean(precision1_all), 4),
    "Recall@10":    round(np.mean(recall10_all), 4),
    "HitRate@10":   round(np.mean(hit10_all), 4),
    "NDCG@10":      round(np.mean(ndcg10_all), 4)
}])
fixed_metrics.to_csv("results/fixed_metrics_knn.csv", index=False)
print("Métricas fixas guardadas em results/fixed_metrics_knn.csv")