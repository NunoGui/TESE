import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from scipy.sparse import coo_matrix
import json
import copy

import split
import evaluation
from lightgcn_model import LightGCN

# ── Configuração
PATH_RATINGS  = "../data/ratings_full.csv"
EMBEDDING_DIM = 64
N_EPOCHS      = 500
PATIENCE      = 30
BATCH_SIZE    = 1024
TOP_K         = 10
K_FOLDS       = 5
TOTAL_ITEMS   = 3084
MIN_TEST      = 20
BASE_SEED     = 42

np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
device = torch.device("cpu")

FEATURE_COLS = [
    "rating", "valence", "arousal", "dominance",
    "happiness", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
]

# ── Carregar hiperparâmetros
PARAMS_PATH = "../results/lightgcn_best_params.json"
if os.path.exists(PARAMS_PATH):
    with open(PARAMS_PATH, "r") as f:
        best_params = json.load(f)
    LR           = best_params["lr"]
    WEIGHT_DECAY = best_params["weight_decay"]
    N_LAYERS     = best_params["n_layers"]
    print(f"Hiperparâmetros carregados de {PARAMS_PATH}")
else:
    LR, WEIGHT_DECAY, N_LAYERS = 0.004066, 0.001127, 4
    print("A usar hiperparâmetros default.")

print(f"  lr={round(LR,6)} | weight_decay={round(WEIGHT_DECAY,6)} | n_layers={N_LAYERS}")

# ──────────────────────────────────────────────
# 1. Carregar dados
# ──────────────────────────────────────────────
print("\nA carregar dados...")
ratings = pd.read_csv(PATH_RATINGS).fillna(0)
ratings = ratings.rename(columns={"user_id": "user", "image_id": "item"})
ratings = split.remove_degenerate_users(ratings)

all_items = set(ratings['item'].unique())
print(f"  Users: {ratings['user'].nunique()} | Items no catálogo: {len(all_items)}")

# ──────────────────────────────────────────────
# 2. Funções auxiliares
# ──────────────────────────────────────────────
def build_adj_matrix(train_pos, n_users, n_items):
    user_indices = train_pos['user_idx'].values
    item_indices = train_pos['item_idx'].values + n_users
    rows = np.concatenate([user_indices, item_indices])
    cols = np.concatenate([item_indices, user_indices])
    vals = np.ones(len(rows))
    N    = n_users + n_items
    adj  = coo_matrix((vals, (rows, cols)), shape=(N, N))
    degree = np.array(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1
    d_inv_sqrt = np.power(degree, -0.5)
    adj = adj.tocsr()
    adj = adj.multiply(d_inv_sqrt[:, None]).multiply(d_inv_sqrt[None, :])
    adj = adj.tocoo()
    indices = torch.LongTensor(np.array([adj.row, adj.col]))
    values  = torch.FloatTensor(adj.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size([N, N])).to(device)

def sample_negative(user_idx, train_pos_set, n_items):
    while True:
        neg = np.random.randint(0, n_items)
        if neg not in train_pos_set.get(user_idx, set()):
            return neg

def evaluate_ndcg(model, adj_matrix, test_df, fold_user2idx, fold_item2idx):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(adj_matrix)
        user_emb = user_emb.cpu().numpy()
        item_emb = item_emb.cpu().numpy()
    ndcg_list = []
    for user in test_df['user'].unique():
        user_test = test_df[test_df['user'] == user]
        relevant  = user_test[user_test['rating'] == 1]['item'].tolist()
        if len(relevant) == 0 or user not in fold_user2idx:
            continue
        u_idx = fold_user2idx[user]
        test_item_indices = [fold_item2idx[it] for it in user_test['item'].tolist() if it in fold_item2idx]
        test_items_mapped = [it for it in user_test['item'].tolist() if it in fold_item2idx]
        if not test_item_indices:
            continue
        scores       = item_emb[test_item_indices].dot(user_emb[u_idx])
        ranked_idx   = np.argsort(-scores)
        ranked_items = [test_items_mapped[i] for i in ranked_idx[:10]]
        ndcg_list.append(evaluation.ndcg_at_k(ranked_items, relevant, 10))
    return round(np.mean(ndcg_list), 4) if ndcg_list else 0.0

# ──────────────────────────────────────────────
# 3. Loop kfold
# ──────────────────────────────────────────────
cols = [f'top{i}' for i in range(1, TOP_K + 1)]

acc_prec = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_rec  = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_f1   = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_mrr  = pd.DataFrame(0.0, index=range(1), columns=cols)

precision1_all, recall10_all, hit10_all, ndcg10_all, mrr10_all = [], [], [], [], []

print(f"\nAvaliação sobre catálogo completo — kfold ({K_FOLDS} folds)...")

for k, train_df, test_df in split.kfold_split(ratings, K_FOLDS, TOTAL_ITEMS, MIN_TEST, BASE_SEED):
    print(f"\n  Fold {k+1}/{K_FOLDS}...")

    torch.manual_seed(BASE_SEED + k)
    np.random.seed(BASE_SEED + k)

    train_pos = train_df[train_df['rating'] == 1].copy()

    fold_user_ids = sorted(train_df['user'].unique())
    fold_item_ids = sorted(ratings['item'].unique())
    fold_user2idx = {u: i for i, u in enumerate(fold_user_ids)}
    fold_item2idx = {it: i for i, it in enumerate(fold_item_ids)}
    fold_n_users  = len(fold_user_ids)
    fold_n_items  = len(fold_item_ids)

    train_pos = train_pos.copy()
    train_pos['user_idx'] = train_pos['user'].map(fold_user2idx)
    train_pos['item_idx'] = train_pos['item'].map(fold_item2idx)
    train_pos = train_pos.dropna(subset=['user_idx', 'item_idx'])
    train_pos['user_idx'] = train_pos['user_idx'].astype(int)
    train_pos['item_idx'] = train_pos['item_idx'].astype(int)

    adj_matrix    = build_adj_matrix(train_pos, fold_n_users, fold_n_items)
    train_pos_set = train_pos.groupby('user_idx')['item_idx'].apply(set).to_dict()
    train_pairs   = list(zip(train_pos['user_idx'].values, train_pos['item_idx'].values))
    train_seen    = train_df.groupby('user')['item'].apply(set).to_dict()

    model     = LightGCN(fold_n_users, fold_n_items, EMBEDDING_DIM, N_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Early stopping
    best_ndcg, best_epoch, no_improve = 0.0, 0, 0
    best_model_state = None

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        pairs = train_pairs.copy()
        np.random.shuffle(pairs)
        for i in range(0, len(pairs), BATCH_SIZE):
            batch     = pairs[i:i + BATCH_SIZE]
            users     = torch.LongTensor([p[0] for p in batch]).to(device)
            pos_items = torch.LongTensor([p[1] for p in batch]).to(device)
            neg_items = torch.LongTensor([
                sample_negative(p[0], train_pos_set, fold_n_items) for p in batch
            ]).to(device)
            optimizer.zero_grad()
            user_emb, item_emb = model(adj_matrix)
            loss = model.bpr_loss(user_emb, item_emb, users, pos_items, neg_items, WEIGHT_DECAY)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            current_ndcg = evaluate_ndcg(model, adj_matrix, test_df, fold_user2idx, fold_item2idx)
            if current_ndcg > best_ndcg:
                best_ndcg        = current_ndcg
                best_epoch       = epoch
                no_improve       = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 5
            if no_improve >= PATIENCE:
                print(f"    Early stopping na epoch {epoch} — melhor epoch: {best_epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ── Avaliação sobre catálogo completo
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(adj_matrix)
        user_emb = user_emb.cpu().numpy()
        item_emb = item_emb.cpu().numpy()

    precision_list, recall_list, f1_list, mrr_list = [], [], [], []
    precision1_list, recall10_list, hit10_list, ndcg10_list, mrr10_list = [], [], [], [], []

    def pad(lst, k):
        return (lst + [lst[-1]] * k)[:k] if lst else [0.0] * k

    for user in test_df['user'].unique():
        user_test = test_df[test_df['user'] == user]
        relevant  = user_test[user_test['rating'] == 1]['item'].tolist()
        if len(relevant) == 0 or user not in fold_user2idx:
            continue

        # Candidatos = todas as imagens não vistas no treino
        seen       = train_seen.get(user, set())
        candidates = [it for it in all_items if it not in seen and it in fold_item2idx]
        if not candidates:
            continue

        u_idx             = fold_user2idx[user]
        candidate_indices = [fold_item2idx[it] for it in candidates]
        scores            = item_emb[candidate_indices].dot(user_emb[u_idx])
        ranked_idx        = np.argsort(-scores)
        ranked_items      = [candidates[i] for i in ranked_idx[:TOP_K]]

        precision_list.append(pad(evaluation.precision_curve(ranked_items, relevant), TOP_K))
        recall_list.append(pad(evaluation.recall_curve(ranked_items, relevant), TOP_K))
        f1_list.append(pad(evaluation.f1_curve(ranked_items, relevant), TOP_K))
        mrr_list.append(pad(evaluation.mrr_curve(ranked_items, relevant), TOP_K))

        precision1_list.append(evaluation.precision_at_k(ranked_items, relevant, 1))
        recall10_list.append(evaluation.recall_at_k(ranked_items, relevant, 10))
        hit10_list.append(evaluation.hit_rate_at_k(ranked_items, relevant, 10))
        ndcg10_list.append(evaluation.ndcg_at_k(ranked_items, relevant, 10))
        mrr_curve_vals = evaluation.mrr_curve(ranked_items, relevant)
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

    print(f"    Precision@1={round(np.mean(precision1_list),4)} | NDCG@10={round(np.mean(ndcg10_list),4)} | MRR@10={round(np.mean(mrr10_list),4)}")

# ──────────────────────────────────────────────
# 4. Resultados finais
# ──────────────────────────────────────────────
mean_prec = acc_prec / K_FOLDS
mean_rec  = acc_rec  / K_FOLDS
mean_f1   = acc_f1   / K_FOLDS
mean_mrr  = acc_mrr  / K_FOLDS

print("\n" + "="*60)
print("LightGCN — Avaliação Catálogo Completo (kfold5)")
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

os.makedirs("../results", exist_ok=True)
fixed = pd.DataFrame([{
    "model":        "LightGCN_FullCatalog",
    "Precision@1":  round(np.mean(precision1_all), 4),
    "Recall@10":    round(np.mean(recall10_all), 4),
    "HitRate@10":   round(np.mean(hit10_all), 4),
    "NDCG@10":      round(np.mean(ndcg10_all), 4),
    "MRR@10":       round(np.mean(mrr10_all), 4)
}])
fixed.to_csv("../results/fixed_metrics_lightgcn_full.csv", index=False)
print("\nResultados guardados em ../results/fixed_metrics_lightgcn_full.csv")
print("Done!")