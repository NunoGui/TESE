import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from scipy.sparse import coo_matrix
import os
import json
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
 
import split
import evaluation
from lightgcn_model import LightGCN
 
# ── Configuração
PATH_RATINGS  = "data/ratings_full.csv"
EMBEDDING_DIM = 64
N_EPOCHS      = 100
BATCH_SIZE    = 1024
K_FOLDS       = 5
TOTAL_ITEMS   = 3084
MIN_TEST      = 20
BASE_SEED     = 42
N_TRIALS      = 20
 
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
device = torch.device("cpu")
 
FEATURE_COLS = [
    "rating", "valence", "arousal", "dominance",
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
 
# ──────────────────────────────────────────────
# 3. Treino e avaliação com kfold5
# ──────────────────────────────────────────────
def train_and_evaluate_kfold(lr, weight_decay, n_layers):
    ndcg_folds = []
 
    for k, train_df, test_df in split.kfold_split(ratings, K_FOLDS, TOTAL_ITEMS, MIN_TEST, BASE_SEED):
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
 
        model     = LightGCN(fold_n_users, fold_n_items, EMBEDDING_DIM, n_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
 
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
                loss = model.bpr_loss(user_emb, item_emb, users, pos_items, neg_items, weight_decay)
                loss.backward()
                optimizer.step()
 
        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model(adj_matrix)
            user_emb = user_emb.cpu().numpy()
            item_emb = item_emb.cpu().numpy()
 
        ndcg_list = []
        for user in test_df['user'].unique():
            user_test  = test_df[test_df['user'] == user]
            test_items = user_test['item'].tolist()
            relevant   = user_test[user_test['rating'] == 1]['item'].tolist()
            if len(relevant) == 0 or user not in fold_user2idx:
                continue
            u_idx             = fold_user2idx[user]
            test_item_indices = [fold_item2idx[it] for it in test_items if it in fold_item2idx]
            test_items_mapped = [it for it in test_items if it in fold_item2idx]
            if len(test_item_indices) == 0:
                continue
            scores       = item_emb[test_item_indices].dot(user_emb[u_idx])
            ranked_idx   = np.argsort(-scores)
            ranked_items = [test_items_mapped[i] for i in ranked_idx[:10]]
            ndcg_list.append(evaluation.ndcg_at_k(ranked_items, relevant, 10))
 
        ndcg_folds.append(np.mean(ndcg_list))
 
    return round(np.mean(ndcg_folds), 4)
 
# ──────────────────────────────────────────────
# 4. Optuna TPE
# ──────────────────────────────────────────────
trial_results = []
 
def objective(trial):
    lr           = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    n_layers     = trial.suggest_int("n_layers", 1, 4)
 
    ndcg = train_and_evaluate_kfold(lr, weight_decay, n_layers)
 
    trial_results.append({
        "trial":        trial.number + 1,
        "lr":           round(lr, 6),
        "weight_decay": round(weight_decay, 6),
        "n_layers":     n_layers,
        "ndcg@10":      ndcg
    })
 
    print(f"Trial {trial.number+1:2d}/{N_TRIALS} | lr={round(lr,5)} | wd={round(weight_decay,5)} | layers={n_layers} | NDCG@10={ndcg}")
    return ndcg
 
print(f"\nA iniciar Optuna TPE Search ({N_TRIALS} trials x {K_FOLDS} folds)...")
print("="*60)
 
sampler = optuna.samplers.TPESampler(seed=BASE_SEED)
study   = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)
 
# ──────────────────────────────────────────────
# 5. Resultados
# ──────────────────────────────────────────────
best      = study.best_params
best_ndcg = study.best_value
 
print("\n" + "="*60)
print("RESULTADOS DO TUNING (kfold5)")
print("="*60)
 
results_df = pd.DataFrame(trial_results).sort_values("ndcg@10", ascending=False)
print(results_df.to_string(index=False))
 
print(f"\n── Melhores hiperparâmetros (Optuna TPE + kfold5) ──")
print(f"  lr:           {best['lr']:.6f}")
print(f"  weight_decay: {best['weight_decay']:.6f}")
print(f"  n_layers:     {best['n_layers']}")
print(f"  NDCG@10:      {best_ndcg}")
 
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/lightgcn_tuning.csv", index=False)
with open("results/lightgcn_best_params.json", "w") as f:
    json.dump(best, f, indent=2)
 
print("\nResultados guardados em results/")
print("Done!")