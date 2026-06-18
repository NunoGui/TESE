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
N_TRAIN       = 12
N_TEST_MIN    = 20
SEED          = 42
N_TRIALS      = 20
 
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")
 
FEATURE_COLS = [
    "rating", "valence", "arousal", "dominance",
    "happiness", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
]
 
# ──────────────────────────────────────────────
# 1. Carregar e preparar dados
# ──────────────────────────────────────────────
print("A carregar dados...")
ratings = pd.read_csv(PATH_RATINGS).fillna(0)
ratings = ratings.rename(columns={"user_id": "user", "image_id": "item"})
ratings = split.remove_degenerate_users(ratings)
 
user_ids = sorted(ratings['user'].unique())
item_ids = sorted(ratings['item'].unique())
user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {it: i for i, it in enumerate(item_ids)}
n_users  = len(user_ids)
n_items  = len(item_ids)
 
ratings['user_idx'] = ratings['user'].map(user2idx)
ratings['item_idx'] = ratings['item'].map(item2idx)
 
print("\nA dividir dados treino/teste...")
train_df, test_df = split.split_user_data(ratings, FEATURE_COLS, N_TRAIN, N_TEST_MIN, SEED)
train_pos = train_df[train_df['rating'] == 1].copy()
print(f"  Treino positivos: {len(train_pos)} | Teste: {len(test_df)}")
 
# ──────────────────────────────────────────────
# 2. Construir matriz de adjacência
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
 
adj_matrix = build_adj_matrix(train_pos, n_users, n_items)
 
train_pos_set = train_pos.groupby('user_idx')['item_idx'].apply(set).to_dict()
train_pairs   = list(zip(train_pos['user_idx'].values, train_pos['item_idx'].values))
 
def sample_negative(user_idx, train_pos_set, n_items):
    while True:
        neg = np.random.randint(0, n_items)
        if neg not in train_pos_set.get(user_idx, set()):
            return neg
 
# ──────────────────────────────────────────────
# 3. Função de treino e avaliação
# ──────────────────────────────────────────────
def train_and_evaluate(lr, weight_decay, n_layers):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
 
    model     = LightGCN(n_users, n_items, EMBEDDING_DIM, n_layers).to(device)
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
                sample_negative(p[0], train_pos_set, n_items) for p in batch
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
        if len(relevant) == 0 or user not in user2idx:
            continue
        u_idx             = user2idx[user]
        test_item_indices = [item2idx[it] for it in test_items if it in item2idx]
        test_items_mapped = [it for it in test_items if it in item2idx]
        if len(test_item_indices) == 0:
            continue
        scores       = item_emb[test_item_indices].dot(user_emb[u_idx])
        ranked_idx   = np.argsort(-scores)
        ranked_items = [test_items_mapped[i] for i in ranked_idx[:10]]
        ndcg_list.append(evaluation.ndcg_at_k(ranked_items, relevant, 10))
 
    return round(np.mean(ndcg_list), 4)
 
# ──────────────────────────────────────────────
# 4. Optuna — objetivo
# ──────────────────────────────────────────────
trial_results = []
 
def objective(trial):
    lr           = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    n_layers     = trial.suggest_int("n_layers", 1, 4)
 
    ndcg = train_and_evaluate(lr, weight_decay, n_layers)
 
    trial_results.append({
        "trial":        trial.number + 1,
        "lr":           round(lr, 6),
        "weight_decay": round(weight_decay, 6),
        "n_layers":     n_layers,
        "ndcg@10":      ndcg
    })
 
    print(f"Trial {trial.number+1:2d}/{N_TRIALS} | lr={round(lr,5)} | wd={round(weight_decay,5)} | layers={n_layers} | NDCG@10={ndcg}")
    return ndcg
 
# ──────────────────────────────────────────────
# 5. Correr Optuna
# ──────────────────────────────────────────────
print(f"\nA iniciar Optuna TPE Search ({N_TRIALS} trials)...")
print("="*60)
 
sampler = optuna.samplers.TPESampler(seed=SEED)
study   = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)
 
# ──────────────────────────────────────────────
# 6. Resultados
# ──────────────────────────────────────────────
best = study.best_params
best_ndcg = study.best_value
 
print("\n" + "="*60)
print("RESULTADOS DO TUNING")
print("="*60)
 
results_df = pd.DataFrame(trial_results).sort_values("ndcg@10", ascending=False)
print(results_df.to_string(index=False))
 
print(f"\n── Melhores hiperparâmetros (Optuna TPE) ──")
print(f"  lr:           {best['lr']:.6f}")
print(f"  weight_decay: {best['weight_decay']:.6f}")
print(f"  n_layers:     {best['n_layers']}")
print(f"  NDCG@10:      {best_ndcg}")
 
# Guardar
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/lightgcn_tuning.csv", index=False)
with open("results/lightgcn_best_params.json", "w") as f:
    json.dump(best, f, indent=2)
 
print("\nResultados guardados em results/")
print("Done!")