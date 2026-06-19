import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from scipy.sparse import coo_matrix
import os
import json
 
import split
import evaluation
from lightgcn_model import LightGCN
 
# ── Configuração base
PATH_RATINGS = "data/ratings_full.csv"
EMBEDDING_DIM = 64
N_EPOCHS      = 100
BATCH_SIZE    = 1024
TOP_K         = 10
K_FOLDS       = 5
TOTAL_ITEMS   = 1300
MIN_TEST      = 20
BASE_SEED     = 42
 
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
device = torch.device("cpu")
 
FEATURE_COLS = [
    "rating", "valence", "arousal", "dominance",
    "happiness", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
]
 
# ── Carregar melhores hiperparâmetros do tuning
PARAMS_PATH = "results/lightgcn_best_params.json"
if os.path.exists(PARAMS_PATH):
    with open(PARAMS_PATH, "r") as f:
        best_params = json.load(f)
    LR           = best_params["lr"]
    WEIGHT_DECAY = best_params["weight_decay"]
    N_LAYERS     = best_params["n_layers"]
    print(f"Hiperparâmetros carregados de {PARAMS_PATH}")
else:
    print("AVISO: lightgcn_best_params.json não encontrado. A usar valores default.")
    LR           = 0.001
    WEIGHT_DECAY = 0.01
    N_LAYERS     = 3
 
print(f"  lr={round(LR,6)} | weight_decay={round(WEIGHT_DECAY,6)} | n_layers={N_LAYERS}")
 
# ──────────────────────────────────────────────
# 1. Carregar dados
# ──────────────────────────────────────────────
print("\nA carregar dados...")
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
 
print(f"  Users: {n_users} | Items: {n_items}")
 
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
# 3. Loop kfold
# ──────────────────────────────────────────────
cols = [f'top{i}' for i in range(1, TOP_K + 1)]
 
acc_prec = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_rec  = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_f1   = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_mrr  = pd.DataFrame(0.0, index=range(1), columns=cols)
 
precision1_all, recall10_all, hit10_all, ndcg10_all = [], [], [], []
 
print(f"\nA iniciar kfold ({K_FOLDS} folds)...")
 
for k, train_df, test_df in split.kfold_split(ratings, K_FOLDS, TOTAL_ITEMS, MIN_TEST, BASE_SEED):
    print(f"\n  Fold {k+1}/{K_FOLDS}...")
 
    torch.manual_seed(BASE_SEED + k)
    np.random.seed(BASE_SEED + k)
 
    # Interações positivas para o grafo
    train_pos = train_df[train_df['rating'] == 1].copy()
 
    # Remapear índices para este fold
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
 
    # Construir grafo
    adj_matrix = build_adj_matrix(train_pos, fold_n_users, fold_n_items)
 
    train_pos_set = train_pos.groupby('user_idx')['item_idx'].apply(set).to_dict()
    train_pairs   = list(zip(train_pos['user_idx'].values, train_pos['item_idx'].values))
 
    # Modelo e otimizador
    model     = LightGCN(fold_n_users, fold_n_items, EMBEDDING_DIM, N_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
 
    # Treino
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        pairs = train_pairs.copy()
        np.random.shuffle(pairs)
 
        total_loss, n_batches = 0, 0
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
 
            total_loss += loss.item()
            n_batches  += 1
 
        if epoch % 25 == 0:
            print(f"    Epoch {epoch:3d}/{N_EPOCHS} | Loss: {total_loss/n_batches:.4f}")
 
    # Avaliação
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(adj_matrix)
        user_emb = user_emb.cpu().numpy()
        item_emb = item_emb.cpu().numpy()
 
    precision_list, recall_list, f1_list, mrr_list = [], [], [], []
    precision1_list, recall10_list, hit10_list, ndcg10_list = [], [], [], []
 
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
        ranked_items = [test_items_mapped[i] for i in ranked_idx[:TOP_K]]
 
        p = evaluation.precision_curve(ranked_items, relevant)
        r = evaluation.recall_curve(ranked_items, relevant)
        f = evaluation.f1_curve(ranked_items, relevant)
        m = evaluation.mrr_curve(ranked_items, relevant)
 
        # Garantir comprimento fixo TOP_K
        def pad(lst, k):
            return (lst + [lst[-1]] * k)[:k] if lst else [0.0] * k
 
        precision_list.append(pad(p, TOP_K))
        recall_list.append(pad(r, TOP_K))
        f1_list.append(pad(f, TOP_K))
        mrr_list.append(pad(m, TOP_K))
 
        precision1_list.append(evaluation.precision_at_k(ranked_items, relevant, 1))
        recall10_list.append(evaluation.recall_at_k(ranked_items, relevant, 10))
        hit10_list.append(evaluation.hit_rate_at_k(ranked_items, relevant, 10))
        ndcg10_list.append(evaluation.ndcg_at_k(ranked_items, relevant, 10))
 
    # Acumular
    acc_prec += pd.DataFrame([np.mean(precision_list, axis=0)], columns=cols)
    acc_rec  += pd.DataFrame([np.mean(recall_list,    axis=0)], columns=cols)
    acc_f1   += pd.DataFrame([np.mean(f1_list,        axis=0)], columns=cols)
    acc_mrr  += pd.DataFrame([np.mean(mrr_list,       axis=0)], columns=cols)
 
    precision1_all.extend(precision1_list)
    recall10_all.extend(recall10_list)
    hit10_all.extend(hit10_list)
    ndcg10_all.extend(ndcg10_list)
 
    print(f"    Precision@1={round(np.mean(precision1_list),4)} | NDCG@10={round(np.mean(ndcg10_list),4)}")
 
# ──────────────────────────────────────────────
# 4. Média dos folds
# ──────────────────────────────────────────────
mean_prec = acc_prec / K_FOLDS
mean_rec  = acc_rec  / K_FOLDS
mean_f1   = acc_f1   / K_FOLDS
mean_mrr  = acc_mrr  / K_FOLDS
 
print("\n" + "="*50)
print("LightGCN — Resultados Finais (kfold5)")
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
os.makedirs("results", exist_ok=True)
mean_prec_save = mean_prec.copy(); mean_prec_save.insert(0, 'model', 'LightGCN')
mean_rec_save  = mean_rec.copy();  mean_rec_save.insert(0,  'model', 'LightGCN')
mean_f1_save   = mean_f1.copy();   mean_f1_save.insert(0,   'model', 'LightGCN')
mean_mrr_save  = mean_mrr.copy();  mean_mrr_save.insert(0,  'model', 'LightGCN')
 
mean_prec_save.to_csv("results/precision_lightgcn.csv", index=False)
mean_rec_save.to_csv("results/recall_lightgcn.csv",     index=False)
mean_f1_save.to_csv("results/f1_lightgcn.csv",          index=False)
mean_mrr_save.to_csv("results/mrr_lightgcn.csv",        index=False)
 
print("\nResultados guardados em results/")
print("Done!")
 
# Guardar métricas fixas
fixed_metrics = pd.DataFrame([{
    "model":        "LightGCN",
    "Precision@1":  round(np.mean(precision1_all), 4),
    "Recall@10":    round(np.mean(recall10_all), 4),
    "HitRate@10":   round(np.mean(hit10_all), 4),
    "NDCG@10":      round(np.mean(ndcg10_all), 4)
}])
fixed_metrics.to_csv("results/fixed_metrics_lightgcn.csv", index=False)
print("Métricas fixas guardadas em results/fixed_metrics_lightgcn.csv")