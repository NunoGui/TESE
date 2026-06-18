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
PATH_RATINGS  = "data/ratings_full.csv"
EMBEDDING_DIM = 64
N_EPOCHS      = 100
BATCH_SIZE    = 1024
TOP_K         = 10
N_TRAIN       = 12
N_TEST_MIN    = 20
SEED          = 42
 
np.random.seed(SEED)
torch.manual_seed(SEED)
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
 
print(f"  lr={LR} | weight_decay={WEIGHT_DECAY} | n_layers={N_LAYERS}")
 
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
# 2. Split treino/teste
# ──────────────────────────────────────────────
print("\nA dividir dados treino/teste...")
train_df, test_df = split.split_user_data(ratings, FEATURE_COLS, N_TRAIN, N_TEST_MIN, SEED)
train_pos = train_df[train_df['rating'] == 1].copy()
print(f"  Treino: {len(train_df)} | Teste: {len(test_df)}")
print(f"  Interações positivas no treino: {len(train_pos)}")
 
# ──────────────────────────────────────────────
# 3. Construir matriz de adjacência normalizada
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
 
print("\nA construir matriz de adjacência...")
adj_matrix = build_adj_matrix(train_pos, n_users, n_items)
 
train_pos_set = train_pos.groupby('user_idx')['item_idx'].apply(set).to_dict()
train_pairs   = list(zip(train_pos['user_idx'].values, train_pos['item_idx'].values))
 
def sample_negative(user_idx, train_pos_set, n_items):
    while True:
        neg = np.random.randint(0, n_items)
        if neg not in train_pos_set.get(user_idx, set()):
            return neg
 
# ──────────────────────────────────────────────
# 4. Inicializar modelo e otimizador
# ──────────────────────────────────────────────
model     = LightGCN(n_users, n_items, EMBEDDING_DIM, N_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
 
# ──────────────────────────────────────────────
# 5. Treino
# ──────────────────────────────────────────────
print(f"\nA treinar LightGCN...")
print(f"  Epochs: {N_EPOCHS} | Embedding: {EMBEDDING_DIM} | Layers: {N_LAYERS} | LR: {LR} | WD: {WEIGHT_DECAY}")
 
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
            sample_negative(p[0], train_pos_set, n_items) for p in batch
        ]).to(device)
 
        optimizer.zero_grad()
        user_emb, item_emb = model(adj_matrix)
        loss = model.bpr_loss(user_emb, item_emb, users, pos_items, neg_items, WEIGHT_DECAY)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
        n_batches  += 1
 
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}/{N_EPOCHS} | Loss: {total_loss/n_batches:.4f}")
 
# ──────────────────────────────────────────────
# 6. Avaliação
# ──────────────────────────────────────────────
print("\nA avaliar...")
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
    if len(relevant) == 0 or user not in user2idx:
        continue
 
    u_idx             = user2idx[user]
    test_item_indices = [item2idx[it] for it in test_items if it in item2idx]
    test_items_mapped = [it for it in test_items if it in item2idx]
    if len(test_item_indices) == 0:
        continue
 
    scores       = item_emb[test_item_indices].dot(user_emb[u_idx])
    ranked_idx   = np.argsort(-scores)
    ranked_items = [test_items_mapped[i] for i in ranked_idx[:TOP_K]]
 
    precision_list.append(evaluation.precision_curve(ranked_items, relevant)[:TOP_K])
    recall_list.append(evaluation.recall_curve(ranked_items, relevant)[:TOP_K])
    f1_list.append(evaluation.f1_curve(ranked_items, relevant)[:TOP_K])
    mrr_list.append(evaluation.mrr_curve(ranked_items, relevant)[:TOP_K])
 
    precision1_list.append(evaluation.precision_at_k(ranked_items, relevant, 1))
    recall10_list.append(evaluation.recall_at_k(ranked_items, relevant, 10))
    hit10_list.append(evaluation.hit_rate_at_k(ranked_items, relevant, 10))
    ndcg10_list.append(evaluation.ndcg_at_k(ranked_items, relevant, 10))
 
# ──────────────────────────────────────────────
# 7. Resultados
# ──────────────────────────────────────────────
cols = [f'top{i}' for i in range(1, TOP_K + 1)]
 
mean_prec = pd.DataFrame([np.mean(precision_list, axis=0)], columns=cols)
mean_rec  = pd.DataFrame([np.mean(recall_list,    axis=0)], columns=cols)
mean_f1   = pd.DataFrame([np.mean(f1_list,        axis=0)], columns=cols)
mean_mrr  = pd.DataFrame([np.mean(mrr_list,       axis=0)], columns=cols)
 
print("\n" + "="*50)
print("LightGCN — Resultados Finais")
print("="*50)
print("\nPrecision@k:") ; print(mean_prec.to_string(index=False))
print("\nRecall@k:")    ; print(mean_rec.to_string(index=False))
print("\nF1@k:")        ; print(mean_f1.to_string(index=False))
print("\nMRR@k:")       ; print(mean_mrr.to_string(index=False))
 
print("\n── Métricas fixas ──")
print(f"  Precision@1:  {round(np.mean(precision1_list), 4)}")
print(f"  Recall@10:    {round(np.mean(recall10_list), 4)}")
print(f"  HitRate@10:   {round(np.mean(hit10_list), 4)}")
print(f"  NDCG@10:      {round(np.mean(ndcg10_list), 4)}")
print(f"  N_users:      {len(precision1_list)}")
 
# Guardar resultados
os.makedirs("results", exist_ok=True)
mean_prec.insert(0, 'model', 'LightGCN')
mean_rec.insert(0,  'model', 'LightGCN')
mean_f1.insert(0,   'model', 'LightGCN')
mean_mrr.insert(0,  'model', 'LightGCN')
 
mean_prec.to_csv("results/precision_lightgcn.csv", index=False)
mean_rec.to_csv("results/recall_lightgcn.csv",     index=False)
mean_f1.to_csv("results/f1_lightgcn.csv",          index=False)
mean_mrr.to_csv("results/mrr_lightgcn.csv",        index=False)
 
print("\nResultados guardados em results/")
print("Done!")