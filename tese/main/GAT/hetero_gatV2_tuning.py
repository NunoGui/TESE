import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import HeteroData
import json
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import split
import evaluation
from hetero_gatV2_model import HeteroGATv2

# ── Configuração
PATH_RATINGS = "../data/ratings_full.csv"
PATH_USERS   = "../data/users.csv"
HIDDEN_DIM   = 64
N_EPOCHS     = 100
BATCH_SIZE   = 1024
K_FOLDS      = 5
TOTAL_ITEMS  = 3084
MIN_TEST     = 20
BASE_SEED    = 42
N_TRIALS     = 20
DROPOUT      = 0.1

np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
device = torch.device("cpu")

EMO_COLS  = ['valence', 'arousal', 'dominance',
             'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
DEMO_COLS = ['age_group', 'populational_aff', 'gender', 'education', 'country']

# ──────────────────────────────────────────────
# 1. Carregar dados
# ──────────────────────────────────────────────
print("A carregar dados...")
ratings = pd.read_csv(PATH_RATINGS).fillna(0)
ratings = ratings.rename(columns={"user_id": "user", "image_id": "item"})
ratings = split.remove_degenerate_users(ratings)

users_df = pd.read_csv(PATH_USERS).fillna("Unknown")
users_df = users_df.rename(columns={"user_id": "user"})

demo_encoded = pd.get_dummies(users_df[['user'] + DEMO_COLS], columns=DEMO_COLS)
demo_encoded = demo_encoded.set_index('user')
user_feat_dim = demo_encoded.shape[1]

image_features = ratings.groupby('item')[EMO_COLS].mean()
image_feat_dim = len(EMO_COLS)

print(f"  Users: {ratings['user'].nunique()} | Items: {ratings['item'].nunique()}")
print(f"  Features user: {user_feat_dim} | Features image: {image_feat_dim}")

# ──────────────────────────────────────────────
# 2. Construir HeteroData
# ──────────────────────────────────────────────
def build_hetero_graph(train_df, user2idx, item2idx):
    data = HeteroData()

    user_ids_sorted = sorted(user2idx.keys())
    user_feat_matrix = []
    for u in user_ids_sorted:
        if u in demo_encoded.index:
            user_feat_matrix.append(demo_encoded.loc[u].values.astype(float))
        else:
            user_feat_matrix.append(np.zeros(user_feat_dim))
    data['user'].x = torch.FloatTensor(np.array(user_feat_matrix))

    item_ids_sorted = sorted(item2idx.keys())
    image_feat_matrix = []
    for i in item_ids_sorted:
        if i in image_features.index:
            image_feat_matrix.append(image_features.loc[i].values.astype(float))
        else:
            image_feat_matrix.append(np.zeros(image_feat_dim))
    data['image'].x = torch.FloatTensor(np.array(image_feat_matrix))

    liked = train_df[train_df['rating'] == 1].copy()
    liked = liked[liked['user'].isin(user2idx) & liked['item'].isin(item2idx)]
    if len(liked) > 0:
        u_idx     = torch.LongTensor([user2idx[u] for u in liked['user']])
        i_idx     = torch.LongTensor([item2idx[i] for i in liked['item']])
        edge_feat = torch.FloatTensor(liked[EMO_COLS].values)
        data['user', 'liked', 'image'].edge_index = torch.stack([u_idx, i_idx])
        data['user', 'liked', 'image'].edge_attr  = edge_feat
        data['image', 'liked_by', 'user'].edge_index = torch.stack([i_idx, u_idx])
        data['image', 'liked_by', 'user'].edge_attr  = edge_feat

    disliked = train_df[train_df['rating'] == 0].copy()
    disliked = disliked[disliked['user'].isin(user2idx) & disliked['item'].isin(item2idx)]
    if len(disliked) > 0:
        u_idx     = torch.LongTensor([user2idx[u] for u in disliked['user']])
        i_idx     = torch.LongTensor([item2idx[i] for i in disliked['item']])
        edge_feat = torch.FloatTensor(disliked[EMO_COLS].values)
        data['user', 'disliked', 'image'].edge_index = torch.stack([u_idx, i_idx])
        data['user', 'disliked', 'image'].edge_attr  = edge_feat
        data['image', 'disliked_by', 'user'].edge_index = torch.stack([i_idx, u_idx])
        data['image', 'disliked_by', 'user'].edge_attr  = edge_feat

    return data

def sample_negative(user_idx, train_pos_set, n_items):
    while True:
        neg = np.random.randint(0, n_items)
        if neg not in train_pos_set.get(user_idx, set()):
            return neg

# ──────────────────────────────────────────────
# 3. Treino e avaliação com kfold5
# ──────────────────────────────────────────────
def train_and_evaluate_kfold(lr, weight_decay, n_layers, n_heads):
    ndcg_folds = []

    for k, train_df, test_df in split.kfold_split(ratings, K_FOLDS, TOTAL_ITEMS, MIN_TEST, BASE_SEED):
        torch.manual_seed(BASE_SEED + k)
        np.random.seed(BASE_SEED + k)

        user_ids = sorted(train_df['user'].unique())
        item_ids = sorted(ratings['item'].unique())
        user2idx = {u: i for i, u in enumerate(user_ids)}
        item2idx = {it: i for i, it in enumerate(item_ids)}
        n_items  = len(item_ids)

        graph = build_hetero_graph(train_df, user2idx, item2idx).to(device)

        train_pos = train_df[train_df['rating'] == 1]
        train_pos_set = {}
        for _, row in train_pos.iterrows():
            if row['user'] in user2idx and row['item'] in item2idx:
                u = user2idx[row['user']]
                i = item2idx[row['item']]
                train_pos_set.setdefault(u, set()).add(i)

        train_pairs = [
            (user2idx[row['user']], item2idx[row['item']])
            for _, row in train_pos.iterrows()
            if row['user'] in user2idx and row['item'] in item2idx
        ]

        edge_index_dict = {k: v.edge_index for k, v in graph.edge_items() if hasattr(v, 'edge_index')}
        edge_attr_dict  = {k: v.edge_attr  for k, v in graph.edge_items() if hasattr(v, 'edge_attr')}
        x_dict = {k: v.x for k, v in graph.node_items()}

        model = HeteroGATv2(
            user_feat_dim=user_feat_dim,
            image_feat_dim=image_feat_dim,
            edge_feat_dim=len(EMO_COLS),
            hidden_dim=HIDDEN_DIM,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=DROPOUT
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            pairs = train_pairs.copy()
            np.random.shuffle(pairs)
            for i in range(0, len(pairs), BATCH_SIZE):
                batch     = pairs[i:i + BATCH_SIZE]
                users_t   = torch.LongTensor([p[0] for p in batch]).to(device)
                pos_items = torch.LongTensor([p[1] for p in batch]).to(device)
                neg_items = torch.LongTensor([
                    sample_negative(p[0], train_pos_set, n_items) for p in batch
                ]).to(device)
                optimizer.zero_grad()
                user_emb, image_emb = model(x_dict, edge_index_dict, edge_attr_dict)
                loss = model.bpr_loss(user_emb, image_emb, users_t, pos_items, neg_items, weight_decay)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            user_emb, image_emb = model(x_dict, edge_index_dict, edge_attr_dict)
            user_emb  = user_emb.cpu().numpy()
            image_emb = image_emb.cpu().numpy()

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
            scores       = image_emb[test_item_indices].dot(user_emb[u_idx])
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
    lr           = trial.suggest_float("lr", 1e-3, 5e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    n_layers     = trial.suggest_int("n_layers", 1, 3)
    n_heads      = trial.suggest_categorical("n_heads", [2, 4, 8])

    ndcg = train_and_evaluate_kfold(lr, weight_decay, n_layers, n_heads)

    trial_results.append({
        "trial":        trial.number + 1,
        "lr":           round(lr, 6),
        "weight_decay": round(weight_decay, 6),
        "n_layers":     n_layers,
        "n_heads":      n_heads,
        "ndcg@10":      ndcg
    })

    print(f"Trial {trial.number+1:2d}/{N_TRIALS} | lr={round(lr,5)} | wd={round(weight_decay,5)} | layers={n_layers} | heads={n_heads} | NDCG@10={ndcg}")
    return ndcg

print(f"\nA iniciar Optuna TPE Search ({N_TRIALS} trials x {K_FOLDS} folds)...")
print("="*70)

sampler = optuna.samplers.TPESampler(seed=BASE_SEED)
study   = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)

# ──────────────────────────────────────────────
# 5. Resultados
# ──────────────────────────────────────────────
best      = study.best_params
best_ndcg = study.best_value

print("\n" + "="*70)
print("RESULTADOS DO TUNING (kfold5)")
print("="*70)

results_df = pd.DataFrame(trial_results).sort_values("ndcg@10", ascending=False)
print(results_df.to_string(index=False))

print(f"\n── Melhores hiperparâmetros (Optuna TPE + kfold5) ──")
print(f"  lr:           {best['lr']:.6f}")
print(f"  weight_decay: {best['weight_decay']:.6f}")
print(f"  n_layers:     {best['n_layers']}")
print(f"  n_heads:      {best['n_heads']}")
print(f"  NDCG@10:      {best_ndcg}")

os.makedirs("../results", exist_ok=True)
results_df.to_csv("../results/heterogatv2_tuning.csv", index=False)
with open("../results/hetero_gatv2_best_params.json", "w") as f:
    json.dump(best, f, indent=2)

print("\nResultados guardados em results/")
print("Done!")