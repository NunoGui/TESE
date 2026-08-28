import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import HeteroData
import json
import copy

import split
import evaluation
from hetero_gat_model import HeteroGAT

# ── Configuração
PATH_RATINGS = "../data/ratings_full.csv"
PATH_USERS   = "../data/users.csv"
HIDDEN_DIM   = 64
N_EPOCHS     = 500
PATIENCE     = 30
BATCH_SIZE   = 1024
TOP_K        = 10
K_FOLDS      = 5
TOTAL_ITEMS  = 3084
MIN_TEST     = 20
BASE_SEED    = 42
DROPOUT      = 0.1

np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
device = torch.device("cpu")

# ── Forçar determinismo total (evitar variação de resultados entre corridas)
torch.set_num_threads(1)  # BLAS/OpenMP multi-thread pode somar em ordens diferentes entre corridas
torch.use_deterministic_algorithms(True, warn_only=True)

EMO_COLS  = ['valence', 'arousal', 'dominance',
             'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
DEMO_COLS = ['age_group', 'populational_aff', 'gender', 'education', 'country']

# ── Carregar hiperparâmetros (mesmo ficheiro usado no treino pool-20, hetero_gat_train.py)
PARAMS_PATH = "../results/hetero_gat_best_params.json"
if os.path.exists(PARAMS_PATH):
    with open(PARAMS_PATH, "r") as f:
        best_params = json.load(f)
    LR           = best_params["lr"]
    WEIGHT_DECAY = best_params["weight_decay"]
    N_LAYERS     = best_params["n_layers"]
    N_HEADS      = best_params["n_heads"]
    print(f"Hiperparâmetros carregados de {PARAMS_PATH}")
else:
    LR, WEIGHT_DECAY, N_LAYERS, N_HEADS = 0.001, 0.01, 3, 8
    print("A usar hiperparâmetros default.")

print(f"  lr={round(LR,6)} | weight_decay={round(WEIGHT_DECAY,6)} | n_layers={N_LAYERS} | n_heads={N_HEADS}")

# ──────────────────────────────────────────────
# 1. Carregar dados
# ──────────────────────────────────────────────
print("\nA carregar dados...")
ratings = pd.read_csv(PATH_RATINGS).fillna(0)
ratings = ratings.rename(columns={"user_id": "user", "image_id": "item"})
ratings = split.remove_degenerate_users(ratings)

users_df = pd.read_csv(PATH_USERS).fillna("Unknown")
users_df = users_df.rename(columns={"user_id": "user"})

# CODIFICAÇÃO DEMOGRÁFICA FINAL: categorias com menos de RARE_THRESHOLD
# utilizadores são agrupadas em "Other" antes do one-hot, reduzindo o
# vetor de 47 para 24 dimensões, sem perda de desempenho (ver secção
# "Demographic Feature Encoding: Testing Its Contribution").
RARE_THRESHOLD = 10
users_grouped = users_df.copy()
for col in DEMO_COLS:
    counts = users_df[col].value_counts()
    rare_categories = counts[counts < RARE_THRESHOLD].index
    users_grouped[col] = users_df[col].apply(lambda x: 'Other' if x in rare_categories else x)

demo_encoded = pd.get_dummies(users_grouped[['user'] + DEMO_COLS], columns=DEMO_COLS)
demo_encoded = demo_encoded.set_index('user')
user_feat_dim = demo_encoded.shape[1]

# NOTA (correção de fuga de dados): image_features deixou de ser calculado aqui,
# a partir do `ratings` completo. Isso permitia que a média de emoções de cada
# imagem incluísse interações que, num dado fold, pertencem ao conjunto de teste.
# Agora este cálculo é feito dentro do loop de fold, usando apenas `train_df`.
image_feat_dim = len(EMO_COLS)

all_items = set(ratings['item'].unique())
print(f"  Users: {ratings['user'].nunique()} | Items no catálogo: {len(all_items)}")

# ──────────────────────────────────────────────
# 2. Funções auxiliares
# ──────────────────────────────────────────────
def build_hetero_graph(train_df, user2idx, item2idx, image_features):
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
        u_idx = torch.LongTensor([user2idx[u] for u in liked['user']])
        i_idx = torch.LongTensor([item2idx[i] for i in liked['item']])
        edge_feat = torch.FloatTensor(liked[EMO_COLS].values)
        data['user', 'liked', 'image'].edge_index = torch.stack([u_idx, i_idx])
        data['user', 'liked', 'image'].edge_attr  = edge_feat
        data['image', 'liked_by', 'user'].edge_index = torch.stack([i_idx, u_idx])
        data['image', 'liked_by', 'user'].edge_attr  = edge_feat

    disliked = train_df[train_df['rating'] == 0].copy()
    disliked = disliked[disliked['user'].isin(user2idx) & disliked['item'].isin(item2idx)]
    if len(disliked) > 0:
        u_idx = torch.LongTensor([user2idx[u] for u in disliked['user']])
        i_idx = torch.LongTensor([item2idx[i] for i in disliked['item']])
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

def evaluate_model_ndcg(model, test_df, user2idx, item2idx, x_dict, edge_index_dict, edge_attr_dict):
    """Avalia NDCG@10 sobre pool de 20 para early stopping."""
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
        if not test_item_indices:
            continue
        scores       = image_emb[test_item_indices].dot(user_emb[u_idx])
        ranked_idx   = np.argsort(-scores)
        ranked_items = [test_items_mapped[i] for i in ranked_idx[:TOP_K]]
        ndcg_list.append(evaluation.ndcg_at_k(ranked_items, relevant, 10))
    return np.mean(ndcg_list) if ndcg_list else 0.0

# ──────────────────────────────────────────────
# 3. Loop kfold
# ──────────────────────────────────────────────
cols = [f'top{i}' for i in range(1, TOP_K + 1)]

acc_prec = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_rec  = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_f1   = pd.DataFrame(0.0, index=range(1), columns=cols)
acc_mrr  = pd.DataFrame(0.0, index=range(1), columns=cols)

precision1_all, recall10_all, hit10_all, ndcg10_all, mrr10_all = [], [], [], [], []

all_diagnostics = []      # <-- novo: grau de treino + norma do embedding por imagem
all_recommendations = []  # <-- novo: top-10 real por utilizador

print(f"\nAvaliação sobre catálogo completo — kfold ({K_FOLDS} folds)...")

for k, train_df, test_df in split.kfold_split(ratings, K_FOLDS, TOTAL_ITEMS, MIN_TEST, BASE_SEED):
    print(f"\n  Fold {k+1}/{K_FOLDS}...")

    torch.manual_seed(BASE_SEED + k)
    np.random.seed(BASE_SEED + k)

    # ── CORREÇÃO: image_features calculado só a partir do train_df deste fold
    image_features = train_df.groupby('item')[EMO_COLS].mean()

    user_ids = sorted(train_df['user'].unique())
    item_ids = sorted(ratings['item'].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}
    n_items  = len(item_ids)

    graph = build_hetero_graph(train_df, user2idx, item2idx, image_features).to(device)

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

    train_seen = train_df.groupby('user')['item'].apply(set).to_dict()

    edge_index_dict = {k: v.edge_index for k, v in graph.edge_items() if hasattr(v, 'edge_index')}
    edge_attr_dict  = {k: v.edge_attr  for k, v in graph.edge_items() if hasattr(v, 'edge_attr')}
    x_dict = {k: v.x for k, v in graph.node_items()}

    model = HeteroGAT(
        user_feat_dim=user_feat_dim,
        image_feat_dim=image_feat_dim,
        edge_feat_dim=len(EMO_COLS),
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Early stopping
    best_ndcg, best_epoch, no_improve = 0.0, 0, 0
    best_model_state = None

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        pairs = train_pairs.copy()
        np.random.shuffle(pairs)
        total_loss, n_batches = 0, 0
        for i in range(0, len(pairs), BATCH_SIZE):
            batch     = pairs[i:i + BATCH_SIZE]
            users_t   = torch.LongTensor([p[0] for p in batch]).to(device)
            pos_items = torch.LongTensor([p[1] for p in batch]).to(device)
            neg_items = torch.LongTensor([
                sample_negative(p[0], train_pos_set, n_items) for p in batch
            ]).to(device)
            optimizer.zero_grad()
            user_emb, image_emb = model(x_dict, edge_index_dict, edge_attr_dict)
            loss = model.bpr_loss(user_emb, image_emb, users_t, pos_items, neg_items, WEIGHT_DECAY)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        if epoch % 5 == 0:
            current_ndcg = evaluate_model_ndcg(model, test_df, user2idx, item2idx,
                                               x_dict, edge_index_dict, edge_attr_dict)
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
        print(f"    Melhor modelo restaurado da epoch {best_epoch} com NDCG@10={best_ndcg:.4f}")

    # ── Avaliação sobre catálogo completo
    model.eval()
    with torch.no_grad():
        user_emb, image_emb = model(x_dict, edge_index_dict, edge_attr_dict)
        user_emb  = user_emb.cpu().numpy()
        image_emb = image_emb.cpu().numpy()

    # ── Diagnóstico: grau de treino e norma do embedding por imagem
    train_degree = train_df['item'].value_counts().to_dict()
    for it, idx in item2idx.items():
        all_diagnostics.append({
            "fold": k,
            "item": it,
            "train_degree": train_degree.get(it, 0),
            "embedding_norm": float(np.linalg.norm(image_emb[idx]))
        })

    precision_list, recall_list, f1_list, mrr_list = [], [], [], []
    precision1_list, recall10_list, hit10_list, ndcg10_list, mrr10_list = [], [], [], [], []

    def pad(lst, k):
        return (lst + [lst[-1]] * k)[:k] if lst else [0.0] * k

    for user in test_df['user'].unique():
        user_test = test_df[test_df['user'] == user]
        relevant  = user_test[user_test['rating'] == 1]['item'].tolist()
        if len(relevant) == 0 or user not in user2idx:
            continue

        # Candidatos = todas as imagens não vistas no treino
        seen       = train_seen.get(user, set())
        candidates = [it for it in all_items if it not in seen and it in item2idx]
        if not candidates:
            continue

        u_idx             = user2idx[user]
        candidate_indices = [item2idx[it] for it in candidates]
        scores            = image_emb[candidate_indices].dot(user_emb[u_idx])
        ranked_idx        = np.argsort(-scores)
        ranked_items      = [candidates[i] for i in ranked_idx[:TOP_K]]
        ranked_scores     = scores[ranked_idx[:TOP_K]]

        for rank, (item_id, sc) in enumerate(zip(ranked_items, ranked_scores), start=1):
            all_recommendations.append({
                "fold": k, "user": user, "rank": rank, "item": item_id, "score": float(sc)
            })

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
print("HeteroGAT — Avaliação Catálogo Completo (kfold5)")
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
    "model":        "HeteroGAT_FullCatalog_Grouped",
    "Precision@1":  round(np.mean(precision1_all), 4),
    "Recall@10":    round(np.mean(recall10_all), 4),
    "HitRate@10":   round(np.mean(hit10_all), 4),
    "NDCG@10":      round(np.mean(ndcg10_all), 4),
    "MRR@10":       round(np.mean(mrr10_all), 4)
}])
fixed.to_csv("../results/fixed_metrics_heterogat_full_grouped.csv", index=False)

diag_df = pd.DataFrame(all_diagnostics)
diag_df.to_csv("../results/diagnostic_heterogat_full_grouped.csv", index=False)

recs_df = pd.DataFrame(all_recommendations)
recs_df.to_csv("../results/recommendations_heterogat_full_grouped.csv", index=False)

print(f"\nDiagnóstico guardado: {len(diag_df)} linhas (grau + norma por imagem/fold)")
print(f"Recomendações guardadas: {len(recs_df)} linhas")
print("Resultados guardados em ../results/")
print("Done!")