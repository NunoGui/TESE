import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random
 
# ── Reprodutibilidade ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
 
# ── Carregar dados ─────────────────────────────────────────────────────────
print("A carregar dados...")
ratings = pd.read_csv('ratings_full.csv')
 
# Encoders para user e image IDs
user_enc  = LabelEncoder()
image_enc = LabelEncoder()
ratings['user_idx']  = user_enc.fit_transform(ratings['user_id'])
ratings['image_idx'] = image_enc.fit_transform(ratings['image_id'])
 
n_users  = ratings['user_idx'].nunique()
n_images = ratings['image_idx'].nunique()
print(f"Utilizadores: {n_users} | Imagens: {n_images} | Ratings: {len(ratings)}")
 
# ── Split 12/3/17 por utilizador ───────────────────────────────────────────
print("\nA construir split 12/3 + 17 random...")
 
train_data = []
test_data  = []
 
all_images = set(ratings['image_idx'].unique())
 
for user_idx, grupo in ratings.groupby('user_idx'):
    imagens_user = grupo.sample(frac=1, random_state=SEED)  # shuffle
 
    treino = imagens_user.iloc[:12]
    teste  = imagens_user.iloc[12:15]
 
    # 17 imagens random que o user nunca viu
    vistas = set(imagens_user['image_idx'].values)
    nao_vistas = list(all_images - vistas)
    random_17  = random.sample(nao_vistas, min(17, len(nao_vistas)))
 
    train_data.append(treino)
 
    for _, row in teste.iterrows():
        test_data.append({
            'user_idx':     user_idx,
            'pos_image':    int(row['image_idx']),
            'neg_images':   random_17,
            'pos_rating':   int(row['rating'])
        })
 
train_df = pd.concat(train_data).reset_index(drop=True)
print(f"Treino: {len(train_df)} interações")
print(f"Teste:  {len(test_data)} entradas (3 por utilizador)")
 
# ── Modelo LightGCN ────────────────────────────────────────────────────────
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3):
        super().__init__()
        self.n_users  = n_users
        self.n_items  = n_items
        self.n_layers = n_layers
 
        self.user_emb  = nn.Embedding(n_users, emb_dim)
        self.item_emb  = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
 
    def forward(self, adj):
        # Concatenar embeddings
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            embs.append(x)
        final = torch.stack(embs, dim=1).mean(dim=1)
        return final[:self.n_users], final[self.n_users:]
 
    def predict(self, user_embs, item_embs, users, items):
        u = user_embs[users]
        i = item_embs[items]
        return (u * i).sum(dim=1)
 
# ── Construir matriz de adjacência normalizada ─────────────────────────────
def build_adj(train_df, n_users, n_images):
    users  = torch.tensor(train_df['user_idx'].values,  dtype=torch.long)
    images = torch.tensor(train_df['image_idx'].values, dtype=torch.long) + n_users
 
    # Grafo bipartido simétrico
    row = torch.cat([users, images])
    col = torch.cat([images, users])
    N   = n_users + n_images
 
    # Normalização
    deg = torch.zeros(N)
    deg.scatter_add_(0, row, torch.ones(len(row)))
    deg_inv = deg.pow(-0.5)
    deg_inv[deg_inv == float('inf')] = 0
 
    vals = deg_inv[row] * deg_inv[col]
    adj  = torch.sparse_coo_tensor(torch.stack([row, col]), vals, (N, N))
    return adj
 
print("\nA construir grafo...")
adj = build_adj(train_df, n_users, n_images)
 
# ── Treino BPR ─────────────────────────────────────────────────────────────
model     = LightGCN(n_users, n_images, emb_dim=64, n_layers=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# Preparar pares positivos de treino (apenas rating=1)
train_pos = train_df[train_df['rating'] == 1].reset_index(drop=True)
 
print("A treinar LightGCN...")
EPOCHS     = 50
BATCH_SIZE = 512
 
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    idx        = np.random.permutation(len(train_pos))
 
    for start in range(0, len(idx), BATCH_SIZE):
        batch = train_pos.iloc[idx[start:start+BATCH_SIZE]]
        u_idx = torch.tensor(batch['user_idx'].values,  dtype=torch.long)
        i_idx = torch.tensor(batch['image_idx'].values, dtype=torch.long)
 
        # Negative sampling
        neg = torch.randint(0, n_images, (len(u_idx),))
 
        user_embs, item_embs = model(adj)
        pos_scores = model.predict(user_embs, item_embs, u_idx, i_idx)
        neg_scores = model.predict(user_embs, item_embs, u_idx, neg)
 
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d} | Loss: {total_loss:.4f}")
 
# ── Avaliação Precision@k e Recall@k ──────────────────────────────────────
print("\nA avaliar...")
model.eval()
with torch.no_grad():
    user_embs, item_embs = model(adj)
 
K_MAX = 20
precision_at_k = np.zeros(K_MAX)
recall_at_k    = np.zeros(K_MAX)
n_test         = 0
 
for entrada in test_data:
    u         = entrada['user_idx']
    pos_img   = entrada['pos_image']
    neg_imgs  = entrada['neg_images']
    pos_rating = entrada['pos_rating']
 
    # Só avaliamos quando a imagem de teste foi avaliada positivamente
    if pos_rating != 1:
        continue
 
    candidatos = [pos_img] + neg_imgs  # 1 positivo + 17 negativos = 18
    c_tensor   = torch.tensor(candidatos, dtype=torch.long)
    u_tensor   = torch.tensor([u] * len(candidatos), dtype=torch.long)
 
    scores  = model.predict(user_embs, item_embs, u_tensor, c_tensor)
    ranking = torch.argsort(scores, descending=True).tolist()
    pos_rank = ranking.index(0)  # posição da imagem positiva (índice 0 na lista)
 
    for k in range(1, K_MAX + 1):
        if pos_rank < k:
            precision_at_k[k-1] += 1 / k
            recall_at_k[k-1]    += 1
 
    n_test += 1
 
precision_at_k /= n_test
recall_at_k    /= n_test
 
print(f"\nAvaliado sobre {n_test} entradas de teste positivas")
print(f"\nPrecision@1:  {precision_at_k[0]:.4f}")
print(f"Precision@5:  {precision_at_k[4]:.4f}")
print(f"Precision@10: {precision_at_k[9]:.4f}")
print(f"Recall@1:     {recall_at_k[0]:.4f}")
print(f"Recall@5:     {recall_at_k[4]:.4f}")
print(f"Recall@10:    {recall_at_k[9]:.4f}")
 
# ── Gráficos ───────────────────────────────────────────────────────────────
ks = list(range(1, K_MAX + 1))
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('LightGCN — EmoRecSys (protocolo 12/3/17)', fontsize=13, fontweight='bold')
 
ax1.plot(ks, precision_at_k, 'o-', color='#2E4057', linewidth=2, markersize=5, label='LightGCN')
ax1.set_xlabel('Top@k')
ax1.set_ylabel('Precision')
ax1.set_title('Precision@k')
ax1.set_xticks(ks)
ax1.grid(True, alpha=0.3)
ax1.legend()
 
ax2.plot(ks, recall_at_k, 'o-', color='#6B8F71', linewidth=2, markersize=5, label='LightGCN')
ax2.set_xlabel('Top@k')
ax2.set_ylabel('Recall')
ax2.set_title('Recall@k')
ax2.set_xticks(ks)
ax2.grid(True, alpha=0.3)
ax2.legend()
 
plt.tight_layout()
plt.savefig('grafico_lightgcn_protocolo.png', dpi=180, bbox_inches='tight')
plt.show()
print("\nGráfico guardado como 'grafico_lightgcn_protocolo.png'")