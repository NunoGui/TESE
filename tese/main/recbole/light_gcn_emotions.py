import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
 
# Colunas emocionais
EMO_COLS = ['valence', 'arousal', 'dominance',
            'happiness', 'sadness', 'anger',
            'fear', 'surprise', 'disgust', 'neutral']
 
# Normalizar features emocionais para [0, 1]
scaler = MinMaxScaler()
ratings[EMO_COLS] = scaler.fit_transform(ratings[EMO_COLS])
 
# Encoders
user_enc  = LabelEncoder()
image_enc = LabelEncoder()
ratings['user_idx']  = user_enc.fit_transform(ratings['user_id'])
ratings['image_idx'] = image_enc.fit_transform(ratings['image_id'])
 
n_users  = ratings['user_idx'].nunique()
n_images = ratings['image_idx'].nunique()
n_emo    = len(EMO_COLS)
print(f"Utilizadores: {n_users} | Imagens: {n_images} | Ratings: {len(ratings)}")
 
# ── Split 12/3/17 por utilizador ───────────────────────────────────────────
print("\nA construir split 12/3 + 17 random...")
 
train_data = []
test_data  = []
all_images = set(ratings['image_idx'].unique())
 
for user_idx, grupo in ratings.groupby('user_idx'):
    imagens_user = grupo.sample(frac=1, random_state=SEED)
    treino = imagens_user.iloc[:12]
    teste  = imagens_user.iloc[12:15]
 
    vistas    = set(imagens_user['image_idx'].values)
    nao_vistas = list(all_images - vistas)
    random_17  = random.sample(nao_vistas, min(17, len(nao_vistas)))
 
    train_data.append(treino)
 
    for _, row in teste.iterrows():
        test_data.append({
            'user_idx':    user_idx,
            'pos_image':   int(row['image_idx']),
            'neg_images':  random_17,
            'pos_rating':  int(row['rating']),
            'emo_features': torch.tensor(row[EMO_COLS].values.astype(float),
                                         dtype=torch.float32)
        })
 
train_df = pd.concat(train_data).reset_index(drop=True)
print(f"Treino: {len(train_df)} interações")
print(f"Teste:  {len(test_data)} entradas")
 
# ── Perfis emocionais por utilizador e imagem ──────────────────────────────
# Agrega os scores emocionais médios por user e por image a partir do treino
user_emo_profile  = train_df.groupby('user_idx')[EMO_COLS].mean()
image_emo_profile = train_df.groupby('image_idx')[EMO_COLS].mean()
 
# Preenche users/images sem dados com 0
user_emo_matrix  = np.zeros((n_users,  n_emo))
image_emo_matrix = np.zeros((n_images, n_emo))
 
for idx, row in user_emo_profile.iterrows():
    user_emo_matrix[idx] = row.values
for idx, row in image_emo_profile.iterrows():
    image_emo_matrix[idx] = row.values
 
user_emo_tensor  = torch.tensor(user_emo_matrix,  dtype=torch.float32)
image_emo_tensor = torch.tensor(image_emo_matrix, dtype=torch.float32)
 
# ── Modelo LightGCN com emoções ────────────────────────────────────────────
class LightGCNEmo(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3, n_emo=10):
        super().__init__()
        self.n_users  = n_users
        self.n_items  = n_items
        self.n_layers = n_layers
 
        # Embeddings colaborativos
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
 
        # Projeção das features emocionais para o espaço de embeddings
        self.user_emo_proj = nn.Linear(n_emo, emb_dim)
        self.item_emo_proj = nn.Linear(n_emo, emb_dim)
 
        # Combinação colaborativo + emocional
        self.fusion = nn.Linear(emb_dim * 2, emb_dim)
 
    def get_embeddings(self, adj, user_emo, item_emo):
        # Embeddings colaborativos via LightGCN
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            embs.append(x)
        final = torch.stack(embs, dim=1).mean(dim=1)
        u_collab = final[:self.n_users]
        i_collab = final[self.n_users:]
 
        # Embeddings emocionais
        u_emo = torch.relu(self.user_emo_proj(user_emo))
        i_emo = torch.relu(self.item_emo_proj(item_emo))
 
        # Fusão colaborativo + emocional
        u_final = torch.relu(self.fusion(torch.cat([u_collab, u_emo], dim=1)))
        i_final = torch.relu(self.fusion(torch.cat([i_collab, i_emo], dim=1)))
 
        return u_final, i_final
 
    def predict(self, user_embs, item_embs, users, items):
        u = user_embs[users]
        i = item_embs[items]
        return (u * i).sum(dim=1)
 
# ── Construir matriz de adjacência ─────────────────────────────────────────
def build_adj(train_df, n_users, n_images):
    users  = torch.tensor(train_df['user_idx'].values,  dtype=torch.long)
    images = torch.tensor(train_df['image_idx'].values, dtype=torch.long) + n_users
    row = torch.cat([users, images])
    col = torch.cat([images, users])
    N   = n_users + n_images
    deg = torch.zeros(N)
    deg.scatter_add_(0, row, torch.ones(len(row)))
    deg_inv = deg.pow(-0.5)
    deg_inv[deg_inv == float('inf')] = 0
    vals = deg_inv[row] * deg_inv[col]
    return torch.sparse_coo_tensor(torch.stack([row, col]), vals, (N, N))
 
print("\nA construir grafo...")
adj = build_adj(train_df, n_users, n_images)
 
# ── Treino ─────────────────────────────────────────────────────────────────
model     = LightGCNEmo(n_users, n_images, emb_dim=64, n_layers=3, n_emo=n_emo)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_pos = train_df[train_df['rating'] == 1].reset_index(drop=True)
 
print("A treinar LightGCN + Emoções...")
EPOCHS     = 50
BATCH_SIZE = 512
 
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    idx = np.random.permutation(len(train_pos))
 
    for start in range(0, len(idx), BATCH_SIZE):
        batch = train_pos.iloc[idx[start:start+BATCH_SIZE]]
        u_idx = torch.tensor(batch['user_idx'].values,  dtype=torch.long)
        i_idx = torch.tensor(batch['image_idx'].values, dtype=torch.long)
        neg   = torch.randint(0, n_images, (len(u_idx),))
 
        user_embs, item_embs = model.get_embeddings(adj, user_emo_tensor, image_emo_tensor)
        pos_scores = model.predict(user_embs, item_embs, u_idx, i_idx)
        neg_scores = model.predict(user_embs, item_embs, u_idx, neg)
 
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d} | Loss: {total_loss:.4f}")
 
# ── Avaliação ──────────────────────────────────────────────────────────────
print("\nA avaliar...")
model.eval()
with torch.no_grad():
    user_embs, item_embs = model.get_embeddings(adj, user_emo_tensor, image_emo_tensor)
 
K_MAX = 20
precision_at_k = np.zeros(K_MAX)
recall_at_k    = np.zeros(K_MAX)
n_test         = 0
 
for entrada in test_data:
    u          = entrada['user_idx']
    pos_img    = entrada['pos_image']
    neg_imgs   = entrada['neg_images']
    pos_rating = entrada['pos_rating']
 
    if pos_rating != 1:
        continue
 
    candidatos = [pos_img] + neg_imgs
    c_tensor   = torch.tensor(candidatos, dtype=torch.long)
    u_tensor   = torch.tensor([u] * len(candidatos), dtype=torch.long)
 
    scores   = model.predict(user_embs, item_embs, u_tensor, c_tensor)
    ranking  = torch.argsort(scores, descending=True).tolist()
    pos_rank = ranking.index(0)
 
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
 
# ── Gráficos comparativos ──────────────────────────────────────────────────
# Resultados do LightGCN base (sem emoções) para comparação
lightgcn_base_precision = [0.8613, 0.4647, 0.3102, 0.2352, 0.1924,
                            0.1624, 0.1399, 0.1232, 0.1102, 0.1000,
                            0.0916, 0.0844, 0.0783, 0.0732, 0.0688,
                            0.0649, 0.0614, 0.0584, 0.0556, 0.0532]
lightgcn_base_recall    = [0.8613, 0.9227, 0.9315, 0.9398, 0.9479,
                            0.9522, 0.9559, 0.9591, 0.9619, 0.9643,
                            0.9663, 0.9681, 0.9697, 0.9711, 0.9724,
                            0.9736, 0.9747, 0.9757, 0.9766, 0.9775]
 
ks = list(range(1, K_MAX + 1))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('LightGCN vs LightGCN + Emoções — EmoRecSys (protocolo 12/3/17)',
             fontsize=13, fontweight='bold')
 
ax1.plot(ks, lightgcn_base_precision, 'o--', color='#888888',
         linewidth=2, markersize=4, label='LightGCN (base)')
ax1.plot(ks, precision_at_k, 'o-', color='#2E4057',
         linewidth=2, markersize=5, label='LightGCN + Emoções')
ax1.set_xlabel('Top@k')
ax1.set_ylabel('Precision')
ax1.set_title('Precision@k')
ax1.set_xticks(ks)
ax1.grid(True, alpha=0.3)
ax1.legend()
 
ax2.plot(ks, lightgcn_base_recall, 'o--', color='#888888',
         linewidth=2, markersize=4, label='LightGCN (base)')
ax2.plot(ks, recall_at_k, 'o-', color='#6B8F71',
         linewidth=2, markersize=5, label='LightGCN + Emoções')
ax2.set_xlabel('Top@k')
ax2.set_ylabel('Recall')
ax2.set_title('Recall@k')
ax2.set_xticks(ks)
ax2.grid(True, alpha=0.3)
ax2.legend()
 
plt.tight_layout()
plt.savefig('grafico_lightgcn_emocoes.png', dpi=180, bbox_inches='tight')
plt.show()
print("\nGráfico guardado como 'grafico_lightgcn_emocoes.png'")