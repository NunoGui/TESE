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
 
EMO_COLS = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
VAD_COLS = ['valence', 'arousal', 'dominance']
 
# Encoders
user_enc  = LabelEncoder()
image_enc = LabelEncoder()
ratings['user_idx']  = user_enc.fit_transform(ratings['user_id'])
ratings['image_idx'] = image_enc.fit_transform(ratings['image_id'])
 
n_users  = ratings['user_idx'].nunique()
n_images = ratings['image_idx'].nunique()
n_emo    = len(EMO_COLS)  # 7 nós de emoção
 
print(f"Utilizadores: {n_users} | Imagens: {n_images} | Emoções: {n_emo}")
 
# ── Split 12/3/17 por utilizador ───────────────────────────────────────────
print("\nA construir split 12/3 + 17 random...")
 
train_data = []
test_data  = []
all_images = set(ratings['image_idx'].unique())
 
for user_idx, grupo in ratings.groupby('user_idx'):
    imagens_user = grupo.sample(frac=1, random_state=SEED)
    treino = imagens_user.iloc[:12]
    teste  = imagens_user.iloc[12:15]
 
    vistas     = set(imagens_user['image_idx'].values)
    nao_vistas = list(all_images - vistas)
    random_17  = random.sample(nao_vistas, min(17, len(nao_vistas)))
 
    train_data.append(treino)
    for _, row in teste.iterrows():
        test_data.append({
            'user_idx':   user_idx,
            'pos_image':  int(row['image_idx']),
            'neg_images': random_17,
            'pos_rating': int(row['rating'])
        })
 
train_df = pd.concat(train_data).reset_index(drop=True)
print(f"Treino: {len(train_df)} | Teste: {len(test_data)}")
 
# ── Normalizar scores emocionais ───────────────────────────────────────────
scaler = MinMaxScaler()
train_df[EMO_COLS] = scaler.fit_transform(train_df[EMO_COLS])
 
# ── Offsets dos nós ────────────────────────────────────────────────────────
IMG_OFFSET = n_users
EMO_OFFSET = n_users + n_images
N_TOTAL    = n_users + n_images + n_emo
 
# ── Construir arestas RATED (User → Image) ─────────────────────────────────
u_rated = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
i_rated = torch.tensor(train_df['image_idx'].values + IMG_OFFSET, dtype=torch.long)
 
# ── Construir arestas FELT (User → Emotion) com image_id como contexto ─────
# Cada aresta representa: user X sentiu emoção Y com score Z ao ver imagem W
# O image_id fica como propriedade da aresta — preserva o contexto
print("\nA construir arestas FELT com contexto de imagem...")
 
felt_src  = []  # user_idx
felt_dst  = []  # emotion_node_idx
felt_wgt  = []  # score normalizado
felt_img  = []  # image_idx (contexto — não usado pelo GNN mas preservado)
 
for _, row in train_df.iterrows():
    u     = int(row['user_idx'])
    img   = int(row['image_idx'])
    for emo_idx, col in enumerate(EMO_COLS):
        score = float(row[col])
        if score > 0:
            felt_src.append(u)
            felt_dst.append(EMO_OFFSET + emo_idx)
            felt_wgt.append(score)
            felt_img.append(img)  # contexto preservado
 
felt_src = torch.tensor(felt_src, dtype=torch.long)
felt_dst = torch.tensor(felt_dst, dtype=torch.long)
felt_wgt = torch.tensor(felt_wgt, dtype=torch.float32)
# felt_img é preservado para rastreabilidade mas não entra no GNN
 
print(f"Arestas RATED: {len(u_rated)}")
print(f"Arestas FELT:  {len(felt_src)} (com image_id preservado em cada aresta)")
print(f"  → Exemplo: User {felt_src[0].item()} sentiu {EMO_COLS[felt_dst[0].item()-EMO_OFFSET]} "
      f"(score={felt_wgt[0]:.2f}) ao ver Imagem {felt_img[0]}")
 
# ── Construir matriz de adjacência normalizada ─────────────────────────────
def build_hetero_adj(u_rated, i_rated, felt_src, felt_dst, felt_wgt, N):
    # RATED — bidirecional
    row_r = torch.cat([u_rated, i_rated])
    col_r = torch.cat([i_rated, u_rated])
    val_r = torch.ones(len(row_r))
 
    # FELT — bidirecional com peso (score emocional)
    row_f = torch.cat([felt_src, felt_dst])
    col_f = torch.cat([felt_dst, felt_src])
    val_f = torch.cat([felt_wgt, felt_wgt])
 
    row = torch.cat([row_r, row_f])
    col = torch.cat([col_r, col_f])
    val = torch.cat([val_r, val_f])
 
    # Normalização
    deg = torch.zeros(N)
    deg.scatter_add_(0, row, torch.ones(len(row)))
    deg_inv = deg.pow(-0.5)
    deg_inv[deg_inv == float('inf')] = 0
    val_norm = deg_inv[row] * val * deg_inv[col]
 
    return torch.sparse_coo_tensor(torch.stack([row, col]), val_norm, (N, N))
 
print("\nA construir grafo heterogéneo...")
adj = build_hetero_adj(u_rated, i_rated, felt_src, felt_dst, felt_wgt, N_TOTAL)
print(f"Grafo: {N_TOTAL} nós ({n_users} users + {n_images} images + {n_emo} emotions)")
 
# ── Modelo HeteroLightGCN ──────────────────────────────────────────────────
class HeteroLightGCN(nn.Module):
    def __init__(self, n_total, emb_dim=64, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.emb = nn.Embedding(n_total, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.01)
 
    def forward(self, adj):
        x    = self.emb.weight
        embs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            embs.append(x)
        return torch.stack(embs, dim=1).mean(dim=1)
 
    def predict(self, all_embs, users, items):
        u = all_embs[users]
        i = all_embs[items]
        return (u * i).sum(dim=1)
 
# ── Treino BPR ─────────────────────────────────────────────────────────────
model     = HeteroLightGCN(N_TOTAL, emb_dim=64, n_layers=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_pos = train_df[train_df['rating'] == 1].reset_index(drop=True)
 
print("\nA treinar HeteroLightGCN com FELT contextualizado...")
EPOCHS     = 50
BATCH_SIZE = 512
 
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    idx = np.random.permutation(len(train_pos))
 
    for start in range(0, len(idx), BATCH_SIZE):
        batch = train_pos.iloc[idx[start:start+BATCH_SIZE]]
        u_idx = torch.tensor(batch['user_idx'].values, dtype=torch.long)
        i_idx = torch.tensor(batch['image_idx'].values + IMG_OFFSET, dtype=torch.long)
        neg   = torch.randint(IMG_OFFSET, IMG_OFFSET + n_images, (len(u_idx),))
 
        all_embs   = model(adj)
        pos_scores = model.predict(all_embs, u_idx, i_idx)
        neg_scores = model.predict(all_embs, u_idx, neg)
 
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
    all_embs = model(adj)
 
K_MAX          = 20
precision_at_k = np.zeros(K_MAX)
recall_at_k    = np.zeros(K_MAX)
n_test         = 0
 
for entrada in test_data:
    u          = entrada['user_idx']
    pos_img    = entrada['pos_image'] + IMG_OFFSET
    neg_imgs   = [x + IMG_OFFSET for x in entrada['neg_images']]
    pos_rating = entrada['pos_rating']
 
    if pos_rating != 1:
        continue
 
    candidatos = [pos_img] + neg_imgs
    c_tensor   = torch.tensor(candidatos, dtype=torch.long)
    u_tensor   = torch.tensor([u] * len(candidatos), dtype=torch.long)
 
    scores   = model.predict(all_embs, u_tensor, c_tensor)
    ranking  = torch.argsort(scores, descending=True).tolist()
    pos_rank = ranking.index(0)
 
    for k in range(1, K_MAX + 1):
        if pos_rank < k:
            precision_at_k[k-1] += 1 / k
            recall_at_k[k-1]    += 1
 
    n_test += 1
 
precision_at_k /= n_test
recall_at_k    /= n_test
 
print(f"\nAvaliado sobre {n_test} entradas de teste")
print(f"\nPrecision@1:  {precision_at_k[0]:.4f}")
print(f"Precision@5:  {precision_at_k[4]:.4f}")
print(f"Precision@10: {precision_at_k[9]:.4f}")
print(f"Recall@1:     {recall_at_k[0]:.4f}")
print(f"Recall@5:     {recall_at_k[4]:.4f}")
print(f"Recall@10:    {recall_at_k[9]:.4f}")
 
# ── Gráficos comparativos ──────────────────────────────────────────────────
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
fig.suptitle('LightGCN vs HeteroLightGCN (FELT com contexto de imagem) — EmoRecSys',
             fontsize=12, fontweight='bold')
 
ax1.plot(ks, lightgcn_base_precision, 'o--', color='#888888',
         linewidth=2, markersize=4, label='LightGCN (base)')
ax1.plot(ks, precision_at_k, 'o-', color='#2E4057',
         linewidth=2, markersize=5, label='HeteroLightGCN + Emoções (FELT)')
ax1.set_xlabel('Top@k')
ax1.set_ylabel('Precision')
ax1.set_title('Precision@k')
ax1.set_xticks(ks)
ax1.grid(True, alpha=0.3)
ax1.legend()
 
ax2.plot(ks, lightgcn_base_recall, 'o--', color='#888888',
         linewidth=2, markersize=4, label='LightGCN (base)')
ax2.plot(ks, recall_at_k, 'o-', color='#6B8F71',
         linewidth=2, markersize=5, label='HeteroLightGCN + Emoções (FELT)')
ax2.set_xlabel('Top@k')
ax2.set_ylabel('Recall')
ax2.set_title('Recall@k')
ax2.set_xticks(ks)
ax2.grid(True, alpha=0.3)
ax2.legend()
 
plt.tight_layout()
plt.savefig('grafico_hetero_felt_contextualizado.png', dpi=180, bbox_inches='tight')
plt.show()
print("\nGráfico guardado como 'grafico_hetero_felt_contextualizado.png'")