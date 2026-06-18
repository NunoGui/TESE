import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ndcg_score

#input
ratings = pd.read_csv('ratings_full.csv')
positivo = ratings[ratings['rating'] == 1][['user_id', 'image_id']].copy()

# reindexar utilizadores e imagens para índices contínuos 0, 1, 2, ...
users = positivo['user_id'].unique()
images = positivo['image_id'].unique()

user2idx = {u: i for i, u in enumerate(users)}
img2idx  = {img: i for i, img in enumerate(images)}

positivo['user_idx'] = positivo['user_id'].map(user2idx)
positivo['img_idx']  = positivo['image_id'].map(img2idx)

n_users  = len(users)
n_images = len(images)
N_TOTAL  = n_users + n_images  # índices das imagens começam a partir de n_users

print(f"Utilizadores: {n_users}")
print(f"Imagens: {n_images}")
print(f"Total de nós no grafo: {N_TOTAL}")
print(f"Interações positivas: {len(positivo)}")

# construir o grafo bipartido , Grafo A

# índices dos utilizadores e imagens no grafo unificado
u_idx = torch.tensor(positivo['user_idx'].values, dtype=torch.long)
i_idx = torch.tensor(positivo['img_idx'].values + n_users, dtype=torch.long)

# User-Image e Image-User
row = torch.cat([u_idx, i_idx])
col = torch.cat([i_idx, u_idx])

# normalização simétrica — evita que nós com muitas ligações dominem
deg = torch.zeros(N_TOTAL)
deg.scatter_add_(0, row, torch.ones(len(row)))
deg_inv = deg.pow(-0.5)
deg_inv[deg_inv == float('inf')] = 0
val = deg_inv[row] * deg_inv[col]

# matriz de adjacência esparsa
adj = torch.sparse_coo_tensor(
    torch.stack([row, col]),
    val,
    (N_TOTAL, N_TOTAL)
)

print(f"Grafo construído: {N_TOTAL} nós, {len(row)} arestas (bidirecionais)")

# Definir o modelo LightGCN 

class LightGCN(nn.Module):
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
        # representação final — média de todas as camadas
        return torch.stack(embs, dim=1).mean(dim=1)

    def predict(self, all_embs, users, items):
        u = all_embs[users]
        i = all_embs[items]
        return (u * i).sum(dim=1)

model     = LightGCN(N_TOTAL, emb_dim=64, n_layers=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Modelo criado — parâmetros: {sum(p.numel() for p in model.parameters())}")

# preparar positivos e negativos reais 

# carregar todas as interações — positivas e negativas
todas = ratings[['user_id', 'image_id', 'rating']].copy()

# mapear para índices contínuos — só para utilizadores e imagens que existem
todas = todas[todas['user_id'].isin(user2idx) & todas['image_id'].isin(img2idx)]
todas['user_idx'] = todas['user_id'].map(user2idx)
todas['img_idx']  = todas['image_id'].map(img2idx)

# separar positivos e negativos confirmados
train_pos = todas[todas['rating'] == 1][['user_idx', 'img_idx']].copy()
train_neg = todas[todas['rating'] == 0][['user_idx', 'img_idx']].copy()

print(f"Interações positivas: {len(train_pos)}")
print(f"Interações negativas confirmadas: {len(train_neg)}")
print(f"Rácio positivo/negativo: {len(train_pos)/len(train_neg):.2f}")

# construir dicionário de negativos confirmados por utilizador
user_neg_dict = train_neg.groupby('user_idx')['img_idx'].apply(list).to_dict()

# treino com BPR Loss usando negativos reais

def bpr_loss(model, adj, users, pos_items, neg_items):
    all_embs = model(adj)
    u   = all_embs[users]
    pos = all_embs[pos_items + n_users]
    neg = all_embs[neg_items + n_users]
    pos_score = (u * pos).sum(dim=1)
    neg_score = (u * neg).sum(dim=1)
    loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
    return loss

EPOCHS = 200
BATCH_SIZE = 512

print("\nA treinar LightGCN com negativos reais...")
for epoch in range(EPOCHS):
    model.train()

    # amostrar batch de positivos
    idx = np.random.choice(len(train_pos), BATCH_SIZE, replace=True)
    batch = train_pos.iloc[idx]
    users_batch     = torch.tensor(batch['user_idx'].values, dtype=torch.long)
    pos_items_batch = torch.tensor(batch['img_idx'].values, dtype=torch.long)

    # para cada positivo, amostrar um negativo confirmado do mesmo utilizador
    neg_items = []
    for u in batch['user_idx'].values:
        negs = user_neg_dict.get(u, [])
        if len(negs) > 0:
            neg_items.append(np.random.choice(negs))
        else:
            # se não houver negativos confirmados, amostrar aleatoriamente
            neg_items.append(np.random.randint(0, n_images))
    neg_items_batch = torch.tensor(neg_items, dtype=torch.long)

    optimizer.zero_grad()
    loss = bpr_loss(model, adj, users_batch, pos_items_batch, neg_items_batch)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/200 — Loss: {loss.item():.4f}")

print("Treino concluído.")

print(f"Utilizadores com pelo menos 1 negativo: {len(user_neg_dict)}")
print(f"Média de negativos por utilizador: {len(train_neg)/len(user_neg_dict):.1f}")
print(f"Utilizadores sem nenhum negativo: {n_users - train_neg['user_idx'].nunique()}")

##################
# avaliar LightGCN 

def avaliar_lightgcn(model, adj, positivo, n_users, n_images, K=10, test_ratio=0.2):
    model.eval()
    with torch.no_grad():
        all_embs = model(adj)

    precisoes = []
    recalls   = []

    for user_idx, grupo in positivo.groupby('user_idx'):
        positivas = list(grupo['img_idx'].values)

        if len(positivas) < 2:
            continue

        # divisão treino/teste
        n_teste = max(1, int(len(positivas) * test_ratio))
        teste   = set(positivas[:n_teste]) # 20%
        treino  = set(positivas[n_teste:]) # 80%

        # candidatas — todas as imagens exceto as de treino
        candidatas = [i for i in range(n_images) if i not in treino]

        # calcular scores para todas as candidatas
        u_emb  = all_embs[user_idx]
        i_embs = all_embs[torch.tensor(candidatas) + n_users]
        scores = (u_emb * i_embs).sum(dim=1)

        # top-K candidatas
        topk_idx     = scores.topk(K).indices.tolist()
        recomendadas = set([candidatas[i] for i in topk_idx])

        # métricas
        acertos  = len(recomendadas & teste)
        precisao = acertos / K
        recall   = acertos / len(teste) if len(teste) > 0 else 0

        precisoes.append(precisao)
        recalls.append(recall)

    print(f"\nAvaliação LightGCN (K={K})")
    print(f"Utilizadores avaliados: {len(precisoes)}")
    print(f"Precision@{K}: {np.mean(precisoes):.3f}")
    print(f"Recall@{K}:    {np.mean(recalls):.3f}")

avaliar_lightgcn(model, adj, positivo, n_users, n_images, K=10)