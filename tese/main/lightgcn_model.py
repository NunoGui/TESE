import torch
import torch.nn as nn
 
 
class LightGCN(nn.Module):
    """
    LightGCN — Light Graph Convolution Network para Recomendação.
    Paper: He et al., 2020 (SIGIR)
 
    Arquitetura:
    - Embeddings iniciais para users e items
    - K camadas de propagação por vizinhança (sem transformação, sem ativação)
    - Embedding final = média dos embeddings de todas as camadas
    - Predição por produto interno
    """
 
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
 
        self.n_users       = n_users
        self.n_items       = n_items
        self.embedding_dim = embedding_dim
        self.n_layers      = n_layers
 
        # Embeddings iniciais (aprendidos durante o treino)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
 
        # Inicialização Xavier — mais estável que random
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
 
    def forward(self, adj_matrix):
        """
        Propagação LightGCN.
 
        adj_matrix: matriz de adjacência normalizada do grafo bipartido
                    shape: (n_users + n_items, n_users + n_items)
 
        Retorna embeddings finais de users e items.
        """
        # Concatenar embeddings iniciais
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)  # (n_users + n_items, embedding_dim)
 
        # Guardar embeddings de cada camada
        layer_embeddings = [all_embeddings]
 
        # Propagação por K camadas
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            layer_embeddings.append(all_embeddings)
 
        # Embedding final = média de todas as camadas (incluindo camada 0)
        final_embeddings = torch.stack(layer_embeddings, dim=1).mean(dim=1)
 
        # Separar users e items
        user_embeddings = final_embeddings[:self.n_users]
        item_embeddings = final_embeddings[self.n_users:]
 
        return user_embeddings, item_embeddings
 
    def predict(self, user_embeddings, item_embeddings, user_ids, item_ids):
        """
        Predição por produto interno entre user e item.
        """
        u_emb = user_embeddings[user_ids]
        i_emb = item_embeddings[item_ids]
        return (u_emb * i_emb).sum(dim=1)
 
    def bpr_loss(self, user_embeddings, item_embeddings,
                 users, pos_items, neg_items, weight_decay=1e-4):
        """
        BPR Loss — Bayesian Personalized Ranking.
        Treina o modelo para pontuar itens positivos acima dos negativos.
        """
        u_emb   = user_embeddings[users]
        pos_emb = item_embeddings[pos_items]
        neg_emb = item_embeddings[neg_items]
 
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
 
        # BPR loss
        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
 
        # Regularização L2 nos embeddings iniciais (não nos propagados)
        reg = weight_decay * (
            self.user_embedding.weight[users].norm(2).pow(2) +
            self.item_embedding.weight[pos_items].norm(2).pow(2) +
            self.item_embedding.weight[neg_items].norm(2).pow(2)
        ) / len(users)
 
        return bpr + reg