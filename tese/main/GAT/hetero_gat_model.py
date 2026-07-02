import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear
import sys
sys.path.append('..')

 
class HeteroGAT(nn.Module):
    """
    Heterogeneous Graph Attention Network para Recomendação Afetiva.
    
    Arquitetura:
    - Nós User: inicializados com features demográficas (47 features one-hot)
    - Nós Image: inicializados com emoções médias (10 features)
    - Arestas LIKED (rating=1): com 10 features emocionais da interação
    - Arestas DISLIKED (rating=0): com 10 features emocionais da interação
    - HGTConv: propaga informação distinguindo tipos de nós e arestas
    - Predição por produto interno entre embeddings de user e image
    """
 
    def __init__(
        self,
        user_feat_dim,   # 47 — features demográficas
        image_feat_dim,  # 10 — features emocionais das imagens
        edge_feat_dim,   # 10 — features emocionais das arestas
        hidden_dim=64,   # dimensão dos embeddings internos
        n_heads=4,       # número de cabeças de atenção
        n_layers=2,      # número de camadas HGT
        dropout=0.1
    ):
        super(HeteroGAT, self).__init__()
 
        self.n_layers  = n_layers
        self.hidden_dim = hidden_dim
 
        # ── Projeções iniciais dos nós para hidden_dim
        self.user_proj  = Linear(user_feat_dim,  hidden_dim)
        self.image_proj = Linear(image_feat_dim, hidden_dim)
 
        # ── Projeção das features das arestas
        self.edge_liked_proj    = Linear(edge_feat_dim, hidden_dim)
        self.edge_disliked_proj = Linear(edge_feat_dim, hidden_dim)
 
        # ── Metadados do grafo heterogéneo
        self.metadata = (
            ['user', 'image'],  # tipos de nós
            [
                ('user', 'liked',    'image'),
                ('user', 'disliked', 'image'),
                ('image', 'liked_by',    'user'),
                ('image', 'disliked_by', 'user'),
            ]  # tipos de arestas (bidirecionais)
        )
 
        # ── Camadas HGTConv
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=self.metadata,
                heads=n_heads,
            )
            for _ in range(n_layers)
        ])
 
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
 
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        x_dict: features dos nós {user: tensor(N_users, 47), image: tensor(N_images, 10)}
        edge_index_dict: índices das arestas por tipo
        edge_attr_dict: features das arestas por tipo
        """
 
        # ── 1. Projetar features dos nós para hidden_dim
        h = {
            'user':  self.activation(self.user_proj(x_dict['user'])),
            'image': self.activation(self.image_proj(x_dict['image'])),
        }
 
        # ── 2. Incorporar features das arestas nos nós de destino
        # Para cada tipo de aresta, agregamos as features da aresta
        # ao nó de destino antes da propagação HGT
        if ('user', 'liked', 'image') in edge_attr_dict:
            edge_feat_liked = self.activation(
                self.edge_liked_proj(edge_attr_dict[('user', 'liked', 'image')])
            )
            # Agregar features da aresta ao nó image via scatter mean
            liked_idx = edge_index_dict[('user', 'liked', 'image')][1]
            h['image'] = h['image'] + self._scatter_mean(
                edge_feat_liked, liked_idx, h['image'].size(0)
            )
 
        if ('user', 'disliked', 'image') in edge_attr_dict:
            edge_feat_disliked = self.activation(
                self.edge_disliked_proj(edge_attr_dict[('user', 'disliked', 'image')])
            )
            disliked_idx = edge_index_dict[('user', 'disliked', 'image')][1]
            h['image'] = h['image'] + self._scatter_mean(
                edge_feat_disliked, disliked_idx, h['image'].size(0)
            )
 
        # ── 3. Propagação HGT por n_layers camadas
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {k: self.activation(self.dropout(v)) for k, v in h.items()}
 
        return h['user'], h['image']
 
    def _scatter_mean(self, src, idx, n_nodes):
        """Agrega features das arestas por média para cada nó de destino."""
        out   = torch.zeros(n_nodes, src.size(1), device=src.device)
        count = torch.zeros(n_nodes, 1, device=src.device)
        out.scatter_add_(0, idx.unsqueeze(1).expand_as(src), src)
        count.scatter_add_(0, idx.unsqueeze(1), torch.ones(idx.size(0), 1, device=src.device))
        count = count.clamp(min=1)
        return out / count
 
    def predict(self, user_emb, image_emb, user_ids, image_ids):
        """Predição por produto interno."""
        u = user_emb[user_ids]
        i = image_emb[image_ids]
        return (u * i).sum(dim=1)
 
    def bpr_loss(self, user_emb, image_emb, users, pos_items, neg_items, weight_decay=1e-4):
        """
        BPR Loss — Bayesian Personalized Ranking.
        Treina o modelo para pontuar itens positivos acima dos negativos.
        """
        u_emb   = user_emb[users]
        pos_emb = image_emb[pos_items]
        neg_emb = image_emb[neg_items]
 
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
 
        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
 
        # Regularização L2
        reg = weight_decay * (
            u_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / len(users)
 
        return bpr + reg