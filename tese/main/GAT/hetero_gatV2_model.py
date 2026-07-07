import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, Linear


class HeteroGATv2(nn.Module):
    """
    Heterogeneous Graph Attention Network v2 para Recomendação Afetiva.

    Diferença principal face ao HeteroGAT:
    - Usa GATv2Conv que aceita edge features nativamente no mecanismo de atenção
    - As emoções de cada interação influenciam diretamente o peso da atenção
    - Aplicado separadamente por tipo de aresta e agregado

    Arquitetura:
    - Nós User: features demográficas (47 one-hot)
    - Nós Image: emoções médias (10 features)
    - Arestas LIKED/DISLIKED: emoções da interação (10 features)
    - GATv2Conv: atenção com edge features nativas
    - Predição por produto interno
    """

    def __init__(
        self,
        user_feat_dim,
        image_feat_dim,
        edge_feat_dim,
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):
        super(HeteroGATv2, self).__init__()

        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads

        # ── Projeções iniciais dos nós para hidden_dim
        self.user_proj  = Linear(user_feat_dim,  hidden_dim)
        self.image_proj = Linear(image_feat_dim, hidden_dim)

        # ── GATv2Conv por tipo de aresta e por camada
        # Cada tipo de aresta tem a sua própria convolução
        self.convs_liked    = nn.ModuleList()
        self.convs_disliked = nn.ModuleList()
        self.convs_liked_by    = nn.ModuleList()
        self.convs_disliked_by = nn.ModuleList()

        for _ in range(n_layers):
            # User → Image (LIKED)
            self.convs_liked.append(GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // n_heads,
                heads=n_heads,
                edge_dim=edge_feat_dim,
                dropout=dropout,
                concat=True,
                add_self_loops=False 
            ))
            # User → Image (DISLIKED)
            self.convs_disliked.append(GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // n_heads,
                heads=n_heads,
                edge_dim=edge_feat_dim,
                dropout=dropout,
                concat=True,
                add_self_loops=False 
            ))
            # Image → User (LIKED_BY)
            self.convs_liked_by.append(GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // n_heads,
                heads=n_heads,
                edge_dim=edge_feat_dim,
                dropout=dropout,
                concat=True,
                add_self_loops=False  
            ))
            # Image → User (DISLIKED_BY)
            self.convs_disliked_by.append(GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // n_heads,
                heads=n_heads,
                edge_dim=edge_feat_dim,
                dropout=dropout,
                concat=True,
                add_self_loops=False  
            ))

        self.dropout    = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        x_dict: features dos nós
        edge_index_dict: índices das arestas por tipo
        edge_attr_dict: features das arestas por tipo
        """
        # Projeção inicial
        h_user  = self.activation(self.user_proj(x_dict['user']))
        h_image = self.activation(self.image_proj(x_dict['image']))

        for layer in range(self.n_layers):
            h_user_new  = torch.zeros_like(h_user)
            h_image_new = torch.zeros_like(h_image)
            count_user  = torch.zeros(h_user.size(0), 1, device=h_user.device)
            count_image = torch.zeros(h_image.size(0), 1, device=h_image.device)

            # User → Image (LIKED): atualiza nós de imagem
            if ('user', 'liked', 'image') in edge_index_dict:
                ei   = edge_index_dict[('user', 'liked', 'image')]
                ea   = edge_attr_dict.get(('user', 'liked', 'image'), None)
                out  = self.convs_liked[layer]((h_user, h_image), ei, edge_attr=ea)
                out  = self.activation(self.dropout(out))
                h_image_new += out
                count_image += 1

            # User → Image (DISLIKED): atualiza nós de imagem
            if ('user', 'disliked', 'image') in edge_index_dict:
                ei   = edge_index_dict[('user', 'disliked', 'image')]
                ea   = edge_attr_dict.get(('user', 'disliked', 'image'), None)
                out  = self.convs_disliked[layer]((h_user, h_image), ei, edge_attr=ea)
                out  = self.activation(self.dropout(out))
                h_image_new += out
                count_image += 1

            # Image → User (LIKED_BY): atualiza nós de user
            if ('image', 'liked_by', 'user') in edge_index_dict:
                ei   = edge_index_dict[('image', 'liked_by', 'user')]
                ea   = edge_attr_dict.get(('image', 'liked_by', 'user'), None)
                out  = self.convs_liked_by[layer]((h_image, h_user), ei, edge_attr=ea)
                out  = self.activation(self.dropout(out))
                h_user_new += out
                count_user += 1

            # Image → User (DISLIKED_BY): atualiza nós de user
            if ('image', 'disliked_by', 'user') in edge_index_dict:
                ei   = edge_index_dict[('image', 'disliked_by', 'user')]
                ea   = edge_attr_dict.get(('image', 'disliked_by', 'user'), None)
                out  = self.convs_disliked_by[layer]((h_image, h_user), ei, edge_attr=ea)
                out  = self.activation(self.dropout(out))
                h_user_new += out
                count_user += 1

            # Média das contribuições
            count_image = count_image.clamp(min=1)
            count_user  = count_user.clamp(min=1)
            h_image = h_image_new / count_image
            h_user  = h_user_new  / count_user

        return h_user, h_image

    def bpr_loss(self, user_emb, image_emb, users, pos_items, neg_items, weight_decay=1e-4):
        u_emb   = user_emb[users]
        pos_emb = image_emb[pos_items]
        neg_emb = image_emb[neg_items]

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        reg = weight_decay * (
            u_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / len(users)

        return bpr + reg