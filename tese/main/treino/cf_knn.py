import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
 
# input 
ratings = pd.read_csv('ratings_full.csv')
 
# separar positivos e negativos confirmados
positivo = ratings[ratings['rating'] == 1][['user_id', 'image_id']].copy()
negativo = ratings[ratings['rating'] == 0][['user_id', 'image_id']].copy()
 
print(f"Total de interações positivas: {len(positivo)}")
print(f"Total de interações negativas confirmadas: {len(negativo)}")
print(f"Utilizadores únicos: {positivo['user_id'].nunique()}")
print(f"Imagens únicas: {positivo['image_id'].nunique()}")
 
# dicionário de negativos confirmados por utilizador
user_neg_dict = negativo.groupby('user_id')['image_id'].apply(set).to_dict()
 
# construir matriz User-Image (apenas positivos) 
positivo['rating'] = 1
 
matriz = positivo.pivot_table(
    index='user_id',
    columns='image_id',
    values='rating',
    aggfunc='max',
    fill_value=0
)
 
print(f"\nDimensões da matriz: {matriz.shape}")
print(f"Utilizadores: {matriz.shape[0]}")
print(f"Imagens: {matriz.shape[1]}")
print(f"\nPrimeiras 5 linhas e 5 colunas:")
print(matriz.iloc[:5, :5])
 
# calcular similaridade do cosseno 
sim_matrix = cosine_similarity(matriz.values)
sim_df = pd.DataFrame(sim_matrix, index=matriz.index, columns=matriz.index)
 
print(f"\nMatriz de similaridade: {sim_df.shape}")
print(f"\nSimilaridade do User 1 com os primeiros 10 utilizadores:")
print(sim_df.iloc[1, :10].round(3))
 
# função de recomendação 
def recomendar(user_id, matriz, sim_df, K=10, N=10):
 
    # vizinhos mais similares (excluindo o próprio)
    similares = sim_df[user_id].drop(user_id).nlargest(K)
 
    # imagens já vistas positivamente pelo utilizador
    ja_vistas_pos = set(matriz.loc[user_id][matriz.loc[user_id] == 1].index)
 
    # imagens rejeitadas confirmadas pelo utilizador
    ja_rejeitadas = user_neg_dict.get(user_id, set())
 
    # imagens a excluir — positivas já vistas + negativas confirmadas
    excluir = ja_vistas_pos | ja_rejeitadas
 
    # agregar scores das imagens dos vizinhos
    scores = {}
    for vizinho, sim in similares.items():
        imagens_vizinho = matriz.loc[vizinho][matriz.loc[vizinho] == 1].index
        for img in imagens_vizinho:
            if img not in excluir:
                scores[img] = scores.get(img, 0) + sim
 
    # ordenar por score e devolver top-N
    recomendadas = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:N]
 
    return recomendadas
 
# testar com o User 1
recomendacoes = recomendar(user_id=1, matriz=matriz, sim_df=sim_df, K=10, N=10)
 
print(f"\nRecomendações para o User 1:")
for rank, (img, score) in enumerate(recomendacoes, 1):
    print(f"  {rank}º — Imagem {img} (score: {score:.3f})")
 
# avaliação: Precision@K e Recall@K 
def avaliar(matriz, K=10, N=10, test_ratio=0.2):
    precisoes = []
    recalls   = []

    for user_id in matriz.index:
        # imagens positivas do utilizador
        positivas = list(matriz.loc[user_id][matriz.loc[user_id] == 1].index)

        if len(positivas) < 2:
            continue

        # treino/teste
        n_teste = max(1, int(len(positivas) * test_ratio))
        teste   = set(positivas[:n_teste]) # 20%
        treino  = set(positivas[n_teste:]) # 80%

        # matriz de treino — remove imagens de teste deste utilizador
        matriz_treino = matriz.copy()
        for img in teste:
            matriz_treino.loc[user_id, img] = 0

        # recalcular similaridade APENAS com dados de treino
        sim_treino = cosine_similarity(matriz_treino.values)
        sim_treino_df = pd.DataFrame(
            sim_treino,
            index=matriz_treino.index,
            columns=matriz_treino.index
        )

        # recomendações com similaridade de treino
        recomendacoes = recomendar(user_id, matriz_treino, sim_treino_df, K=K, N=N)
        recomendadas  = set([img for img, score in recomendacoes])

        # métricas
        acertos  = len(recomendadas & teste)
        precisao = acertos / N
        recall   = acertos / len(teste) if len(teste) > 0 else 0

        precisoes.append(precisao)
        recalls.append(recall)

    print(f"\nAvaliação CF KNN — sem data leakage (K={K}, N={N})")
    print(f"Utilizadores avaliados: {len(precisoes)}")
    print(f"Precision@{N}: {np.mean(precisoes):.3f}")
    print(f"Recall@{N}:    {np.mean(recalls):.3f}")

avaliar(matriz, K=10, N=10)
 
# visualizar matriz de similaridade 
plt.figure(figsize=(12, 10))
sns.heatmap(
    sim_df.iloc[:30, :30],
    cmap='YlOrRd',
    vmin=0,
    vmax=1,
    xticklabels=sim_df.index[:30],
    yticklabels=sim_df.index[:30]
)
plt.title('Matriz de Similaridade do Cosseno — primeiros 30 utilizadores')
plt.xlabel('Utilizador')
plt.ylabel('Utilizador')
plt.tight_layout()
plt.savefig('similaridade_matrix.png', dpi=150)
plt.show()
print("Gráfico guardado em similaridade_matrix.png")
 
# perfil de utilizador específico e explicação das recomendações
def explicar_recomendacoes(user_id, matriz, sim_df, K=10, N=10):
 
    # imagens vistas positivamente pelo utilizador
    ja_vistas = list(matriz.loc[user_id][matriz.loc[user_id] == 1].index)
 
    # imagens rejeitadas confirmadas
    rejeitadas = list(user_neg_dict.get(user_id, set()))
 
    print(f"\n{'='*50}")
    print(f"PERFIL DO USER {user_id}")
    print(f"{'='*50}")
    print(f"Imagens avaliadas positivamente: {len(ja_vistas)}")
    print(f"IDs positivos: {ja_vistas}")
    print(f"Imagens rejeitadas confirmadas: {len(rejeitadas)}")
    print(f"IDs negativos: {rejeitadas}")
 
    # K vizinhos mais similares
    similares = sim_df[user_id].drop(user_id).nlargest(K)
 
    print(f"\nTop {K} vizinhos mais similares:")
    for vizinho, sim in similares.items():
        n_comum = len(set(ja_vistas) & set(matriz.loc[vizinho][matriz.loc[vizinho] == 1].index))
        print(f"  User {vizinho} — similaridade: {sim:.3f} — imagens em comum: {n_comum}")
 
    # recomendações com explicação
    recomendacoes = recomendar(user_id, matriz, sim_df, K=K, N=N)
 
    print(f"\nTop {N} recomendações e porquê:")
    for rank, (img, score) in enumerate(recomendacoes, 1):
        contribuidores = []
        for vizinho, sim in similares.items():
            if matriz.loc[vizinho, img] == 1:
                contribuidores.append(f"User {vizinho} (sim={sim:.3f})")
        print(f"  {rank}º — Imagem {img} (score: {score:.3f})")
        print(f"         Recomendada por: {', '.join(contribuidores)}")
 
# User 1
explicar_recomendacoes(user_id=1, matriz=matriz, sim_df=sim_df, K=10, N=10)