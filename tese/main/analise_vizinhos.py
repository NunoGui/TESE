import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
 
# ── Configuração
PATH_RATINGS = "data/ratings_full.csv"
TOTAL_ITEMS  = 3084
MIN_TEST     = 20
SEED         = 42
K_VIZINHOS   = 10
N_USERS      = 5
 
FEATURE_COLS = [
    "rating", "valence", "arousal", "dominance",
    "happiness", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
]
 
# ──────────────────────────────────────────────
# 1. Carregar dados e fazer split
# ──────────────────────────────────────────────
r = pd.read_csv(PATH_RATINGS).fillna(0)
r = r.rename(columns={"user_id": "user", "image_id": "item"})
 
# Remover degenerados
users_all_zeros = r.groupby('user')['rating'].transform(lambda x: (x == 0).all())
users_all_ones  = r.groupby('user')['rating'].transform(lambda x: (x == 1).all())
r = r[~users_all_zeros & ~users_all_ones].copy()
 
# Split por user
train_list, test_list = [], []
for user, group in r.groupby('user'):
    if len(group) < 2:
        train_list.append(group)
        continue
    train, test = train_test_split(group, test_size=0.2, random_state=SEED)
    train_list.append(train)
    test_list.append(test)
 
train_df = pd.concat(train_list).reset_index(drop=True)
test_df  = pd.concat(test_list).reset_index(drop=True)
 
# ──────────────────────────────────────────────
# 2. Calcular similaridade entre users
# ──────────────────────────────────────────────
user_profiles = train_df.groupby("user")[FEATURE_COLS].mean()
scaler = StandardScaler()
scaled = pd.DataFrame(
    scaler.fit_transform(user_profiles),
    index=user_profiles.index,
    columns=user_profiles.columns
)
sim_matrix = cosine_similarity(scaled)
sim_df = pd.DataFrame(sim_matrix, index=scaled.index, columns=scaled.index)
 
# ──────────────────────────────────────────────
# 3. Escolher 5 users aleatórios com positivos no teste
# ──────────────────────────────────────────────
rng = np.random.default_rng(SEED)
valid_users = []
for user in test_df['user'].unique():
    pos = test_df[(test_df['user'] == user) & (test_df['rating'] == 1)]
    if len(pos) > 0:
        valid_users.append(user)
 
sample_users = rng.choice(valid_users, size=N_USERS, replace=False).tolist()
 
# ──────────────────────────────────────────────
# 4. Análise manual por user
# ──────────────────────────────────────────────
all_items = set(range(1, TOTAL_ITEMS + 1))
rng2 = np.random.default_rng(SEED)
 
for user in sample_users:
    print("\n" + "="*65)
    print(f"USER {user}")
    print("="*65)
 
    # Interações completas
    user_data  = r[r['user'] == user]
    train_user = train_df[train_df['user'] == user]
    test_user  = test_df[test_df['user'] == user]
 
    pos_treino = train_user[train_user['rating'] == 1]['item'].tolist()
    neg_treino = train_user[train_user['rating'] == 0]['item'].tolist()
    pos_teste  = test_user[test_user['rating'] == 1]['item'].tolist()
    neg_teste  = test_user[test_user['rating'] == 0]['item'].tolist()
 
    print(f"\n  TREINO ({len(train_user)} itens):")
    print(f"    Positivos ({len(pos_treino)}): {sorted(pos_treino)}")
    print(f"    Negativos ({len(neg_treino)}): {sorted(neg_treino)}")
 
    print(f"\n  TESTE antes de negativos ({len(test_user)} itens):")
    print(f"    Positivos ({len(pos_teste)}): {sorted(pos_teste)}")
    print(f"    Negativos ({len(neg_teste)}): {sorted(neg_teste)}")
 
    # Adicionar negativos
    seen   = set(user_data['item'])
    unseen = list(all_items - seen)
    n_neg  = max(0, MIN_TEST - len(test_user))
    neg_sample = rng2.choice(unseen, size=n_neg, replace=False).tolist()
 
    print(f"\n  Negativos adicionados ({n_neg}): {sorted(neg_sample)}")
    print(f"  Total candidatos no teste: {len(test_user) + n_neg}")
 
    # Vizinhos
    if user in sim_df.index:
        vizinhos = sim_df[user].drop(index=user).nlargest(K_VIZINHOS).index.tolist()
        print(f"\n  Top {K_VIZINHOS} vizinhos: {vizinhos}")
 
        # Verificar se vizinhos avaliaram os positivos do teste
        print(f"\n  Cobertura dos positivos do teste pelos vizinhos:")
        for img in pos_teste:
            avaliaram = train_df[
                (train_df['item'] == img) & (train_df['user'].isin(vizinhos))
            ]
            if len(avaliaram) == 0:
                print(f"    Imagem {img}: NENHUM vizinho avaliou ❌")
            else:
                ratings_viz = avaliaram[['user', 'rating']].values.tolist()
                print(f"    Imagem {img}: avaliada por {len(avaliaram)} vizinho(s) ✓ → {ratings_viz}")
 
    print()
 
print("\n" + "="*65)
print("Análise concluída.")