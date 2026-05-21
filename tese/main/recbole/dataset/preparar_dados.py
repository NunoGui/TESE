import pandas as pd

# Carrega o ficheiro de ratings
ratings = pd.read_csv('ratings_full.csv')  # ajusta o caminho se necessário

# Filtra apenas interações positivas (rating = 1)
positivo = ratings[ratings['rating'] == 1][['user_id', 'image_id', 'rating']].copy()

# Renomeia colunas para o formato RecBole
positivo.columns = ['user_id:token', 'item_id:token', 'rating:float']

# Guarda o ficheiro .inter
positivo.to_csv('emorecsys.inter', sep='\t', index=False)

print(f"Ficheiro criado com sucesso!")
print(f"Total de interações: {len(positivo)}")
print(f"Utilizadores: {positivo['user_id:token'].nunique()}")
print(f"Imagens: {positivo['item_id:token'].nunique()}")