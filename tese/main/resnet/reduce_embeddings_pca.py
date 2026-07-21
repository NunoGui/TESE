"""
reduce_embeddings_pca.py

Reduz os embeddings visuais de 2048 para 64 dimensões usando PCA, para
equilibrar melhor com as ~10 features emocionais já usadas nos nós de
imagem do grafo.

O PCA é ajustado (fit) apenas com as imagens que realmente têm interações
no dataset (as que aparecem em ratings_full.csv) - são essas que importam
para o modelo, e é essa distribuição que queremos capturar bem.

Corre a partir de tese\main:
    python reduce_embeddings_pca.py

Requisitos:
    pip install scikit-learn pandas
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

N_COMPONENTES = 64

PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_OUTPUT = os.path.join(PASTA_SCRIPT, "output")
os.makedirs(PASTA_OUTPUT, exist_ok=True)

# --- INPUT partilhado, lido de main\data, NUNCA modificado ---
RATINGS_CSV = os.path.join(PASTA_SCRIPT, "..", "data", "ratings_full.csv")

# --- INPUT próprio (gerado pelo extract_embeddings.py, dentro de resnet\output) ---
EMBEDDINGS_CSV = os.path.join(PASTA_OUTPUT, "image_embeddings.csv")

# --- OUTPUTS: tudo escrito isoladamente dentro de resnet\output ---
OUTPUT_CSV = os.path.join(PASTA_OUTPUT, "image_embeddings_pca64.csv")
OUTPUT_MODELO_PCA = os.path.join(PASTA_OUTPUT, "pca_model.joblib")

# ---------------------------------------------------------------------------
# LÓGICA
# ---------------------------------------------------------------------------

def main():
    ratings = pd.read_csv(RATINGS_CSV)
    embeddings = pd.read_csv(EMBEDDINGS_CSV)

    embed_cols = [c for c in embeddings.columns if c.startswith("embed_")]
    print(f"Embeddings originais: {len(embeddings)} imagens, {len(embed_cols)} dimensões")

    # --- Ajustar o PCA apenas com as imagens que entram no dataset de treino ---
    ids_usados = ratings["image_id"].unique()
    emb_usados = embeddings[embeddings["image_id"].isin(ids_usados)].reset_index(drop=True)
    print(f"Imagens usadas para ajustar o PCA (com interações): {len(emb_usados)}")

    X_fit = emb_usados[embed_cols].values

    pca = PCA(n_components=N_COMPONENTES, random_state=42)
    pca.fit(X_fit)

    variancia_explicada = pca.explained_variance_ratio_.sum()
    print(f"Variância explicada com {N_COMPONENTES} dimensões: {variancia_explicada * 100:.2f}%")

    # --- Aplicar o PCA a TODAS as imagens (não só às usadas em ratings) ---
    # Isto garante que, se no futuro precisares de embeddings para outras
    # imagens (ex: novo lote de dados), já estão prontos e consistentes.
    X_todas = embeddings[embed_cols].values
    X_reduzido = pca.transform(X_todas)

    colunas_pca = [f"visual_pca_{i:02d}" for i in range(N_COMPONENTES)]
    df_reduzido = pd.DataFrame(X_reduzido, columns=colunas_pca)
    df_reduzido.insert(0, "image_id", embeddings["image_id"].values)

    df_reduzido.to_csv(OUTPUT_CSV, index=False)
    print(f"\nGuardado: {OUTPUT_CSV}")
    print(f"Shape final: {df_reduzido.shape}")

    # --- Guardar o modelo PCA treinado (para reproduzir/aplicar a novas imagens depois) ---
    joblib.dump(pca, OUTPUT_MODELO_PCA)
    print(f"Modelo PCA guardado: {OUTPUT_MODELO_PCA}")


if __name__ == "__main__":
    main()