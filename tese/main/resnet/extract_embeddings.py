"""
extract_embeddings.py

Extrai embeddings visuais com ResNet50 (pré-treinado em ImageNet) para todas
as imagens do dataset, usando photos.csv como correspondência entre
image_id e o ficheiro real na pasta de imagens.

Gera um CSV com uma linha por imagem: image_id + 2048 colunas de embedding
(embed_0000 ... embed_2047).

Corre a partir de tese\main:
    python extract_embeddings.py

Requisitos (instalar se necessário):
    pip install torch torchvision pandas pillow tqdm
"""

import os
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

# Caminhos relativos à pasta onde este script está guardado (main\resnet),
# resolvidos automaticamente para apontar para main\data e main\imagens,
# independentemente de onde correres o comando python.
PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

PHOTOS_CSV = os.path.join(PASTA_SCRIPT, "..", "data", "photos.csv")
IMAGES_DIR = os.path.join(PASTA_SCRIPT, "..", "imagens")
OUTPUT_CSV = os.path.join(PASTA_SCRIPT, "..", "data", "image_embeddings.csv")
FALHAS_CSV = os.path.join(PASTA_SCRIPT, "..", "data", "embeddings_falhas.csv")

# ---------------------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"A usar device: {device}")

print("A carregar ResNet50 pré-treinado...")
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove a camada de classificação final
resnet = resnet.to(device)
resnet.eval()

# Transformações standard usadas no treino da ResNet (ImageNet)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# CARREGAR A LISTA DE IMAGENS
# ---------------------------------------------------------------------------

photos = pd.read_csv(PHOTOS_CSV)
print(f"Total de imagens em {PHOTOS_CSV}: {len(photos)}")

# ---------------------------------------------------------------------------
# EXTRAÇÃO
# ---------------------------------------------------------------------------

embeddings = []
falhas = []

with torch.no_grad():
    for _, row in tqdm(photos.iterrows(), total=len(photos), desc="A extrair embeddings"):
        image_id = row["image_id"]
        file_name = row["file_name"]
        ext = row["ext"]

        img_path = os.path.join(IMAGES_DIR, f"{file_name}.{ext}")

        if not os.path.exists(img_path):
            falhas.append((image_id, img_path, "ficheiro não encontrado"))
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)

            vetor = resnet(tensor)                # shape: (1, 2048, 1, 1)
            vetor = vetor.squeeze().cpu().numpy()  # shape: (2048,)

            linha = {"image_id": image_id}
            linha.update({f"embed_{i:04d}": v for i, v in enumerate(vetor)})
            embeddings.append(linha)

        except Exception as e:
            falhas.append((image_id, img_path, str(e)))

# ---------------------------------------------------------------------------
# GUARDAR RESULTADOS
# ---------------------------------------------------------------------------

df_embeddings = pd.DataFrame(embeddings)
df_embeddings.to_csv(OUTPUT_CSV, index=False)

print(f"\nGuardado: {OUTPUT_CSV}")
print(f"Embeddings extraídos com sucesso: {len(df_embeddings)} / {len(photos)}")

if falhas:
    print(f"\nFalharam {len(falhas)} imagens:")
    for image_id, path, motivo in falhas[:20]:  # mostra só as primeiras 20
        print(f"  image_id={image_id} | {path} | {motivo}")
    if len(falhas) > 20:
        print(f"  ... e mais {len(falhas) - 20} falhas.")

    # guardar a lista completa de falhas para inspecionar depois
    df_falhas = pd.DataFrame(falhas, columns=["image_id", "path", "motivo"])
    df_falhas.to_csv(FALHAS_CSV, index=False)
    print(f"\nLista completa de falhas guardada em: {FALHAS_CSV}")