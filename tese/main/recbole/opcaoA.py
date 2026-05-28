import pandas as pd
from neo4j import GraphDatabase
 
# ── Configuração
URI      = "neo4j://127.0.0.1:7687"
USER     = "neo4j"
PASSWORD = "EmoRecSys"  
 
# ── Caminhos dos CSVs
PATH_USERS   = "users.csv"
PATH_IMAGES  = "images.csv"
PATH_RATINGS = "ratings_full.csv"
 
# ── Carregar dados dos CSVs 
print("A carregar CSVs...")
users   = pd.read_csv(PATH_USERS)
images  = pd.read_csv(PATH_IMAGES)
ratings = pd.read_csv(PATH_RATINGS)
 
# Limpar valores nulos
users  = users.fillna("")
images = images.fillna("")
ratings = ratings.fillna(0)
 
print(f"Users:   {len(users)}")
print(f"Images:  {len(images)}")
print(f"Ratings: {len(ratings)}")
 
# ── Ligação ao Neo4j 
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
 
def criar_indices(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User)  REQUIRE u.user_id  IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.image_id IS UNIQUE")
 
def criar_users(tx, batch):
    tx.run("""
        UNWIND $rows AS row
        MERGE (u:User {user_id: toString(row.user_id)})
        SET u.age_group      = row.age_group,
            u.gender         = row.gender,
            u.education      = row.education,
            u.location_city  = row.location_city,
            u.country        = row.country,
            u.populational_aff = row.populational_aff,
            u.interest_tags  = row.interest_tags
    """, rows=batch)
 
def criar_images(tx, batch):
    tx.run("""
        UNWIND $rows AS row
        MERGE (i:Image {image_id: toString(row.image_id)})
    """, rows=batch)
 
def criar_rated(tx, batch):
    tx.run("""
        UNWIND $rows AS row
        MATCH (u:User  {user_id:  toString(row.user_id)})
        MATCH (i:Image {image_id: toString(row.image_id)})
        MERGE (u)-[r:RATED]->(i)
        SET r.rating     = toInteger(row.rating),
            r.valence    = toFloat(row.valence),
            r.arousal    = toFloat(row.arousal),
            r.dominance  = toFloat(row.dominance),
            r.happiness  = toInteger(row.happiness),
            r.sadness    = toInteger(row.sadness),
            r.anger      = toInteger(row.anger),
            r.fear       = toInteger(row.fear),
            r.surprise   = toInteger(row.surprise),
            r.disgust    = toInteger(row.disgust),
            r.neutral    = toInteger(row.neutral)
    """, rows=batch)
 
def inserir_em_batches(session, dados, funcao, tamanho=500, label=""):
    total = len(dados)
    for i in range(0, total, tamanho):
        batch = dados[i:i+tamanho].to_dict('records')
        session.execute_write(funcao, batch)
        print(f"  {label}: {min(i+tamanho, total)}/{total}")
 
# ── Popular grafo 
with driver.session() as session:
 
    print("\n1. A criar índices...")
    session.execute_write(criar_indices)
 
    print("\n2. A criar nós User...")
    inserir_em_batches(session, users, criar_users, label="Users")
 
    print("\n3. A criar nós Image...")
    inserir_em_batches(session, images, criar_images, label="Images")
 
    print("\n4. A criar relações RATED...")
    inserir_em_batches(session, ratings, criar_rated, label="RATED")
 
    # Verificação final
    resultado = session.run("""
        MATCH (n) WITH labels(n) AS label, count(n) AS total
        RETURN label, total
        UNION
        MATCH ()-[r]->() RETURN [type(r)] AS label, count(r) AS total
    """)
    print("\n=== RESULTADO FINAL ===")
    for r in resultado:
        print(f"  {r['label'][0]}: {r['total']}")
 
driver.close()
print("\nGrafo Opção A construído com sucesso!")