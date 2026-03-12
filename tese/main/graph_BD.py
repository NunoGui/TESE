#pip install sshtunnel mysql-connector-python pandas neo4j

import mysql.connector
import pandas as pd
from sshtunnel import SSHTunnelForwarder
from neo4j import GraphDatabase

# -----------------------------
# NEO4J CONNECTION
# -----------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# -----------------------------
# CREATE GRAPH STRUCTURE
# -----------------------------
def create_graph(tx, user_id, image_id, emotion, intensity):

    tx.run("""
    MERGE (u:User {userId:$user_id})
    MERGE (i:Image {imageId:$image_id})
    MERGE (e:Emotion {type:$emotion})

    MERGE (u)-[:VIEWED]->(i)
    MERGE (u)-[:FELT {intensity:$intensity}]->(e)
    MERGE (i)-[:EVOKED]->(e)
    """,
    user_id=user_id,
    image_id=image_id,
    emotion=emotion,
    intensity=intensity
    )


# -----------------------------
# CONNECT TO MYSQL
# -----------------------------
with SSHTunnelForwarder(
    ('rachel.waik.eu', 22),
    ssh_username='emorecsys',
    ssh_password='TMgx64gP8iD37wmx3wdn',
    remote_bind_address=('127.0.0.1', 3306)
) as tunnel:

    connection = mysql.connector.connect(
        user='emorecsys_reader',
        password='pass',
        host='127.0.0.1',
        port=tunnel.local_bind_port,
        database='emorecsys',
        use_pure=True
    )

    print("Connected to MySQL")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    query_ratings = "SELECT * FROM ratings"
    df_ratings = pd.read_sql(query_ratings, con=connection)

    query_surveys = "SELECT * FROM surveys"
    df_surveys = pd.read_sql(query_surveys, con=connection)

    print("Ratings Loaded:", len(df_ratings))
    print("Surveys Loaded:", len(df_surveys))

    connection.close()


# -----------------------------
# INSERT DATA INTO NEO4J
# -----------------------------
print("Sending data to Neo4j...")

with driver.session() as session:

    for index, row in df_ratings.iterrows():

        user_id = row["user_id"]
        image_id = row["image_id"]
        emotion = row["emotion"]
        intensity = row["rating"]

        session.write_transaction(
            create_graph,
            user_id,
            image_id,
            emotion,
            intensity
        )

print("Graph creation complete!")

driver.close()