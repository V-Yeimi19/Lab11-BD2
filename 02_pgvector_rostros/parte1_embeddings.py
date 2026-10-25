import os
import glob
import pandas as pd
import face_recognition
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

DATASET_PATH = r"D:\Descargas\Lab11_pgvector\lfw_funneled"

coleccion = pd.DataFrame([
    {
        "person": os.path.basename(os.path.dirname(p)),
        "path": p
    }
    for p in glob.iglob(
        os.path.join(DATASET_PATH, "**", "*.jpg"),
        recursive=True
    )
])

print("Cantidad de imágenes:", len(coleccion))
print(coleccion.head())

def generate_face_embeddings(coleccion, n):
    resultados = []

    for _, fila in coleccion.head(n).iterrows():
        image = face_recognition.load_image_file(fila["path"])
        encodings = face_recognition.face_encodings(image)

        if encodings:
            embedding = encodings[0]
            resultados.append(
                (
                    fila["person"],
                    fila["path"],
                    embedding
                )
            )

    return resultados

prueba = generate_face_embeddings(coleccion, 10)

print("Rostros detectados:", len(prueba))

if prueba:
    print("Persona:", prueba[0][0])
    print("Ruta:", prueba[0][1])
    print("Dimensión del embedding:", len(prueba[0][2]))

conteos = coleccion["person"].value_counts()                 # Cuenta cuántas fotos tiene cada persona

personas_validas = conteos[conteos >= 2].index             # Conserva solo personas que tienen al menos 2 fotos

candidatos_consulta = coleccion[
    coleccion["person"].isin(personas_validas)
]                                                           # Filtra las imágenes pertenecientes a esas personas

consultas = candidatos_consulta.sample(
    n=100,
    random_state=42
)                                                           # Selecciona 100 imágenes aleatorias y reproducibles

consultas.to_csv(
    "consultas_reservadas.csv",
    index=False
)                                                           # Guarda esas 100 consultas para usarlas después en P4

coleccion_para_bd = coleccion.drop(
    consultas.index
)                                                           # Elimina esas 100 imágenes de la colección que irá a PostgreSQL

print("Consultas reservadas:", len(consultas))
print("Imágenes restantes para BD:", len(coleccion_para_bd))

conn = psycopg2.connect(
    host="localhost",
    port=5434,
    user="postgres",
    password="root",
    dbname="postgres"
)

conn.autocommit = True
cur = conn.cursor()

cur.execute("TRUNCATE TABLE face_embeddings RESTART IDENTITY")

registros = generate_face_embeddings(
    coleccion_para_bd,
    len(coleccion_para_bd)
)

print("Embeddings generados:", len(registros))

execute_values(
    cur,
    """
    INSERT INTO face_embeddings
    (name, path, embedding)
    VALUES %s
    """,
    registros
)

cur.execute("SELECT COUNT(*) FROM face_embeddings")

print(
    "Embeddings guardados:",
    cur.fetchone()[0]
)

cur.close()
conn.close()