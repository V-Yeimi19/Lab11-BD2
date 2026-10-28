import os, glob
import pandas as pd
import face_recognition
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

DATASET_PATH = os.getenv("LFW_PATH", r"D:\Descargas\Lab11_pgvector\lfw_funneled")

imagenes = list(glob.iglob(os.path.join(DATASET_PATH, "**", "*.jpg"), recursive=True))

coleccion = pd.DataFrame(
    [{"person": os.path.basename(os.path.dirname(p)), "path": p} for p in imagenes]
)

print("Cantidad de imágenes:", len(coleccion))
print(coleccion.head())


def extraer_embeddings(coleccion_df, n):
    resultados = []
    for _, fila in coleccion_df.head(n).iterrows():
        imagen = face_recognition.load_image_file(fila["path"])
        codificaciones = face_recognition.face_encodings(imagen)
        if codificaciones:
            resultados.append((fila["person"], fila["path"], codificaciones[0]))
    return resultados


print(f"Rostros detectados: {len(extraer_embeddings(coleccion, 10))}")
primer = extraer_embeddings(coleccion, 1)[0]
print("Persona:", primer[0])
print("Ruta:", primer[1])
print("Dimensión del embedding:", len(primer[2]))

conteos = coleccion["person"].value_counts()
personas_validas = conteos[conteos >= 2].index
candidatos = coleccion[coleccion["person"].isin(personas_validas)].copy()

consultas_reservadas = candidatos.sample(n=100, random_state=123)
coleccion_bd = candidatos.drop(consultas_reservadas.index)

consultas_reservadas.to_csv("consultas_reservadas.csv", index=False)
print("Consultas reservadas: 100")
print("Imágenes restantes para BD:", len(coleccion_bd))

embeddings = extraer_embeddings(coleccion_bd, len(coleccion_bd))
print("Embeddings generados:", len(embeddings))

conn = psycopg2.connect(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5434")),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", "root"),
    dbname=os.getenv("PGDATABASE", "postgres"),
)
conn.autocommit = True
register_vector(conn)
cur = conn.cursor()

cur.execute("TRUNCATE TABLE face_embeddings RESTART IDENTITY")

datos = [(e[0], e[1], e[2]) for e in embeddings]
execute_values(cur, "INSERT INTO face_embeddings (name, path, embedding) VALUES %s", datos)

print("Embeddings guardados:", len(datos))
cur.close()
conn.close()
