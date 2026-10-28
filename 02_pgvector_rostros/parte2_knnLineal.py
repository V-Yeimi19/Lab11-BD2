#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, os, statistics, sys, time
from pathlib import Path

import face_recognition, matplotlib

if "--mostrar" not in sys.argv:
    matplotlib.use("Agg")

import numpy as np
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from parte0_preparacion import mostrar_fotos

DB_PARAMS = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5434")),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "root"),
    "dbname": os.getenv("PGDATABASE", "postgres"),
}

DATOS_DIR = ROOT / "01_busqueda_numpy" / "data"
CONSULTA_POR_DEFECTO = DATOS_DIR / "EvaAyllon.png"
FOTO_POR_DEFECTO = DATOS_DIR / "DobleFamoso.jpeg"
SALIDA_POR_DEFECTO = BASE_DIR / "result" / "doble_famoso.png"

OPERADORES = {"euclidiana": "<->", "coseno": "<=>"}

UMBRAL_MISMA_PERSONA = 0.6

conn = None
cur = None


def conectar():
    global conn, cur
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    register_vector(conn)
    cur = conn.cursor()
    return conn, cur


def extraer_embedding(ruta: str | Path) -> np.ndarray:
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe la imagen: {ruta}")

    imagen = face_recognition.load_image_file(str(ruta))
    ubicaciones = face_recognition.face_locations(imagen)
    if not ubicaciones:
        raise ValueError(f"No se detectó ningún rostro en: {ruta}")

    def area(ubicacion):
        arriba, derecha, abajo, izquierda = ubicacion
        return (abajo - arriba) * (derecha - izquierda)

    if len(ubicaciones) > 1:
        print(f"Advertencia: {len(ubicaciones)} rostros detectados en {ruta.name}; se usa el más grande.")

    idx = max(range(len(ubicaciones)), key=lambda i: area(ubicaciones[i]))
    codificaciones = face_recognition.face_encodings(imagen, [ubicaciones[idx]])
    if not codificaciones:
        raise ValueError(f"No se pudo calcular the embedding de: {ruta}")
    return codificaciones[0]


def knn(q, k: int = 10, op: str = "<->"):
    if op not in OPERADORES.values():
        raise ValueError(f"Operador no permitido: {op!r}")
    cur.execute(f"SELECT id FROM face_embeddings ORDER BY embedding {op} %s LIMIT %s", (q, k))
    return [r[0] for r in cur.fetchall()]


def knn_secuencial(q, k: int = 10, op: str = "<->"):
    cur.execute("SET enable_indexscan = off")
    try:
        return knn(q, k, op)
    finally:
        cur.execute("RESET enable_indexscan")


def medir_knn_secuencial(q, k: int = 10, op: str = "<->", repeticiones: int = 10):
    ejecuciones = []
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        ejecuciones.append(knn_secuencial(q, k, op))
        tiempos.append((time.perf_counter() - inicio) * 1000)
    estable = all(x == ejecuciones[0] for x in ejecuciones)
    return ejecuciones[0], tiempos, estable


def a_numpy(vector) -> np.ndarray:
    if hasattr(vector, "to_numpy"):
        return np.asarray(vector.to_numpy(), dtype=float)
    return np.asarray(list(vector), dtype=float)


def obtener_detalles(ids):
    if not ids:
        return []
    cur.execute("SELECT id, name, path, embedding FROM face_embeddings WHERE id = ANY(%s)", (list(ids),))
    filas = {f[0]: f for f in cur.fetchall()}
    return [filas[i] for i in ids if i in filas]


def obtener_explain(q, k: int = 10, op: str = "<->"):
    cur.execute("SET enable_indexscan = off")
    try:
        cur.execute(
            f"EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, SUMMARY OFF) SELECT id FROM face_embeddings ORDER BY embedding {op} %s LIMIT %s",
            (q, k),
        )
        return [f[0] for f in cur.fetchall()]
    finally:
        cur.execute("RESET enable_indexscan")


def listar_indices():
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'face_embeddings' ORDER BY indexname")
    return [f[0] for f in cur.fetchall()]


def buscar_top_por_persona(q, candidatos: int = 50, top: int = 5):
    cur.execute(
        """
        SELECT name, MIN(dist) AS dist, COUNT(*) AS fotos
        FROM (SELECT name, embedding <-> %s AS dist FROM face_embeddings ORDER BY embedding <-> %s LIMIT %s) t
        GROUP BY name ORDER BY dist LIMIT %s
        """,
        (q, q, candidatos, top),
    )
    return cur.fetchall()


def obtener_representante(q, nombre):
    cur.execute(
        """
        SELECT path, embedding <-> %s AS dist
        FROM face_embeddings
        WHERE name = %s
        ORDER BY embedding <-> %s
        LIMIT 1
        """,
        (q, nombre, q),
    )
    return cur.fetchone()


def imprimir_plan(lineas):
    relevantes = []
    for linea in lineas:
        if not any(c in linea for c in ("Scan", "Sort", "Limit")):
            continue
        texto = " ".join(linea.split())
        relevantes.append(texto if len(texto) <= 110 else f"{texto[:110]}...")
    print("Plan de ejecución (EXPLAIN ANALYZE):")
    for l in relevantes:
        print(f"  {l}")
    return any("Seq Scan on face_embeddings" in l for l in lineas)


def comparar_distancias(ruta_consulta: str | Path, k: int, repeticiones: int) -> dict:
    print("\n" + "=" * 66)
    print("A) kNN SECUENCIAL: EUCLIDIANA vs COSENO")
    print("=" * 66)

    q = extraer_embedding(ruta_consulta)
    print(f"Consulta: {ruta_consulta}")
    print(f"Dimensión: {q.shape[0]} | norma L2 = {np.linalg.norm(q):.4f}")
    print(f"Índices: {listar_indices() or 'ninguno'}")
    print(f"Repeticiones: {repeticiones} | k = {k}")

    resultados = {}
    for nombre, op in OPERADORES.items():
        ids, tiempos, estable = medir_knn_secuencial(q, k=k, op=op, repeticiones=repeticiones)
        resultados[nombre] = {
            "op": op,
            "ids": ids,
            "tiempos": tiempos,
            "mediana": statistics.median(tiempos),
            "estable": estable,
            "plan": obtener_explain(q, k=k, op=op),
        }
        print(f"\n--- {nombre} ({op}) ---")
        print("Tiempos (ms): " + ", ".join(f"{t:.2f}" for t in tiempos))
        print(f"Mediana: {resultados[nombre]['mediana']:.2f} ms | Mín: {min(tiempos):.2f} ms | Máx: {max(tiempos):.2f} ms")
        es_seq = imprimir_plan(resultados[nombre]["plan"])
        print(f"¿Búsqueda secuencial? {'Sí' if es_seq else 'No'}")
        print(f"¿Mismos resultados en todas las ejecuciones? {'Sí' if estable else 'No'}")

        for pos, (id_, nom, _ruta, _emb) in enumerate(obtener_detalles(ids), start=1):
            print(f"  {pos:2d}. id={id_:5d} | {nom}")

    ids_euc = [int(i) for i in resultados["euclidiana"]["ids"]]
    ids_cos = [int(i) for i in resultados["coseno"]["ids"]]
    solapamiento = len(set(ids_euc) & set(ids_cos))
    mismo_orden = ids_euc == ids_cos

    print("\n--- Comparación ---")
    print(f"Vecinos en común: {solapamiento}/{k}")
    print(f"¿Mismo orden? {'Sí' if mismo_orden else 'No'}")

    if solapamiento == k and mismo_orden:
        print("Conclusión: ambos operadores devuelven el mismo conjunto y orden.")
    elif solapamiento == k:
        print("Conclusión: mismo conjunto, distinto orden. Los embeddings no están normalizados.")
    else:
        print("Conclusión: conjuntos distintos por la diferencia entre norma y ángulo.")

    resultados["q"] = q
    return resultados


def buscar_dobles(ruta_foto: str | Path, candidatos: int = 50, top: int = 5, salida: str | Path = SALIDA_POR_DEFECTO, mostrar: bool = False):
    print("\n" + "=" * 66)
    print("B) PERSONAS DISTINTAS MÁS PARECIDAS")
    print("=" * 66)

    q = extraer_embedding(ruta_foto)
    print(f"Foto: {ruta_foto}")
    print(f"Norma L2: {np.linalg.norm(q):.4f}")

    dobles = buscar_top_por_persona(q, candidatos=candidatos, top=top)
    if not dobles:
        raise ValueError("La tabla face_embeddings está vacía.")

    rutas = [str(ruta_foto)]
    titulos = ["MI FOTO\n(consulta)"]
    personas = ["MI FOTO"]

    print(f"\n{'#':>2} {'persona':<25} {'dist':>7} {'fotos':>6}  {'< 0.6':>6}")
    for pos, (nombre, dist, fotos) in enumerate(dobles, start=1):
        dist = float(dist)
        rep = obtener_representante(q, nombre)
        if rep:
            rutas.append(rep[0])
            titulos.append(f"#{pos} {nombre}\n{dist:.3f} | {fotos} fotos")
            personas.append(nombre)
        print(f"{pos:2d} {nombre:<25} {dist:7.4f} {fotos:6d}  {'sí' if dist < UMBRAL_MISMA_PERSONA else 'no':>6}")

    galeria = pd.DataFrame({"person": personas, "path": rutas})
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    mostrar_fotos(galeria, list(range(len(galeria))), salida=str(salida), mostrar=mostrar, titulos=titulos, columnas=3)
    print(f"\nGrilla guardada en: {salida}")
    return dobles


def main() -> None:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Parte 2: kNN secuencial y búsqueda de personas similares")
    parser.add_argument("--consulta", default=str(CONSULTA_POR_DEFECTO))
    parser.add_argument("--foto", default=str(FOTO_POR_DEFECTO))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeticiones", type=int, default=10)
    parser.add_argument("--candidatos", type=int, default=50)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--salida", default=str(SALIDA_POR_DEFECTO))
    parser.add_argument("--mostrar", action="store_true")
    args = parser.parse_args()

    conectar()
    try:
        comparar_distancias(args.consulta, k=args.k, repeticiones=args.repeticiones)
        buscar_dobles(args.foto, candidatos=args.candidatos, top=args.top, salida=args.salida, mostrar=args.mostrar)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
