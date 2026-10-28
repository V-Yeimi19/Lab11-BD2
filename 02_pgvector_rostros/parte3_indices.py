#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib

if "--mostrar" not in sys.argv:
    matplotlib.use("Agg")

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from parte0_preparacion import mostrar_fotos
import parte2_knnLineal as p2

DATOS_DIR = ROOT / "01_busqueda_numpy" / "data"
FOTO_POR_DEFECTO = DATOS_DIR / "DobleFamoso.jpeg"
SALIDA_POR_DEFECTO = BASE_DIR / "result" / "comparacion_indices.png"


def conectar():
    return p2.conectar()


def listar_indices():
    p2.cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'face_embeddings' ORDER BY indexname")
    return [f[0] for f in p2.cur.fetchall()]


def explicar_plan(q, k=10):
    p2.cur.execute(
        "EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY OFF, BUFFERS OFF) SELECT id FROM face_embeddings ORDER BY embedding <-> %s LIMIT %s",
        (q, k),
    )
    return [f[0] for f in p2.cur.fetchall()]


def mostrar_plan(plan):
    print("Plan de ejecución:")
    for linea in plan:
        if any(x in linea for x in ("Index Scan", "Seq Scan", "Sort", "Limit")):
            print("  " + " ".join(linea.split()))


def obtener_resultados(q, candidatos=50, top=5, metodo="lineal", probes=10, ef_search=100):
    if metodo == "lineal":
        p2.cur.execute("SET enable_indexscan = off")
        p2.cur.execute("RESET ivfflat.probes")
        p2.cur.execute("RESET hnsw.ef_search")
    elif metodo == "ivfflat":
        p2.cur.execute("RESET enable_indexscan")
        p2.cur.execute(f"SET ivfflat.probes = {probes}")
        p2.cur.execute("RESET hnsw.ef_search")
    elif metodo == "hnsw":
        p2.cur.execute("RESET enable_indexscan")
        p2.cur.execute("RESET ivfflat.probes")
        p2.cur.execute(f"SET hnsw.ef_search = {ef_search}")
    else:
        raise ValueError("metodo debe be: lineal | ivfflat | hnsw")

    inicio = time.perf_counter()
    dobles = p2.buscar_top_por_persona(q, candidatos=candidatos, top=top)
    elapsed = (time.perf_counter() - inicio) * 1000

    if metodo == "lineal":
        p2.cur.execute("RESET enable_indexscan")

    return dobles, elapsed


def main():
    parser = argparse.ArgumentParser(description="Comparación entre búsqueda secuencial, IVFFlat y HNSW")
    parser.add_argument("--foto", default=str(FOTO_POR_DEFECTO))
    parser.add_argument("--metodo", choices=["lineal", "ivfflat", "hnsw", "todos"], default="todos")
    parser.add_argument("--candidatos", type=int, default=50)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--probes", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--salida", default=str(SALIDA_POR_DEFECTO))
    parser.add_argument("--mostrar", action="store_true")
    parser.add_argument("--sin-explain", action="store_true")
    args = parser.parse_args()

    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8", errors="replace")

    conectar()
    try:
        print("=" * 80)
        print("COMPARACIÓN DE ÍNDICES")
        print("=" * 80)
        print("Índices activos:")
        for idx in listar_indices():
            print(f"  - {idx}")
        print()

        q = p2.extraer_embedding(args.foto)
        metodos = ["lineal", "ivfflat", "hnsw"] if args.metodo == "todos" else [args.metodo]
        resultados = {}

        for m in metodos:
            print("-" * 80)
            print(f"MÉTODO: {m.upper()}")
            if m == "ivfflat":
                print(f"ivfflat.probes = {args.probes}")
            if m == "hnsw":
                print(f"hnsw.ef_search = {args.ef_search}")
            print("-" * 80)

            dobles, elapsed = obtener_resultados(q, args.candidatos, args.top, m, args.probes, args.ef_search)
            resultados[m] = {"dobles": dobles, "ms": elapsed}

            if not args.sin_explain and m != "lineal":
                plan = explicar_plan(q, k=args.top)
                mostrar_plan(plan)
            elif m == "lineal" and not args.sin_explain:
                p2.cur.execute("SET enable_indexscan = off")
                plan = explicar_plan(q, k=args.top)
                mostrar_plan(plan)
                p2.cur.execute("RESET enable_indexscan")

            print(f"Tiempo: {elapsed:.2f} ms\n")
            for i, (nombre, dist, fotos) in enumerate(dobles, 1):
                print(f"{i}. {nombre} - {float(dist):.4f} ({fotos} fotos)")
            print()

        if "lineal" in resultados and len(metodos) > 1:
            ref = [x[0] for x in resultados["lineal"]["dobles"]]
            print("=" * 80)
            print("COMPARATIVA VS BÚSQUEDA SECUENCIAL")
            print("=" * 80)
            print(f"{'MÉTODO':<10} {'ms':>10} {'mismo_conjunto':>16} {'mismo_orden':>14} {'coinciden':>12}")
            print("-" * 70)
            for m in metodos:
                if m == "lineal":
                    print(f"{m:<10} {resultados[m]['ms']:10.2f} {'-':>16} {'-':>14} {'-':>12}")
                    continue
                cur_list = [x[0] for x in resultados[m]["dobles"]]
                mismo_conjunto = set(cur_list) == set(ref)
                mismo_orden = cur_list == ref
                coinciden = sum(1 for a, b in zip(cur_list, ref) if a == b)
                print(f"{m:<10} {resultados[m]['ms']:10.2f} {str(mismo_conjunto):>16} {str(mismo_orden):>14} {f'{coinciden}/{len(ref)}':>12}")
            print()

        rutas = [str(args.foto)]
        titulos = ["MI FOTO"]
        personas = ["MI FOTO"]
        for m in metodos:
            for i, (nombre, dist, fotos) in enumerate(resultados[m]["dobles"], 1):
                rep = p2.obtener_representante(q, nombre)
                if rep:
                    rutas.append(rep[0])
                    titulos.append(f"[{m.upper()}] #{i} {nombre}\n{float(dist):.4f} ({fotos})")
                    personas.append(f"{m}|{nombre}")

        galeria = pd.DataFrame({"person": personas, "path": rutas})
        salida = Path(args.salida)
        salida.parent.mkdir(parents=True, exist_ok=True)
        mostrar_fotos(galeria, list(range(len(galeria))), salida=str(salida), mostrar=args.mostrar, titulos=titulos, columnas=6)
        print(f"Grilla guardada: {salida}")
        print("=" * 80)

    finally:
        if p2.cur:
            p2.cur.close()
        if p2.conn:
            p2.conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
