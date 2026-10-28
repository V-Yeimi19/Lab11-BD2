from __future__ import annotations

import argparse, os, statistics, sys, time, matplotlib
from pathlib import Path

if "--mostrar" not in sys.argv:
    matplotlib.use("Agg")

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import parte2_knnLineal as p2

CSV_POR_DEFECTO = ROOT / "consultas_reservadas.csv"


def cargar_consultas(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encuentra {csv_path}")
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("Falta la columna 'path' en el CSV")
    return df.to_dict("records")


def main() -> None:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Recall@k frente a búsqueda secuencial")
    parser.add_argument("--csv", default=str(CSV_POR_DEFECTO))
    parser.add_argument("--metodo", choices=["ivfflat", "hnsw"], required=True)
    parser.add_argument("--probes", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeticiones", type=int, default=1)
    args = parser.parse_args()

    print("=" * 80)
    print("EVALUACIÓN DE RECALL Y LATENCIA")
    print("=" * 80)

    p2.conectar()
    try:
        consultas = cargar_consultas(Path(args.csv))
        print(f"Consultas: {len(consultas)} | k={args.k} | método={args.metodo.upper()}")

        print("\n[1/3] Obteniendo ground truth con búsqueda secuencial...")
        exactos = []
        for i, row in enumerate(consultas, 1):
            q = p2.extraer_embedding(row["path"])
            ids_exact = p2.knn_secuencial(q, k=args.k, op="<->")
            exactos.append(set(int(x) for x in ids_exact))
            if i % 10 == 0:
                print(f"  {i}/{len(consultas)}")

        print("Listo.\n[2/3] Evaluando con índice...")
        p2.cur.execute("SET enable_seqscan = off")
        p2.cur.execute("RESET enable_indexscan")
        if args.metodo == "ivfflat":
            p2.cur.execute("RESET hnsw.ef_search")
            p2.cur.execute(f"SET ivfflat.probes = {args.probes}")
            cfg = f"probes={args.probes}"
        else:
            p2.cur.execute("RESET ivfflat.probes")
            p2.cur.execute(f"SET hnsw.ef_search = {args.ef_search}")
            cfg = f"ef_search={args.ef_search}"

        recalls = []
        latencias = []
        for i, row in enumerate(consultas, 1):
            q = p2.extraer_embedding(row["path"])
            tiempos_q = []
            res_q = None
            for _ in range(args.repeticiones):
                t0 = time.perf_counter()
                ids_aprox = p2.knn(q, k=args.k, op="<->")
                t1 = time.perf_counter()
                res_q = set(int(x) for x in ids_aprox)
                tiempos_q.append((t1 - t0) * 1000.0)
            latencias.append(statistics.median(tiempos_q))
            recalls.append(len(res_q & exactos[i - 1]) / args.k)
            if i % 10 == 0:
                print(f"  {i}/{len(consultas)}")

        p2.cur.execute("RESET enable_seqscan")
        p2.cur.execute("RESET ivfflat.probes")
        p2.cur.execute("RESET hnsw.ef_search")

        print("Listo.\n[3/3] Resultados")
        print("=" * 80)
        print(f"Método: {args.metodo.upper()} ({cfg})")
        print(f"Recall@{args.k} promedio: {sum(recalls)/len(recalls):.4f} ({(sum(recalls)/len(recalls))*100:.2f}%)")
        print(f"Recall@{args.k} min/max:   {min(recalls):.4f} / {max(recalls):.4f}")
        print(f"Latencia mediana:       {statistics.median(latencias):.3f} ms")
        print(f"Latencia media:         {sum(latencias)/len(latencias):.3f} ms")
        print(f"Latencia min/max:       {min(latencias):.3f} / {max(latencias):.3f} ms")
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
