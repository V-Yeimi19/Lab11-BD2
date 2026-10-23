
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Permite guardar la figura sin necesidad de un display.
import matplotlib.pyplot as plt

# Permite ejecutar el archivo directamente con `python src/search_rango.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exploracion import construir_dataset

ED = lambda x, y: np.sqrt(np.sum((x - y) ** 2))

# DATA contiene todos los datos; la ultima columna es la clase (categoria).
DATA = construir_dataset()

# Las distancias se calculan solo sobre las variables predictoras.
FEATURES = DATA.iloc[:, :-1]

N = 5000


def distancias_N(N: int = 5000) -> np.ndarray:
    """Distancias euclidianas entre pares consecutivos de filas muestreadas."""
    data = FEATURES.sample(n=2 * N)
    dist_N = np.zeros(N)

    for i in range(N):
        dist_N[i] = ED(data.iloc[i, :].to_numpy(), data.iloc[i + 1, :].to_numpy())

    return dist_N


def radios(N: int = 5000, dist_N=None) -> np.ndarray:
    """Cinco radios candidatos basados en percentiles de las distancias.

    La distribucion de distancias esta sesgada a la derecha, por lo que
    `media +/- 2*desv` produce radios negativos. Se usan los percentiles
    10, 25, 50, 75 y 90 para obtener radios positivos y representativos.
    """
    if dist_N is None:
        dist_N = distancias_N(N=N)
    return np.percentile(dist_N, [10, 25, 50, 75, 90])


def search_range_lineal(Q, coleccion, r):
    """Busqueda por rango lineal.

    Recorre la coleccion una sola vez y devuelve los indices de las filas cuya
    distancia euclidiana a Q es menor o igual que el radio r. La ultima columna
    (Class) se excluye del calculo de la distancia.
    """
    q = Q.iloc[:-1].to_numpy(dtype=float)
    feats = coleccion.iloc[:, :-1].to_numpy(dtype=float)
    dists = np.sqrt(((feats - q) ** 2).sum(axis=1))
    return coleccion.index[dists <= r]


def graficar_analisis_distancias(dist_N, rs, salida):
    """Grafica las 5000 distancias muestreadas y los radios seleccionados."""
    fig, (ax_hist, ax_disp) = plt.subplots(1, 2, figsize=(15, 6))
    colores = matplotlib.colormaps["viridis"](np.linspace(0.15, 0.85, len(rs)))

    # Histograma de las 5000 distancias.
    ax_hist.hist(dist_N, bins=60, color="#4c72b0", edgecolor="white", alpha=0.85)
    for i, (r, color) in enumerate(zip(rs, colores), 1):
        ax_hist.axvline(
            r, color=color, linestyle="--", linewidth=2, label=f"r{i} = {r:.0f}"
        )
    ax_hist.set_title("Distribucion de las 5000 distancias")
    ax_hist.set_xlabel("Distancia euclidiana")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.legend(title="Radios (percentiles)")
    ax_hist.grid(axis="y", alpha=0.3, linestyle="--")

    # Dispersion de cada una de las 5000 muestras aleatorias.
    ax_disp.scatter(
        np.arange(len(dist_N)), dist_N, s=8, alpha=0.4, color="#4c72b0"
    )
    for i, (r, color) in enumerate(zip(rs, colores), 1):
        ax_disp.axhline(
            r, color=color, linestyle="--", linewidth=2, label=f"r{i} = {r:.0f}"
        )
    ax_disp.set_title("Distancias de las 5000 muestras aleatorias")
    ax_disp.set_xlabel("Indice de la muestra")
    ax_disp.set_ylabel("Distancia euclidiana")
    ax_disp.legend(title="Radios (percentiles)")
    ax_disp.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def precision_rango(Q, coleccion, r):
    """Precision de una ejecucion del search.

    Devuelve (precision, correctos, total), donde precision es la fraccion de
    vecinos recuperados que comparten la clase de Q.
    """
    idx = search_range_lineal(Q, coleccion, r)
    vecinos = coleccion.loc[idx]
    total = len(vecinos)
    if total == 0:
        return np.nan, 0, 0
    correctos = int((vecinos["Class"] == Q["Class"]).sum())
    return correctos / total, correctos, total


if __name__ == "__main__":
    qs = [DATA.iloc[15], DATA.iloc[2166], DATA.iloc[4768]]
    posiciones = [15, 2166, 4768]

    dist_N = distancias_N(N)
    rs = radios(N, dist_N=dist_N)

    # Se quitan los elementos (queries) antes de aplicar el search (no cambiar).
    _data = DATA.drop(DATA.index[[15, 2166, 4768]])

    # ------------------------------------------------------------------
    # 1. Analisis y seleccion de radios
    # ------------------------------------------------------------------
    print("=== ANALISIS Y SELECCION DE RADIOS ===")
    print(f"Distancia euclidiana entre pares de puntos (N = {N}):")
    print(f"  media = {dist_N.mean():.4f}")
    print(f"  desv. = {dist_N.std():.4f}")
    print(f"  min   = {dist_N.min():.4f}")
    print(f"  max   = {dist_N.max():.4f}")
    print("\nRadios seleccionados (percentiles 10, 25, 50, 75, 90):")
    for i, r in enumerate(rs, 1):
        print(f"  r{i} = {r:.4f}")

    salida_distancias = ROOT / "analisis_distancias.png"
    graficar_analisis_distancias(dist_N, rs, salida_distancias)
    print(f"\nGrafica de analisis de distancias guardada en: {salida_distancias}")

    # ------------------------------------------------------------------
    # 2. Search para cada query y radio: precision
    # ------------------------------------------------------------------
    detalle = []
    for p, q in zip(posiciones, qs):
        etiqueta = f"Q{p}"
        for i, r in enumerate(rs, 1):
            prec, correctos, total = precision_rango(q, _data, r)
            detalle.append(
                {
                    "query": etiqueta,
                    "clase": q["Class"],
                    "radio": f"r{i}={r:.4f}",
                    "precision": prec,
                    "vecinos": total,
                    "correctos": correctos,
                }
            )

    detalle_df = pd.DataFrame(detalle)
    orden_queries = [f"Q{p}" for p in posiciones]
    etiquetas = [f"Q{p} ({q['Class']})" for p, q in zip(posiciones, qs)]

    # Tabla de precisiones: filas = queries, columnas = radios.
    tabla = detalle_df.pivot(index="query", columns="radio", values="precision")
    tabla = tabla.reindex(orden_queries)
    tabla.index = etiquetas

    print("\n=== TABLA DE PRECISIONES (query x radio) ===")
    print(tabla.round(4).to_string(na_rep="-"))

    # Tabla auxiliar: cantidad de vecinos recuperados.
    vecinos_tabla = detalle_df.pivot(index="query", columns="radio", values="vecinos")
    vecinos_tabla = vecinos_tabla.reindex(orden_queries)
    vecinos_tabla.index = etiquetas

    print("\n=== VECINOS RECUPERADOS (query x radio) ===")
    print(vecinos_tabla.astype("Int64").to_string(na_rep="-"))

    # ------------------------------------------------------------------
    # 3. Grafica de precision
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(rs))
    ancho = 0.8 / len(posiciones)
    colores = matplotlib.colormaps["viridis"](
        np.linspace(0.15, 0.85, len(posiciones))
    )

    for j, (p, q) in enumerate(zip(posiciones, qs)):
        serie = detalle_df.loc[
            detalle_df["query"] == f"Q{p}", "precision"
        ].to_numpy()
        alturas = np.nan_to_num(serie, nan=0.0)
        offset = (j - (len(posiciones) - 1) / 2) * ancho

        barras = ax.bar(
            x + offset,
            alturas,
            width=ancho,
            color=colores[j],
            label=f"Q{p} ({q['Class']})",
        )

        for barra, valor in zip(barras, serie):
            texto = "-" if np.isnan(valor) else f"{valor:.2f}"
            ax.annotate(
                texto,
                (barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.01),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"r{i}\n{r:.3f}" for i, r in enumerate(rs, 1)])
    ax.set_xlabel("Radio")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.12)
    ax.set_title("Precision de la busqueda por rango lineal (query x radio)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(title="Query (clase)", loc="best")
    fig.tight_layout()

    salida = ROOT / "precision_rango.png"
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f"\nGrafica guardada en: {salida}")
