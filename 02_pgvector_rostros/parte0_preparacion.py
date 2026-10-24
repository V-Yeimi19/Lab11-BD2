"""Parte 0: carga y observación del dataset LFW (Laboratorio 11 - pgvector).

Escanea una carpeta local con fotos LFW, arma la colección ``person``/``path``
y permite observar una grilla de fotos.

Uso:
    python 02_pgvector_rostros/parte0_preparacion.py <ruta_lfw>
        [--n 16] [--posiciones 0 2 5] [--aleatorio] [--semilla 42]
        [--salida grid.png] [--mostrar]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

# Backend sin pantalla cuando solo se guarda el PNG; si se pide --mostrar se
# deja el backend interactivo por defecto (macosx/TkAgg).
if "--mostrar" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def cargar_coleccion(ruta_dataset: str) -> pd.DataFrame:
    """Escanea ``ruta_dataset/**/*.jpg`` y devuelve un DataFrame person/path."""
    ruta = os.path.abspath(os.path.expanduser(ruta_dataset))
    if not os.path.isdir(ruta):
        raise NotADirectoryError(f"No existe la carpeta del dataset: {ruta}")

    registros = [
        {
            "person": os.path.basename(os.path.dirname(archivo)),
            "path": archivo,
        }
        for archivo in glob.iglob(os.path.join(ruta, "**", "*.jpg"), recursive=True)
    ]
    return pd.DataFrame(registros, columns=["person", "path"])


def resumen(coleccion: pd.DataFrame) -> None:
    """Imprime cantidad de fotos, personas y distribución fotos/persona."""
    total = len(coleccion)
    personas = coleccion["person"].nunique()
    print(f"Imágenes encontradas: {total:,}")
    print(f"Personas distintas: {personas:,}")
    if total:
        por_persona = coleccion.groupby("person").size()
        print(
            "Fotos por persona: "
            f"min={por_persona.min()}, "
            f"mediana={por_persona.median():.0f}, "
            f"max={por_persona.max()}"
        )
        print(f"Personas con una sola foto: {(por_persona == 1).sum():,}")


def seleccionar_posiciones(
    coleccion: pd.DataFrame,
    n: int,
    posiciones: list[int] | None = None,
    aleatorio: bool = False,
    semilla: int | None = None,
) -> list[int]:
    """Devuelve los índices de las fotos a mostrar."""
    total = len(coleccion)
    if not total:
        raise ValueError("No hay imágenes para mostrar.")
    if posiciones:
        invalidas = [p for p in posiciones if not 0 <= p < total]
        if invalidas:
            raise IndexError(
                f"Posiciones fuera de rango (0..{total - 1}): {invalidas}"
            )
        return posiciones
    if aleatorio:
        muestra = coleccion.sample(n=min(n, total), random_state=semilla)
        return muestra.index.tolist()
    return list(range(min(n, total)))


def mostrar_fotos(
    coleccion: pd.DataFrame,
    posiciones: list[int],
    salida: str | None = None,
    mostrar: bool = False,
) -> None:
    """Dibuja una grilla con las fotos indicadas y la guarda y/o muestra."""
    if not posiciones:
        raise ValueError("Debe indicar al menos una posición.")

    columnas = 4
    filas = (len(posiciones) + columnas - 1) // columnas
    fig, ejes = plt.subplots(filas, columnas, figsize=(4 * columnas, 4 * filas))
    ejes = ejes.flatten() if hasattr(ejes, "flatten") else [ejes]

    for eje, idx in zip(ejes, posiciones):
        imagen = plt.imread(coleccion["path"].iloc[idx])
        eje.imshow(imagen)
        eje.set_title(
            f"{coleccion['person'].iloc[idx]}\n{imagen.shape}", fontsize=8
        )
        eje.axis("off")
    for eje in ejes[len(posiciones) :]:
        eje.axis("off")

    fig.tight_layout()
    if salida:
        fig.savefig(salida, dpi=150)
        print(f"Grilla guardada en: {salida}")
    if mostrar:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parte 0: carga el dataset LFW y muestra fotos para observarlo."
        )
    )
    parser.add_argument("ruta_dataset", help="Carpeta raíz del dataset LFW")
    parser.add_argument(
        "--n", type=int, default=16, help="Cuántas fotos mostrar (default: 16)"
    )
    parser.add_argument(
        "--posiciones",
        type=int,
        nargs="+",
        help="Índices exactos a mostrar (alternativa a --n)",
    )
    parser.add_argument(
        "--aleatorio", action="store_true", help="Muestrear fotos al azar"
    )
    parser.add_argument(
        "--semilla", type=int, help="Semilla para --aleatorio"
    )
    parser.add_argument("--salida", help="Guardar la grilla en un PNG")
    parser.add_argument(
        "--mostrar", action="store_true", help="Mostrar la grilla en pantalla"
    )
    args = parser.parse_args()

    coleccion = cargar_coleccion(args.ruta_dataset)
    resumen(coleccion)

    # Solo se renderiza si se pidió guardar o mostrar.
    if args.salida or args.mostrar:
        posiciones = seleccionar_posiciones(
            coleccion, args.n, args.posiciones, args.aleatorio, args.semilla
        )
        mostrar_fotos(coleccion, posiciones, args.salida, args.mostrar)


if __name__ == "__main__":
    try:
        main()
    except (NotADirectoryError, IndexError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
