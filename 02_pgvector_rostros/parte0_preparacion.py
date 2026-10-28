from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import face_recognition
import matplotlib.pyplot as plt
from PIL import Image


def cargar_imagenes(carpeta: str | Path, extension: str = ".jpg") -> list[Path]:
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {carpeta}")
    imagenes = [p for p in carpeta.rglob(f"*{extension.lower()}")]
    imagenes.extend([p for p in carpeta.rglob(f"*{extension.upper()}")])
    return sorted(set(imagenes))


def mostrar_fotos(
    coleccion,
    indices,
    salida: str | Path = "resultados.png",
    mostrar: bool = False,
    titulos: list[str] | None = None,
    columnas: int = 4,
):
    rutas = coleccion.iloc[indices]["path"].tolist()
    personas = coleccion.iloc[indices]["person"].tolist()

    n = len(rutas)
    filas = math.ceil(n / columnas) if n else 1
    fig, ejes = plt.subplots(filas, columnas, figsize=(columnas * 4, filas * 4))
    ejes = ejes.flatten() if hasattr(ejes, "flatten") else [ejes]

    for i, (ruta, persona) in enumerate(zip(rutas, personas)):
        ax = ejes[i]
        try:
            img = Image.open(ruta)
            img_cara = face_recognition.load_image_file(ruta)
            ubicaciones = face_recognition.face_locations(img_cara)
            if ubicaciones:
                top, right, bottom, left = ubicaciones[0]
                margen = 20
                top = max(0, top - margen)
                left = max(0, left - margen)
                bottom = min(img.height, bottom + margen)
                right = min(img.width, right + margen)
                img = img.crop((left, top, right, bottom))
            ax.imshow(img)
            if titulos and i < len(titulos):
                ax.set_title(titulos[i], fontsize=8)
            else:
                ax.set_title(persona, fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center")
        ax.axis("off")

    for j in range(i + 1, len(ejes)):
        ejes[j].axis("off")

    plt.tight_layout()
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida, dpi=150)
    if mostrar:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Muestra una grilla de rostros del dataset LFW")
    parser.add_argument("carpeta", help="Carpeta con imágenes organizadas por persona")
    parser.add_argument("-n", type=int, default=16, help="Número de imágenes a mostrar")
    parser.add_argument("--aleatorio", action="store_true", help="Seleccionar aleatoriamente")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla para selección aleatoria")
    parser.add_argument("--salida", default="grilla_lfw.png", help="Archivo de salida")
    parser.add_argument("--mostrar", action="store_true", help="Mostrar en pantalla")
    args = parser.parse_args()

    import pandas as pd

    imagenes = cargar_imagenes(args.carpeta)
    print(f"Imágenes encontradas: {len(imagenes):,}")
    coleccion = pd.DataFrame(
        [{"person": p.parent.name, "path": str(p)} for p in imagenes]
    )
    print(f"Personas distintas: {coleccion['person'].nunique():,}")
    conteos = coleccion["person"].value_counts()
    print(
        f"Fotos por persona: min={conteos.min()}, mediana={conteos.median():.0f}, max={conteos.max()}"
    )
    print(f"Personas con una sola foto: {(conteos == 1).sum():,}")

    if args.aleatorio:
        random.seed(args.semilla)
        indices = random.sample(range(len(coleccion)), min(args.n, len(coleccion)))
    else:
        indices = list(range(min(args.n, len(coleccion))))

    mostrar_fotos(coleccion, indices, salida=args.salida, mostrar=args.mostrar)
    print(f"Grilla guardada en: {args.salida}")


if __name__ == "__main__":
    main()
