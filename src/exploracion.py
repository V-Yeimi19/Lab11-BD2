
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Permite ejecutar el archivo directamente con `python src/exploracion.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import X, y


def construir_dataset() -> pd.DataFrame:
    """Combina las variables predictoras y el objetivo en un solo DataFrame."""
    if len(X) != len(y):
        raise ValueError(
            f"X e y tienen distinta cantidad de filas: {len(X)} y {len(y)}."
        )

    objetivo = y.iloc[:, 0].rename("Class")
    return X.copy().assign(Class=objetivo)


def explorar(df: pd.DataFrame) -> None:
    """Imprime un informe breve de estructura, calidad y distribución del dataset."""
    objetivo = "Class"
    predictoras = df.drop(columns=objetivo)

    print("=== RESUMEN DEL DATASET ===")
    print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print("\nPrimeras filas:")
    print(df.head().to_string(index=False))

    print("\n=== ESTRUCTURA ===")
    estructura = pd.DataFrame(
        {
            "tipo": df.dtypes.astype(str),
            "no_nulos": df.notna().sum(),
            "nulos": df.isna().sum(),
            "unicos": df.nunique(dropna=False),
        }
    )
    print(estructura.to_string())

    print("\n=== CALIDAD ===")
    print(f"Filas duplicadas: {df.duplicated().sum():,}")
    no_numericas = predictoras.select_dtypes(exclude="number").columns.tolist()
    print(f"Columnas no numéricas en variables predictoras: {len(no_numericas)}")
    if no_numericas:
        print(f"  Revisar: {no_numericas}")

    numeric = predictoras.select_dtypes(include="number")
    valores_infinitos = (~np.isfinite(numeric)).sum().sum()
    print(f"Valores infinitos en variables predictoras: {valores_infinitos:,}")
    if valores_infinitos:
        print("Advertencia: hay valores infinitos que deben revisarse.")
    else:
        print("Las variables predictoras contienen valores finitos.")

    print("\n=== ESTADÍSTICOS NUMÉRICOS ===")
    print(predictoras.describe().T.to_string())

    print("\n=== DISTRIBUCIÓN DEL OBJETIVO ===")
    clases = df[objetivo].value_counts(dropna=False).rename("cantidad").to_frame()
    clases["porcentaje"] = (clases["cantidad"] / len(df) * 100).round(2)
    print(clases.to_string())


if __name__ == "__main__":
    explorar(construir_dataset())
