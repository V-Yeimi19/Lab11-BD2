from pathlib import Path
import heapq
import sys
from data.dataset import X as FEATURES, y as TARGET
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))



RUTA_RESULTADOS = BASE_DIR / "result" / "knn_resultados.csv"
COLUMNA_CLASE = "Class"
CONSULTAS = [15, 2166, 4768]
VALORES_K = [2, 4, 8, 16, 32]


def cargar_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Valida y convierte el dataset descargado desde UCI."""
    if len(FEATURES) != len(TARGET):
        raise ValueError(
            f"X e y tienen distinta cantidad de filas: "
            f"{len(FEATURES)} y {len(TARGET)}."
        )
    if TARGET.shape[1] != 1:
        raise ValueError("El objetivo debe contener exactamente una columna.")
    if FEATURES.empty:
        raise ValueError("El dataset no contiene filas.")
    if not all(pd.api.types.is_numeric_dtype(tipo) for tipo in FEATURES.dtypes):
        raise ValueError("Todas las características deben ser numéricas.")
    if FEATURES.isna().any().any() or TARGET.isna().any().any():
        raise ValueError("El dataset contiene valores nulos.")

    X = FEATURES.to_numpy(dtype=float)
    y = TARGET.iloc[:, 0].to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Las características contienen valores no finitos.")

    df = FEATURES.copy()
    df[COLUMNA_CLASE] = TARGET.iloc[:, 0].to_numpy()
    print(f"Dataset cargado: {len(df):,} objetos")
    print(f"Número de características: {X.shape[1]}")
    print(f"Categorías: {sorted(pd.unique(y))}")
    print(f"Filas duplicadas: {df.duplicated().sum():,}")
    return df, X, y


def estandarizar_caracteristicas(X: np.ndarray) -> np.ndarray:
    """Pone cada característica en escala comparable usando z-score."""
    medias = X.mean(axis=0)
    desviaciones = X.std(axis=0)
    if not np.isfinite(desviaciones).all() or np.any(desviaciones == 0):
        raise ValueError("No se pueden estandarizar columnas constantes o no finitas.")
    return (X - medias) / desviaciones


def distancia_euclidiana(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def knn_max_heap(
    X: np.ndarray,
    y: np.ndarray,
    indice_consulta: int,
    k: int,) -> tuple[list[int], list[float], int, float]:
    """Devuelve los k vecinos más cercanos, sin incluir la consulta."""
    n = len(X)
    if len(y) != n:
        raise ValueError("X e y deben tener la misma cantidad de filas.")
    if not 0 <= indice_consulta < n:
        raise IndexError("El índice de consulta no es válido.")
    if not 1 <= k < n:
        raise ValueError("k debe ser positivo y menor que el tamaño del dataset.")

    heap: list[tuple[float, int, int]] = []
    consulta = X[indice_consulta]

    for indice, candidato in enumerate(X):
        if indice == indice_consulta:
            continue

        distancia = distancia_euclidiana(consulta, candidato)
        entrada = (-distancia, -indice, indice)
        if len(heap) < k:
            heapq.heappush(heap, entrada)
        elif entrada > heap[0]:
            heapq.heapreplace(heap, entrada)

    vecinos = sorted(
        [(-distancia_negativa, indice) for distancia_negativa, _, indice in heap],
        key=lambda elemento: (elemento[0], elemento[1]),
    )
    indices = [indice for _, indice in vecinos]
    distancias = [distancia for distancia, _ in vecinos]
    relevantes = sum(y[indice] == y[indice_consulta] for indice in indices)
    precision = relevantes / k
    return indices, distancias, relevantes, precision


def main() -> None:
    df, X, y = cargar_dataset()
    X = estandarizar_caracteristicas(X)

    if not all(1 <= consulta <= len(df) for consulta in CONSULTAS):
        raise ValueError("Todas las consultas deben referirse a filas existentes.")
    if any(k >= len(df) for k in VALORES_K):
        raise ValueError("Cada valor de k debe ser menor que el tamaño del dataset.")

    resultados = []
    for numero_consulta in CONSULTAS:
        indice_consulta = numero_consulta - 1
        print("\n" + "=" * 55)
        print(f"Consulta Q{numero_consulta} | Categoría: {y[indice_consulta]}")
        print("=" * 55)

        for k in VALORES_K:
            indices, distancias, relevantes, precision = knn_max_heap(
                X, y, indice_consulta, k
            )
            print(
                f"k={k:2d} | Relevantes={relevantes:2d}/{k:2d} | "
                f"Precisión={precision:.4f}"
            )
            for posicion, (indice, distancia) in enumerate(
                zip(indices, distancias), start=1
            ):
                print(
                    f"    Vecino {posicion:2d}: fila={indice + 1}, "
                    f"clase={y[indice]}, distancia={distancia:.4f}"
                )
            resultados.append(
                {
                    "consulta": f"Q{numero_consulta}",
                    "k": k,
                    "categoria_consulta": y[indice_consulta],
                    "relevantes_recuperados": relevantes,
                    "total_recuperados": k,
                    "precision": precision,
                }
            )

    tabla = pd.DataFrame(resultados)
    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_RESULTADOS, index=False)
    print("\nRESUMEN DE PRECISIONES")
    print(tabla.pivot(index="k", columns="consulta", values="precision").to_string())
    print(f"\nResultados guardados en: {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
