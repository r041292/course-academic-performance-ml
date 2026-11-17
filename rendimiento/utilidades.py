"""Utilidades compartidas: registro de tiempos, escritura de CSV y versiones."""

import json
import logging
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

log = logging.getLogger("rendimiento")


def configurar_registro(nivel=logging.INFO) -> None:
    """Mensajes cortos en consola: hora, nivel y texto."""
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def guardar_csv(df: pd.DataFrame, ruta: Path) -> Path:
    """Formato único para todos los CSV: separador ';', coma decimal, UTF-8 con BOM."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, sep=";", decimal=",", index=False, encoding="utf-8-sig")
    log.info("CSV guardado: %s (%d filas)", ruta.name, len(df))
    return ruta


def leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, sep=";", decimal=",", encoding="utf-8-sig", low_memory=False)


def guardar_json(datos: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class RegistroTiempos:
    """Acumula la duración de cada etapa para exportarla como CSV (punto R4).

    Uso:
        tiempos = RegistroTiempos()
        with tiempos.medir("optuna", asignatura="MAT1101", algoritmo="CAT", tarea="regresion"):
            ...
    """

    def __init__(self):
        self.filas = []

    @contextmanager
    def medir(self, etapa, asignatura=None, algoritmo=None, tarea=None, n_registros=None, **extra):
        inicio = time.perf_counter()
        try:
            yield
        finally:
            self.filas.append({
                "etapa": etapa,
                "asignatura": asignatura,
                "algoritmo": algoritmo,
                "tarea": tarea,
                "n_registros": n_registros,
                "segundos": round(time.perf_counter() - inicio, 3),
                **extra,
            })

    def como_tabla(self) -> pd.DataFrame:
        return pd.DataFrame(self.filas)


def tabla_versiones(usa_gpu: bool) -> pd.DataFrame:
    """Versiones de Python y bibliotecas, para el párrafo de reproducibilidad (punto 19)."""
    import catboost
    import numpy
    import optuna
    import sklearn
    import xgboost

    filas = [
        ("python", platform.python_version()),
        ("pandas", pd.__version__),
        ("numpy", numpy.__version__),
        ("scikit-learn", sklearn.__version__),
        ("xgboost", xgboost.__version__),
        ("catboost", catboost.__version__),
        ("optuna", optuna.__version__),
        ("plataforma", platform.platform()),
        ("gpu_usada", str(usa_gpu)),
    ]
    try:
        import shap
        filas.append(("shap", shap.__version__))
    except ImportError:
        pass
    return pd.DataFrame(filas, columns=["componente", "version"])
