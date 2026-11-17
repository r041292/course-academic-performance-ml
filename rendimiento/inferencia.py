"""Inferencia en el periodo reservado (2025-30) con los modelos finales guardados.

Produce una tabla con una fila por registro: identificación, resultado real y, por
algoritmo, la nota estimada, la probabilidad y la predicción de retiro y la clase integrada.
Regla de integración (sistema de tres clases):
    retiro (2)      si el clasificador predice retiro;
    aprobación (0)  si no, y la nota estimada es >= 3,00;
    pérdida (1)     si no, y la nota estimada es < 3,00.
En asignaturas sin clasificador de retiro decide solo la nota estimada.
"""

import joblib
import numpy as np
import pandas as pd

from .entrenamiento import cargar_definicion, ruta_definicion, ruta_modelo
from .preprocesamiento import predecir_nota, predecir_retiro
from .utilidades import log
from .variables import OBJETIVO_NOTA, OBJETIVO_RETIRO, construir_tabla, matriz_predictores

APROBACION, PERDIDA, RETIRO = 0, 1, 2
NOMBRES_CLASES = {APROBACION: "Aprobación", PERDIDA: "Pérdida", RETIRO: "Retiro"}


def clase_real(nota: pd.Series, retiro: pd.Series, umbral: float) -> pd.Series:
    nota = pd.to_numeric(nota, errors="coerce")
    clase = pd.Series(np.nan, index=nota.index)
    clase[retiro == 1] = RETIRO
    clase[(retiro == 0) & (nota >= umbral)] = APROBACION
    clase[(retiro == 0) & (nota >= 0) & (nota < umbral)] = PERDIDA
    return clase


def clase_integrada(nota_estimada, retiro_predicho, umbral: float) -> np.ndarray:
    clase = np.where(np.asarray(nota_estimada) >= umbral, APROBACION, PERDIDA)
    retiro = np.nan_to_num(np.asarray(retiro_predicho, dtype=float), nan=0) == 1
    return np.where(retiro, RETIRO, clase)


def predecir_asignatura(prueba: pd.DataFrame, asignatura: str, cfg, tiempos) -> pd.DataFrame:
    definicion = cargar_definicion(cfg, asignatura)
    tabla = construir_tabla(prueba, definicion, eliminar_filas=False)
    X = matriz_predictores(tabla, definicion)

    salida = tabla[[c for c in list(cfg.columnas_id) + ["Estado asignatura"] if c in tabla]].copy()
    salida["nota_real"] = pd.to_numeric(tabla[OBJETIVO_NOTA], errors="coerce")
    salida["retiro_real"] = tabla[OBJETIVO_RETIRO].astype(int)
    salida["clase_real"] = clase_real(salida["nota_real"], salida["retiro_real"], cfg.umbral_aprobacion)
    salida["regimen_promedio"] = definicion.regimen_promedio

    for algoritmo in cfg.algoritmos:
        with tiempos.medir("inferencia_2025_30", asignatura, algoritmo, n_registros=len(X)):
            regresor = joblib.load(ruta_modelo(cfg, algoritmo, "regresion", asignatura))
            nota = predecir_nota(regresor, X)
            ruta_clf = ruta_modelo(cfg, algoritmo, "clasificacion", asignatura)
            if ruta_clf.exists():
                probabilidad, retiro = predecir_retiro(joblib.load(ruta_clf), X)
            else:
                probabilidad, retiro = np.full(len(X), np.nan), np.full(len(X), np.nan)
        salida[f"nota_estimada_{algoritmo}"] = np.round(nota, 2)
        salida[f"prob_retiro_{algoritmo}"] = probabilidad
        salida[f"retiro_predicho_{algoritmo}"] = retiro
        salida[f"clase_{algoritmo}"] = clase_integrada(nota, retiro, cfg.umbral_aprobacion)

    salida["tiene_clasificador"] = ruta_modelo(cfg, cfg.algoritmos[0], "clasificacion", asignatura).exists()
    # Variables con las que se hizo la inferencia (tal como entran al modelo)
    return pd.concat([salida, X.set_index(salida.index)], axis=1)


def predecir_periodo(prueba_completa: pd.DataFrame, asignaturas: list, cfg, tiempos) -> pd.DataFrame:
    """Predicciones de todas las asignaturas con modelo y registros en el periodo de prueba."""
    partes = []
    for asignatura in asignaturas:
        registros = prueba_completa[prueba_completa["Cod materia curso"] == asignatura]
        if registros.empty or not ruta_definicion(cfg, asignatura).exists():
            continue
        try:
            partes.append(predecir_asignatura(registros, asignatura, cfg, tiempos))
        except Exception as error:  # una asignatura con problemas no detiene la corrida
            log.warning("Inferencia fallida en %s: %s", asignatura, error)
    predicciones = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
    return predicciones[predicciones["clase_real"].notna()].reset_index(drop=True)
