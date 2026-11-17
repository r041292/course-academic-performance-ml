"""Entrenamiento por asignatura: validación cruzada con Optuna y modelo final.

Por cada asignatura y algoritmo:
  - Regresión de la nota (registros no retirados).
  - Clasificación de retiro, solo si el retiro supera el 3 % y hay al menos 5 registros
    de cada clase (uno por fold).
El modelo final se ajusta con TODOS los datos de desarrollo (periodos <= periodo_max_desarrollo)
y se guarda como un pipeline completo (joblib), junto con la definición de la asignatura.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .metricas import metricas_regresion
from .optimizacion import buscar_hiperparametros, crear_folds, evaluar_en_fold
from .preprocesamiento import ImputadorPromedio, ajustar_pipeline, construir_pipeline, predecir_nota
from .utilidades import guardar_json, log
from .variables import (OBJETIVO_NOTA, OBJETIVO_RETIRO, DefinicionAsignatura, construir_tabla,
                        definir_asignatura, matriz_predictores)

TAREAS = ("regresion", "clasificacion")


# ---------------------------------------------------------------------------
# Rutas de modelos
# ---------------------------------------------------------------------------
def ruta_modelo(cfg, algoritmo, tarea, asignatura):
    return cfg.ruta_modelos / algoritmo / tarea / f"{asignatura}.joblib"


def ruta_definicion(cfg, asignatura):
    return cfg.ruta_modelos / "definiciones" / f"{asignatura}.json"


# ---------------------------------------------------------------------------
# Datos por tarea
# ---------------------------------------------------------------------------
def filas_regresion(tabla: pd.DataFrame) -> pd.Index:
    """Registros con nota final (no retirados)."""
    filas = (tabla[OBJETIVO_RETIRO] == 0) & (pd.to_numeric(tabla[OBJETIVO_NOTA], errors="coerce") >= 0)
    return tabla.index[filas]


def datos_regresion(tabla: pd.DataFrame, definicion):
    filas = filas_regresion(tabla)
    X = matriz_predictores(tabla.loc[filas], definicion).reset_index(drop=True)
    y = pd.to_numeric(tabla.loc[filas, OBJETIVO_NOTA]).reset_index(drop=True)
    return X, y


def datos_clasificacion(tabla: pd.DataFrame, definicion):
    X = matriz_predictores(tabla, definicion).reset_index(drop=True)
    y = tabla[OBJETIVO_RETIRO].astype(int).reset_index(drop=True)
    return X, y


def parametros_efectivos(pipeline) -> dict:
    """Hiperparámetros con que quedó el modelo final (incluye los fijos y la ponderación de clases)."""
    parametros = pipeline[-1].get_params()
    return {k: v for k, v in parametros.items()
            if isinstance(v, (int, float, str, bool)) or (isinstance(v, list) and len(v) < 20)}


def tabla_registros_desarrollo(tabla: pd.DataFrame, definicion, cfg) -> pd.DataFrame:
    """Registros de desarrollo con sus variables (tal como entran al modelo) y su resultado real.
    Las predicciones fuera de muestra se agregan después, por algoritmo."""
    columnas_id = [c for c in cfg.columnas_id if c in tabla]
    registros = tabla[columnas_id].copy()
    registros["nota_real"] = pd.to_numeric(tabla[OBJETIVO_NOTA], errors="coerce")
    registros["retiro_real"] = tabla[OBJETIVO_RETIRO].astype(int)
    registros["regimen_promedio"] = definicion.regimen_promedio
    return pd.concat([registros, matriz_predictores(tabla, definicion)], axis=1)


def clasificador_viable(y: pd.Series, cfg) -> tuple[bool, str]:
    """Regla de la propuesta: más del 3 % de retiros y al menos un retiro por fold."""
    proporcion = y.mean() if len(y) else 0
    if proporcion <= cfg.umbral_retiro_clasificador:
        return False, f"retiro {proporcion:.1%} <= {cfg.umbral_retiro_clasificador:.0%}"
    if y.value_counts().min() < cfg.n_folds:
        return False, f"menos de {cfg.n_folds} registros en alguna clase"
    return True, ""


# ---------------------------------------------------------------------------
# Entrenamiento de una asignatura
# ---------------------------------------------------------------------------
def entrenar_asignatura(desarrollo: pd.DataFrame, asignatura: str, nombres: pd.DataFrame, cfg, tiempos) -> dict:
    """Entrena los modelos de una asignatura. Devuelve tablas de validación cruzada y de Optuna."""
    definicion = definir_asignatura(desarrollo, asignatura, nombres, cfg)
    tabla = construir_tabla(desarrollo, definicion, eliminar_filas=True).reset_index(drop=True)
    registros = tabla_registros_desarrollo(tabla, definicion, cfg)
    posiciones = {"regresion": filas_regresion(tabla), "clasificacion": tabla.index}

    filas_cv, filas_optuna, filas_estado = [], [], []
    for tarea in TAREAS:
        X, y = (datos_regresion if tarea == "regresion" else datos_clasificacion)(tabla, definicion)
        if tarea == "regresion" and len(y) < cfg.n_folds:
            filas_estado.append({"asignatura": asignatura, "tarea": tarea, "entrenado": False,
                                 "motivo": "registros insuficientes"})
            continue
        if tarea == "clasificacion":
            viable, motivo = clasificador_viable(y, cfg)
            filas_estado.append({"asignatura": asignatura, "tarea": tarea, "entrenado": viable, "motivo": motivo,
                                 "proporcion_retiro": float(y.mean()), "n_registros": len(y)})
            if not viable:
                continue
        else:
            filas_estado.append({"asignatura": asignatura, "tarea": tarea, "entrenado": True, "motivo": "",
                                 "n_registros": len(y)})

        folds = crear_folds(y, tarea, cfg.n_folds, cfg.semilla)   # mismos folds para los tres algoritmos
        for algoritmo in cfg.algoritmos:
            with tiempos.medir("busqueda_optuna_y_cv", asignatura, algoritmo, tarea, len(y)):
                parametros, metricas_folds, predicciones_folds, resumen = buscar_hiperparametros(
                    X, y, folds, algoritmo, tarea, definicion, cfg)
            _agregar_predicciones_fuera_de_muestra(registros, posiciones[tarea], predicciones_folds, algoritmo, tarea)
            for numero, metricas in enumerate(metricas_folds, start=1):
                filas_cv.append({"asignatura": asignatura, "algoritmo": algoritmo, "tarea": tarea,
                                 "fold": numero, **metricas})
            with tiempos.medir("ajuste_modelo_final", asignatura, algoritmo, tarea, len(y)):
                pipeline = construir_pipeline(algoritmo, tarea, definicion, parametros, cfg)
                ajustar_pipeline(pipeline, X, y, algoritmo, tarea)
            filas_optuna.append({"asignatura": asignatura, "algoritmo": algoritmo, "tarea": tarea,
                                 "n_registros": len(y), "n_predictores": len(definicion.predictores),
                                 "regimen_promedio": definicion.regimen_promedio, **resumen,
                                 "parametros_optuna": json.dumps(parametros, ensure_ascii=False),
                                 "parametros_efectivos": json.dumps(parametros_efectivos(pipeline), ensure_ascii=False,
                                                                    default=str)})
            ruta = ruta_modelo(cfg, algoritmo, tarea, asignatura)
            ruta.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, ruta)

    guardar_json({**definicion.a_dict(), "n_registros_desarrollo": len(tabla)}, ruta_definicion(cfg, asignatura))
    return {"cv": pd.DataFrame(filas_cv), "optuna": pd.DataFrame(filas_optuna), "estado": pd.DataFrame(filas_estado),
            "registros": registros}


def _agregar_predicciones_fuera_de_muestra(registros, posiciones, predicciones_folds, algoritmo, tarea):
    """Coloca en cada registro la predicción hecha cuando ese registro estaba en el fold de validación."""
    for fold, prediccion in enumerate(predicciones_folds, start=1):
        filas = posiciones[prediccion["indices"]]
        if tarea == "regresion":
            registros.loc[filas, f"nota_oof_{algoritmo}"] = prediccion["nota"]
            registros.loc[filas, "fold_regresion"] = fold
        else:
            registros.loc[filas, f"prob_retiro_oof_{algoritmo}"] = prediccion["probabilidad"]
            registros.loc[filas, f"retiro_oof_{algoritmo}"] = prediccion["clase"]
            registros.loc[filas, "fold_clasificacion"] = fold


def cargar_definicion(cfg, asignatura) -> DefinicionAsignatura:
    import json
    datos = json.loads(ruta_definicion(cfg, asignatura).read_text(encoding="utf-8"))
    datos.pop("n_registros_desarrollo", None)
    datos["prerrequisitos"] = {k: tuple(v) for k, v in datos["prerrequisitos"].items()}
    return DefinicionAsignatura.desde_dict(datos)


# ---------------------------------------------------------------------------
# Línea base: regresión lineal en las 8 asignaturas de referencia (Tablas 6 y 7)
# ---------------------------------------------------------------------------
def pipeline_lineal(definicion, cfg) -> Pipeline:
    """Regresión lineal: imputación del promedio (igual que los demás), mediana en numéricas,
    categoría más frecuente + one-hot en categóricas."""
    columnas = list(definicion.predictores) + (["promedio_imputado"] if definicion.imputa_promedio else [])
    categoricas = list(definicion.categoricas)
    numericas = [c for c in columnas if c not in categoricas]
    preparacion = ColumnTransformer([
        ("categoricas", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                      OneHotEncoder(handle_unknown="ignore")), categoricas),
        ("numericas", SimpleImputer(strategy="median", keep_empty_features=True), numericas),
    ])
    return Pipeline([
        ("promedio", ImputadorPromedio(activo=definicion.imputa_promedio, escala=cfg.escala_promedio)),
        ("preparacion", preparacion),
        ("modelo", LinearRegression()),
    ])


def linea_base(desarrollo: pd.DataFrame, prueba: pd.DataFrame, asignatura: str, nombres, cfg) -> pd.DataFrame:
    """Métricas de la regresión lineal con los mismos folds que los modelos y en 2025-30."""
    definicion = definir_asignatura(desarrollo, asignatura, nombres, cfg)
    X, y = datos_regresion(construir_tabla(desarrollo, definicion, eliminar_filas=True), definicion)
    folds = crear_folds(y, "regresion", cfg.n_folds, cfg.semilla)
    filas = []
    for numero, (ent, val) in enumerate(folds, start=1):
        metricas = evaluar_en_fold(pipeline_lineal(definicion, cfg), X, y, ent, val, "LIN", "regresion")
        filas.append({"asignatura": asignatura, "conjunto": "cv", "fold": numero, **metricas})

    X_prueba, y_prueba = datos_regresion(construir_tabla(prueba, definicion, eliminar_filas=False), definicion)
    if len(y_prueba):
        modelo = pipeline_lineal(definicion, cfg).fit(X, y)
        filas.append({"asignatura": asignatura, "conjunto": "prueba_2025_30", "fold": np.nan,
                      **metricas_regresion(y_prueba, predecir_nota(modelo, X_prueba))})
    return pd.DataFrame(filas)
