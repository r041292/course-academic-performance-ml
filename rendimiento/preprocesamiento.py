"""Pipelines de scikit-learn por algoritmo.

Todo lo que "aprende" de los datos (escala de la imputación del promedio, categorías
conocidas, codificación ordinal) vive dentro del pipeline. Así, en la validación cruzada
se ajusta solo con el fold de entrenamiento, y el modelo final guarda lo aprendido para
aplicarlo sin cambios al periodo de prueba.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

from .variables import COL_INDICADOR_IMPUTADO, COL_PGA, COL_SABER11_GLOBAL

FALTANTE = "__FALTANTE__"


# ---------------------------------------------------------------------------
# Transformadores propios
# ---------------------------------------------------------------------------
class ImputadorPromedio(BaseEstimator, TransformerMixin):
    """Imputa el promedio acumulado (PGA) faltante a partir del puntaje global de Saber 11.

    El puntaje se lleva a la escala del promedio con un escalamiento Min-Max:
        pga = minimo_escala + (saber - min) / (max - min) * (maximo_escala - minimo_escala)
    El mínimo y el máximo de Saber 11 se aprenden en `fit` (solo con datos de entrenamiento).
    Además agrega la variable `promedio_imputado` (1 si el PGA fue imputado).
    Si `activo=False`, no hace nada (asignaturas que no imputan).
    """

    def __init__(self, activo=False, escala=(1.5, 5.0), conservar_saber_global=False):
        self.activo = activo
        self.escala = escala
        self.conservar_saber_global = conservar_saber_global

    def fit(self, X, y=None):
        if self.activo:
            saber = pd.to_numeric(X[COL_SABER11_GLOBAL], errors="coerce")
            self.minimo_ = float(saber.min())
            self.maximo_ = float(saber.max())
        self.ajustado_ = True
        return self

    def transform(self, X):
        X = X.copy()
        if not self.activo:
            return X
        bajo, alto = self.escala
        saber = pd.to_numeric(X[COL_SABER11_GLOBAL], errors="coerce")
        rango = self.maximo_ - self.minimo_
        estimado = bajo + (saber - self.minimo_) / rango * (alto - bajo) if rango > 0 else saber * np.nan
        estimado = estimado.clip(bajo, alto)
        faltante = X[COL_PGA].isna() & estimado.notna()
        X.loc[faltante, COL_PGA] = estimado[faltante]
        X[COL_INDICADOR_IMPUTADO] = faltante.astype(float)
        if not self.conservar_saber_global:
            X = X.drop(columns=[COL_SABER11_GLOBAL])
        return X

    def set_output(self, *, transform=None):
        return self


class CategoriasFijas(BaseEstimator, TransformerMixin):
    """Convierte las columnas categóricas al tipo `category` con las categorías vistas en `fit`.

    Una categoría nueva en prueba (p. ej., un profesor no visto) queda como faltante. Lo usa XGBoost.
    """

    def __init__(self, columnas=()):
        self.columnas = columnas

    def fit(self, X, y=None):
        self.categorias_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in self.columnas if c in X}
        return self

    def transform(self, X):
        X = X.copy()
        for col, categorias in self.categorias_.items():
            X[col] = pd.Categorical(X[col].astype(object).where(X[col].notna(), None), categories=categorias)
        return X

    def set_output(self, *, transform=None):
        return self


class CategoriasTexto(BaseEstimator, TransformerMixin):
    """Deja las categóricas como texto, con un valor explícito para los faltantes. Lo usa CatBoost,
    que maneja de forma nativa las categorías, incluidas las no vistas en el entrenamiento."""

    def __init__(self, columnas=()):
        self.columnas = columnas

    def fit(self, X, y=None):
        self.columnas_presentes_ = [c for c in self.columnas if c in X]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columnas:
            if col in X:
                X[col] = X[col].astype(object).where(X[col].notna(), FALTANTE).astype(str)
        return X

    def set_output(self, *, transform=None):
        return self


# ---------------------------------------------------------------------------
# Construcción de pipelines
# ---------------------------------------------------------------------------
def _columnas_salida_imputador(definicion) -> list:
    columnas = list(definicion.predictores)
    if definicion.imputa_promedio:
        columnas.append(COL_INDICADOR_IMPUTADO)
    return columnas


def _modelo(algoritmo: str, tarea: str, parametros: dict, categoricas: list, cfg):
    semilla = cfg.semilla
    if algoritmo == "RF":
        if tarea == "regresion":
            return RandomForestRegressor(random_state=semilla, n_jobs=-1, **parametros)
        return RandomForestClassifier(random_state=semilla, n_jobs=-1, class_weight="balanced", **parametros)
    if algoritmo == "XGB":
        comunes = dict(random_state=semilla, n_jobs=-1, enable_categorical=True, tree_method="hist",
                       device="cuda" if cfg.usar_gpu else "cpu")
        if tarea == "regresion":
            return XGBRegressor(objective="reg:squarederror", **comunes, **parametros)
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **comunes, **parametros)
    if algoritmo == "CAT":
        comunes = dict(random_seed=semilla, verbose=False, thread_count=-1, cat_features=categoricas,
                       allow_writing_files=False, **cfg.catboost_parametros_fijos)
        if tarea == "regresion":
            return CatBoostRegressor(loss_function="RMSE", **comunes, **parametros)
        return CatBoostClassifier(loss_function="Logloss", **comunes, **parametros)
    raise ValueError(f"Algoritmo no reconocido: {algoritmo}")


def construir_pipeline(algoritmo: str, tarea: str, definicion, parametros: dict, cfg) -> Pipeline:
    """Pipeline completo: imputación del promedio → tratamiento de categóricas → modelo."""
    imputador = ImputadorPromedio(activo=definicion.imputa_promedio, escala=cfg.escala_promedio)
    categoricas = list(definicion.categoricas)

    if algoritmo == "RF":
        numericas = [c for c in _columnas_salida_imputador(definicion) if c not in categoricas]
        codificador = ColumnTransformer(
            [
                ("categoricas", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                               encoded_missing_value=-1), categoricas),
                ("numericas", "passthrough", numericas),
            ],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        pasos = [("promedio", imputador), ("codificacion", codificador)]
    elif algoritmo == "XGB":
        pasos = [("promedio", imputador), ("categorias", CategoriasFijas(categoricas))]
    else:
        pasos = [("promedio", imputador), ("categorias", CategoriasTexto(categoricas))]

    pasos.append(("modelo", _modelo(algoritmo, tarea, parametros, categoricas, cfg)))
    return Pipeline(pasos)


def ajustar_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, algoritmo: str, tarea: str) -> Pipeline:
    """Ajusta el pipeline. En clasificación, XGBoost y CatBoost ponderan la clase retiro con la
    razón no retiro / retiro del conjunto de entrenamiento (RF usa class_weight='balanced')."""
    if tarea == "clasificacion" and algoritmo in ("XGB", "CAT"):
        razon = float((y == 0).sum() / max((y == 1).sum(), 1))
        pipeline.set_params(modelo__scale_pos_weight=razon)
    return pipeline.fit(X, y)


def predecir_nota(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Nota estimada, limitada a la escala 0-5."""
    return np.clip(np.asarray(pipeline.predict(X), dtype=float).ravel(), 0.0, 5.0)


def predecir_retiro(pipeline: Pipeline, X: pd.DataFrame, umbral: float = 0.5):
    """Probabilidad de retiro y clase predicha con el umbral indicado (0,5 por defecto)."""
    probabilidad = np.asarray(pipeline.predict_proba(X))[:, 1]
    return probabilidad, (probabilidad >= umbral).astype(int)
