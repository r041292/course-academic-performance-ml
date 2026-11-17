"""Búsqueda de hiperparámetros con Optuna dentro de la validación cruzada.

Para cada modelo (asignatura × algoritmo × tarea):
  1. Optuna propone una configuración (muestreador TPE con semilla fija).
  2. La configuración se evalúa con los MISMOS 5 folds para los tres algoritmos;
     en cada fold el pipeline completo se ajusta solo con el fold de entrenamiento.
  3. Los ensayos poco prometedores se detienen antes de terminar los 5 folds (MedianPruner).
Las métricas por fold del mejor ensayo son las que se reportan como validación cruzada.
Nota metodológica: como se usan los mismos folds para elegir y para reportar, esa
validación es algo optimista; la evaluación no sesgada es el periodo 2025-30.
"""

import numpy as np
import optuna
from sklearn.model_selection import KFold, StratifiedKFold

from .metricas import OBJETIVOS_OPTUNA, metricas_clasificacion, metricas_regresion
from .preprocesamiento import ajustar_pipeline, construir_pipeline, predecir_nota, predecir_retiro

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Espacios de búsqueda (rangos de referencia: Tablas B2-B4 de la versión anterior)
# ---------------------------------------------------------------------------
def proponer_parametros(trial: optuna.Trial, algoritmo: str) -> dict:
    if algoritmo == "RF":
        profundidad = trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30])
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": profundidad,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        }
    if algoritmo == "XGB":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if algoritmo == "CAT":
        return {
            "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
    raise ValueError(algoritmo)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def crear_folds(y, tarea: str, n_folds: int, semilla: int) -> list:
    """KFold con mezcla en regresión; StratifiedKFold por retiro en clasificación."""
    indices = np.zeros(len(y))
    if tarea == "regresion":
        divisor = KFold(n_splits=n_folds, shuffle=True, random_state=semilla)
        return list(divisor.split(indices))
    divisor = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=semilla)
    return list(divisor.split(indices, y))


def evaluar_en_fold(pipeline, X, y, entrenamiento, validacion, algoritmo, tarea, devolver_predicciones=False):
    """Ajusta el pipeline con el fold de entrenamiento y mide en el de validación.

    Con `devolver_predicciones=True` devuelve también las predicciones del fold de validación
    (predicciones fuera de muestra, para el archivo por registro de los datos de desarrollo).
    """
    X_ent, y_ent = X.iloc[entrenamiento], y.iloc[entrenamiento]
    X_val, y_val = X.iloc[validacion], y.iloc[validacion]
    ajustar_pipeline(pipeline, X_ent, y_ent, algoritmo, tarea)
    if tarea == "regresion":
        nota = predecir_nota(pipeline, X_val)
        metricas, predicciones = metricas_regresion(y_val, nota), {"nota": nota.tolist()}
    else:
        probabilidad, clase = predecir_retiro(pipeline, X_val)
        metricas = metricas_clasificacion(y_val, clase, probabilidad)
        predicciones = {"probabilidad": probabilidad.tolist(), "clase": clase.tolist()}
    if devolver_predicciones:
        return metricas, {"indices": list(map(int, validacion)), **predicciones}
    return metricas


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------
def buscar_hiperparametros(X, y, folds, algoritmo, tarea, definicion, cfg):
    """Devuelve (mejores_parámetros, métricas_por_fold, predicciones_por_fold, resumen_de_la_búsqueda),
    todo del mejor ensayo."""
    clave = "regresion" if tarea == "regresion" else cfg.metrica_optuna_clasificacion
    metrica, direccion = OBJETIVOS_OPTUNA[clave]

    def objetivo(trial):
        parametros = proponer_parametros(trial, algoritmo)
        resultados, predicciones = [], []
        for numero, (entrenamiento, validacion) in enumerate(folds):
            pipeline = construir_pipeline(algoritmo, tarea, definicion, parametros, cfg)
            metricas, prediccion = evaluar_en_fold(pipeline, X, y, entrenamiento, validacion, algoritmo, tarea,
                                                   devolver_predicciones=True)
            resultados.append(metricas)
            predicciones.append(prediccion)
            valor_parcial = float(np.nanmean([r[metrica] for r in resultados]))
            trial.report(valor_parcial, numero)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("folds", resultados)
        trial.set_user_attr("predicciones", predicciones)
        return float(np.nanmean([r[metrica] for r in resultados]))

    estudio = optuna.create_study(
        direction=direccion,
        sampler=optuna.samplers.TPESampler(seed=cfg.semilla),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    estudio.optimize(objetivo, n_trials=cfg.optuna_ensayos, timeout=cfg.optuna_tiempo_max_seg,
                     catch=(ValueError,))

    estados = [t.state for t in estudio.trials]
    resumen = {
        "metrica_optimizada": metrica,
        "direccion": direccion,
        "ensayos_totales": len(estudio.trials),
        "ensayos_completos": sum(e == optuna.trial.TrialState.COMPLETE for e in estados),
        "ensayos_podados": sum(e == optuna.trial.TrialState.PRUNED for e in estados),
        "mejor_valor": estudio.best_value,
    }
    mejor = estudio.best_trial
    parametros = {k: v for k, v in mejor.params.items()}
    return parametros, mejor.user_attrs["folds"], mejor.user_attrs["predicciones"], resumen
