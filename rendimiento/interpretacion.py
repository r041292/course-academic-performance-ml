"""Inferencia en 2025-30 con interpretación SHAP (formato de las corridas anteriores).

Para cada asignatura y algoritmo se calculan los valores SHAP del modelo de regresión de
la nota (el componente que se interpreta en el documento) y se agregan tres columnas:

  interpretacion_general_<ALG>   "Variables con mayor impacto promedio (|SHAP|): var (0.291), ..."
                                 (promedio del |SHAP| en los registros de la asignatura)
  interpretacion_registro_<ALG>  "A favor: var (+0.197), ... | En contra: var (-0.451), ..."
                                 (contribuciones del registro, ordenadas por magnitud)
  shap_valor_base_<ALG>          valor esperado del modelo (punto de partida de la explicación)

Opcionalmente se exporta un CSV largo con el valor SHAP numérico de cada variable y registro,
útil para regenerar las Figuras 6 a 9.
"""

import joblib
import numpy as np
import pandas as pd

from .entrenamiento import cargar_definicion, ruta_definicion, ruta_modelo
from .inferencia import predecir_asignatura
from .utilidades import log


def calcular_shap(algoritmo, modelo, X, categoricas):
    """Valores SHAP con el método TreeSHAP de cada biblioteca.

    Devuelve (matriz n_registros × n_variables, valor_base).
    """
    if algoritmo == "CAT":
        from catboost import Pool
        valores = modelo.get_feature_importance(Pool(X, cat_features=list(categoricas)), type="ShapValues")
        return valores[:, :-1], float(valores[0, -1])
    if algoritmo == "XGB":
        import xgboost
        valores = modelo.get_booster().predict(xgboost.DMatrix(X, enable_categorical=True), pred_contribs=True)
        return valores[:, :-1], float(valores[0, -1])
    import shap
    explicador = shap.TreeExplainer(modelo)
    base = explicador.expected_value
    return np.asarray(explicador.shap_values(X)), float(np.ravel(base)[0])


def texto_general(valores: np.ndarray, variables, top_n: int) -> str:
    importancia = pd.Series(np.abs(valores).mean(axis=0), index=variables).sort_values(ascending=False)
    return "Variables con mayor impacto promedio (|SHAP|): " + ", ".join(
        f"{v} ({i:.3f})" for v, i in importancia.head(top_n).items())


def texto_registro(fila: np.ndarray, variables, top_n: int) -> str:
    contribucion = pd.Series(fila, index=variables)
    a_favor = contribucion[contribucion > 0].sort_values(ascending=False).head(top_n)
    en_contra = contribucion[contribucion < 0].sort_values().head(top_n)
    favor = ", ".join(f"{v} (+{c:.3f})" for v, c in a_favor.items()) or "0"
    contra = ", ".join(f"{v} ({c:.3f})" for v, c in en_contra.items()) or "0"
    return f"A favor: {favor} | En contra: {contra}"


def interpretar_asignatura(prueba: pd.DataFrame, asignatura: str, cfg, tiempos, algoritmos, top_n: int,
                           exportar_valores: bool):
    """Predicciones de la asignatura más las columnas de interpretación SHAP de cada algoritmo."""
    salida = predecir_asignatura(prueba, asignatura, cfg, tiempos)
    definicion = cargar_definicion(cfg, asignatura)
    X = salida[definicion.columnas_entrada]
    largos = []
    for algoritmo in algoritmos:
        pipeline = joblib.load(ruta_modelo(cfg, algoritmo, "regresion", asignatura))
        X_modelo = pipeline[:-1].transform(X)
        variables = list(X_modelo.columns)
        with tiempos.medir("shap_inferencia_2025_30", asignatura, algoritmo, "regresion", len(X_modelo)):
            valores, base = calcular_shap(algoritmo, pipeline[-1], X_modelo, definicion.categoricas)
        salida[f"interpretacion_general_{algoritmo}"] = texto_general(valores, variables, top_n)
        salida[f"interpretacion_registro_{algoritmo}"] = [texto_registro(f, variables, top_n) for f in valores]
        salida[f"shap_valor_base_{algoritmo}"] = base
        if exportar_valores:
            largo = pd.DataFrame(valores, columns=variables, index=salida.index)
            largo[["Pidm", "nrc", "Cod materia curso"]] = salida[["Pidm", "nrc", "Cod materia curso"]]
            largo = largo.melt(id_vars=["Pidm", "nrc", "Cod materia curso"], var_name="variable", value_name="shap")
            largo["algoritmo"] = algoritmo
            largos.append(largo)
    return salida, largos


def inferencia_con_shap(prueba_completa: pd.DataFrame, asignaturas, cfg, tiempos, algoritmos=("CAT", "XGB"),
                        top_n: int = 10, exportar_valores: bool = False):
    """Recorre las asignaturas con modelo. Devuelve (predicciones con SHAP, valores SHAP en formato largo)."""
    partes, largos = [], []
    for numero, asignatura in enumerate(asignaturas, start=1):
        registros = prueba_completa[prueba_completa["Cod materia curso"] == asignatura]
        if registros.empty or not ruta_definicion(cfg, asignatura).exists():
            continue
        try:
            salida, largo = interpretar_asignatura(registros, asignatura, cfg, tiempos, algoritmos, top_n,
                                                   exportar_valores)
            partes.append(salida)
            largos.extend(largo)
        except Exception as error:
            log.warning("SHAP/inferencia fallida en %s: %s", asignatura, error)
        if numero % 50 == 0:
            log.info("SHAP: %d/%d asignaturas", numero, len(asignaturas))
    predicciones = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
    predicciones = predicciones[predicciones["clase_real"].notna()].reset_index(drop=True)
    valores = pd.concat(largos, ignore_index=True) if largos else pd.DataFrame()
    return predicciones, valores
