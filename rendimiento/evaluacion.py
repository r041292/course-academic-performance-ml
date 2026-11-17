"""Tablas de resultados del documento a partir de la validación cruzada y de las predicciones 2025-30.

Cada función devuelve un DataFrame listo para exportar. La correspondencia con el
documento está en el nombre de la función y en el README.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .inferencia import APROBACION, NOMBRES_CLASES, PERDIDA, RETIRO
from .metricas import metricas_clasificacion, metricas_regresion

METRICAS_REGRESION = ["mae", "rmse", "r2"]
METRICAS_CLASIFICACION = ["accuracy", "precision_retiro", "recall_retiro", "f1_retiro", "f1_macro", "roc_auc", "pr_auc"]


# ---------------------------------------------------------------------------
# Tabla 8: validación cruzada
# ---------------------------------------------------------------------------
def cv_por_asignatura(cv: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación entre folds de cada métrica, por asignatura, algoritmo y tarea."""
    metricas = [m for m in METRICAS_REGRESION + METRICAS_CLASIFICACION if m in cv]
    largo = cv.melt(id_vars=["asignatura", "algoritmo", "tarea", "fold"], value_vars=metricas,
                    var_name="metrica", value_name="valor").dropna(subset=["valor"])
    largo = largo[((largo["tarea"] == "regresion") & largo["metrica"].isin(METRICAS_REGRESION))
                  | ((largo["tarea"] == "clasificacion") & largo["metrica"].isin(METRICAS_CLASIFICACION))]
    return (largo.groupby(["asignatura", "algoritmo", "tarea", "metrica"])["valor"]
            .agg(media="mean", desviacion=lambda s: s.std(ddof=0)).reset_index())


def tabla8_validacion_cruzada(cv_asig: pd.DataFrame) -> pd.DataFrame:
    """Promedio entre asignaturas de la media y de la desviación entre folds."""
    return (cv_asig.groupby(["tarea", "metrica", "algoritmo"])
            .agg(media=("media", "mean"), desviacion_media_entre_folds=("desviacion", "mean"),
                 n_asignaturas=("asignatura", "nunique"))
            .reset_index())


# ---------------------------------------------------------------------------
# Tabla 9 y tabla nueva de R²: 2025-30 por asignatura
# ---------------------------------------------------------------------------
def metricas_prueba_por_asignatura(pred: pd.DataFrame, algoritmos) -> pd.DataFrame:
    filas = []
    for asignatura, grupo in pred.groupby("Cod materia curso"):
        con_nota = grupo[grupo["retiro_real"] == 0]
        for algoritmo in algoritmos:
            if len(con_nota):
                filas.append({"asignatura": asignatura, "algoritmo": algoritmo, "tarea": "regresion",
                              "sd_nota_real": con_nota["nota_real"].std(ddof=0),
                              **metricas_regresion(con_nota["nota_real"], con_nota[f"nota_estimada_{algoritmo}"])})
            if bool(grupo["tiene_clasificador"].iloc[0]):
                filas.append({"asignatura": asignatura, "algoritmo": algoritmo, "tarea": "clasificacion",
                              **metricas_clasificacion(grupo["retiro_real"], grupo[f"retiro_predicho_{algoritmo}"],
                                                       grupo[f"prob_retiro_{algoritmo}"])})
    return pd.DataFrame(filas)


def tabla9_prueba_por_asignatura(por_asig: pd.DataFrame) -> pd.DataFrame:
    """Promedio y desviación entre asignaturas. Las métricas de la clase retiro excluyen
    las asignaturas sin retiros reales en el periodo (recall no definido)."""
    filas = []
    for (tarea, algoritmo), grupo in por_asig.groupby(["tarea", "algoritmo"]):
        metricas = METRICAS_REGRESION if tarea == "regresion" else METRICAS_CLASIFICACION
        for metrica in metricas:
            datos = grupo
            if metrica in ("recall_retiro", "f1_retiro", "roc_auc", "pr_auc"):
                datos = grupo[grupo["n_retiros_reales"] > 0]
            valores = datos[metrica].dropna()
            filas.append({"tarea": tarea, "algoritmo": algoritmo, "metrica": metrica,
                          "media": valores.mean(), "desviacion_entre_asignaturas": valores.std(ddof=0),
                          "n_asignaturas": len(valores),
                          "n_asignaturas_excluidas": len(grupo) - len(valores)})
    return pd.DataFrame(filas)


def tabla_distribucion_r2(por_asig: pd.DataFrame) -> pd.DataFrame:
    """Tabla nueva (punto 10): distribución del R² por asignatura en 2025-30."""
    reg = por_asig[por_asig["tarea"] == "regresion"]
    filas = []
    for algoritmo, grupo in reg.groupby("algoritmo"):
        r2 = grupo["r2"].dropna()
        filas.append({"algoritmo": algoritmo, "asignaturas_con_r2": len(r2),
                      "asignaturas_sin_r2": int(grupo["r2"].isna().sum()),
                      "media": r2.mean(), "mediana": r2.median(), "p25": r2.quantile(0.25),
                      "p75": r2.quantile(0.75), "minimo": r2.min(), "maximo": r2.max(),
                      "n_r2_negativo": int((r2 < 0).sum()), "pct_r2_negativo": float((r2 < 0).mean()),
                      "mae_mediana": grupo["mae"].median(), "mae_p25": grupo["mae"].quantile(0.25),
                      "mae_p75": grupo["mae"].quantile(0.75)})
    ancho = reg.pivot_table(index="asignatura", columns="algoritmo", values="r2")
    filas.append({"algoritmo": "al_menos_uno", "asignaturas_con_r2": int(ancho.notna().all(axis=1).sum()),
                  "n_r2_negativo": int((ancho < 0).any(axis=1).sum())})
    filas.append({"algoritmo": "todos", "n_r2_negativo": int((ancho < 0).all(axis=1).sum())})
    return pd.DataFrame(filas)


def caso_r2_extremo(por_asig: pd.DataFrame, pred: pd.DataFrame, algoritmo: str) -> pd.DataFrame:
    """Asignatura con el R² más bajo del algoritmo, con los datos para explicar el caso (tipo LEY4625)."""
    reg = por_asig[(por_asig["tarea"] == "regresion") & (por_asig["algoritmo"] == algoritmo)].dropna(subset=["r2"])
    if reg.empty:
        return pd.DataFrame()
    fila = reg.loc[reg["r2"].idxmin()]
    notas = pred[(pred["Cod materia curso"] == fila["asignatura"]) & (pred["retiro_real"] == 0)]["nota_real"]
    return pd.DataFrame([{"asignatura": fila["asignatura"], "algoritmo": algoritmo, "n": int(fila["n"]),
                          "media_nota_real": notas.mean(), "sd_nota_real": notas.std(ddof=0),
                          "rmse_referencia": notas.std(ddof=0), "rmse_modelo": fila["rmse"],
                          "razon_rmse": fila["rmse"] / notas.std(ddof=0) if notas.std(ddof=0) > 0 else np.nan,
                          "r2": fila["r2"]}])


# ---------------------------------------------------------------------------
# Tabla 10: global sobre todos los registros
# ---------------------------------------------------------------------------
def tabla10_global(pred: pd.DataFrame, algoritmos) -> pd.DataFrame:
    con_nota = pred[pred["retiro_real"] == 0]
    con_clf = pred[pred["tiene_clasificador"]]
    filas = []
    for algoritmo in algoritmos:
        filas.append({"algoritmo": algoritmo, "tarea": "regresion",
                      **metricas_regresion(con_nota["nota_real"], con_nota[f"nota_estimada_{algoritmo}"])})
        if len(con_clf):
            filas.append({"algoritmo": algoritmo, "tarea": "clasificacion",
                          **metricas_clasificacion(con_clf["retiro_real"], con_clf[f"retiro_predicho_{algoritmo}"],
                                                   con_clf[f"prob_retiro_{algoritmo}"])})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Tablas 11-13: sistema integrado de tres clases
# ---------------------------------------------------------------------------
def matrices_confusion_3_clases(pred: pd.DataFrame, algoritmos) -> pd.DataFrame:
    filas = []
    for algoritmo in algoritmos:
        tabla = pd.crosstab(pred["clase_real"].astype(int), pred[f"clase_{algoritmo}"].astype(int))
        tabla = tabla.reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
        for real in (0, 1, 2):
            filas.append({"algoritmo": algoritmo, "clase_real": NOMBRES_CLASES[real],
                          **{f"pred_{NOMBRES_CLASES[p]}": int(tabla.loc[real, p]) for p in (0, 1, 2)}})
    return pd.DataFrame(filas)


def tabla13_metricas_por_clase(pred: pd.DataFrame, algoritmos) -> pd.DataFrame:
    filas = []
    for algoritmo in algoritmos:
        real = pred["clase_real"].astype(int)
        estimada = pred[f"clase_{algoritmo}"].astype(int)
        p, r, f, n = precision_recall_fscore_support(real, estimada, labels=[0, 1, 2], zero_division=0)
        for i in (0, 1, 2):
            filas.append({"algoritmo": algoritmo, "clase": NOMBRES_CLASES[i], "precision": p[i], "recall": r[i],
                          "f1_score": f[i], "registros_reales": int(n[i])})
        for promedio, nombre in (("macro", "Promedio macro"), ("weighted", "Promedio ponderado")):
            p, r, f, _ = precision_recall_fscore_support(real, estimada, labels=[0, 1, 2], average=promedio,
                                                         zero_division=0)
            filas.append({"algoritmo": algoritmo, "clase": nombre, "precision": p, "recall": r, "f1_score": f,
                          "registros_reales": len(real)})
        filas.append({"algoritmo": algoritmo, "clase": "Accuracy", "precision": np.nan, "recall": np.nan,
                      "f1_score": float((real == estimada).mean()), "registros_reales": len(real)})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Tablas 14-15: acierto institucional (regla asimétrica de la Tabla 5) y sensibilidad
# ---------------------------------------------------------------------------
CATEGORIAS_ACIERTO = [
    ("Asignatura no retirada", "Acierto en nota", True),
    ("Asignatura no retirada", "Error en nota", False),
    ("Asignatura no retirada", "Predicción de retiro", False),
    ("Asignatura retirada", "Predicción de aprobación", False),
    ("Asignatura retirada", "Predicción de pérdida", True),
    ("Asignatura retirada", "Predicción de retiro", True),
]


def _categoria_acierto(real: int, estimada: int) -> str:
    if real == RETIRO:
        return {APROBACION: "Predicción de aprobación", PERDIDA: "Predicción de pérdida", RETIRO: "Predicción de retiro"}[estimada]
    if estimada == RETIRO:
        return "Predicción de retiro"
    return "Acierto en nota" if estimada == real else "Error en nota"


def tablas14_15_acierto_institucional(pred: pd.DataFrame, algoritmos) -> pd.DataFrame:
    filas = []
    total = len(pred)
    for algoritmo in algoritmos:
        real = pred["clase_real"].astype(int)
        estimada = pred[f"clase_{algoritmo}"].astype(int)
        categoria = [_categoria_acierto(r, e) for r, e in zip(real, estimada)]
        grupo_real = np.where(real == RETIRO, "Asignatura retirada", "Asignatura no retirada")
        conteo = pd.Series(list(zip(grupo_real, categoria))).value_counts()
        n_grupo = pd.Series(grupo_real).value_counts()
        for grupo, nombre, es_acierto in CATEGORIAS_ACIERTO:
            n = int(conteo.get((grupo, nombre), 0))
            filas.append({"algoritmo": algoritmo, "resultado_real": grupo, "categoria": nombre, "es_acierto": es_acierto,
                          "n": n, "pct_del_total": n / total, "pct_del_resultado_real": n / n_grupo.get(grupo, np.nan),
                          "n_resultado_real": int(n_grupo.get(grupo, 0))})
        aciertos = sum(f["n"] for f in filas if f["algoritmo"] == algoritmo and f["es_acierto"])
        # Sensibilidad (punto 1): contar también la pérdida real predicha como retiro.
        perdida_como_retiro = int(((real == PERDIDA) & (estimada == RETIRO)).sum())
        filas.append({"algoritmo": algoritmo, "resultado_real": "Total", "categoria": "Acierto institucional",
                      "n": aciertos, "pct_del_total": aciertos / total})
        filas.append({"algoritmo": algoritmo, "resultado_real": "Total",
                      "categoria": "Acierto institucional con regla simétrica (sensibilidad)",
                      "n": aciertos + perdida_como_retiro, "pct_del_total": (aciertos + perdida_como_retiro) / total})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Punto 2 y Anexo C: aprobación frente a no aprobación
# ---------------------------------------------------------------------------
def no_aprobacion(pred: pd.DataFrame, algoritmos) -> pd.DataFrame:
    """Detección de resultados desfavorables (pérdida o retiro) y matriz binaria de cada algoritmo."""
    real = (pred["clase_real"].astype(int) != APROBACION).astype(int)
    filas = []
    for algoritmo in algoritmos:
        estimada = (pred[f"clase_{algoritmo}"].astype(int) != APROBACION).astype(int)
        vp = int(((real == 1) & (estimada == 1)).sum())
        fp = int(((real == 0) & (estimada == 1)).sum())
        fn = int(((real == 1) & (estimada == 0)).sum())
        vn = int(((real == 0) & (estimada == 0)).sum())
        p, r, f, _ = precision_recall_fscore_support(real, estimada, labels=[0, 1], zero_division=0)
        filas.append({"algoritmo": algoritmo, "registros": len(real),
                      "real_A_pred_A": vn, "real_A_pred_NA": fp, "real_NA_pred_A": fn, "real_NA_pred_NA": vp,
                      "recall_no_aprobacion": r[1], "precision_no_aprobacion": p[1],
                      "predicciones_desfavorables": vp + fp, "no_aprobados_reales": vp + fn,
                      "f1_macro_binario": float(np.mean(f)), "accuracy_binario": (vp + vn) / len(real),
                      "proporcion_aprobacion_real": float((real == 0).mean())})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Puntos 17 y 24: alcance del clasificador y asignaturas evaluadas
# ---------------------------------------------------------------------------
def retiros_sin_clasificador(pred: pd.DataFrame) -> pd.DataFrame:
    grupos = pred.groupby("tiene_clasificador").agg(
        asignaturas=("Cod materia curso", "nunique"), registros=("retiro_real", "size"),
        retiros_reales=("retiro_real", "sum"))
    con_retiro = pred[pred["retiro_real"] == 1].groupby("tiene_clasificador")["Cod materia curso"].nunique()
    grupos["asignaturas_con_al_menos_un_retiro"] = con_retiro
    grupos["pct_de_los_retiros"] = grupos["retiros_reales"] / grupos["retiros_reales"].sum()
    return grupos.reset_index()


def conteo_asignaturas(modeladas: list, registros_prueba: pd.DataFrame, pred: pd.DataFrame,
                       historial: pd.DataFrame, periodo_prueba: int) -> pd.DataFrame:
    """Cuántas asignaturas modeladas se evaluaron en 2025-30 y por qué no las demás."""
    ofrecidas = set(historial.loc[historial["Periodo"] == periodo_prueba, "Cod materia curso"])
    con_registros_validos = set(registros_prueba["Cod materia curso"])
    evaluadas = set(pred["Cod materia curso"])
    modeladas = set(modeladas)
    return pd.DataFrame([
        {"concepto": "asignaturas_modeladas", "n": len(modeladas)},
        {"concepto": "evaluadas_en_prueba", "n": len(modeladas & evaluadas)},
        {"concepto": "no_ofrecidas_en_prueba", "n": len(modeladas - ofrecidas)},
        {"concepto": "ofrecidas_sin_registros_validos", "n": len((modeladas & ofrecidas) - evaluadas)},
        {"concepto": "registros_evaluados", "n": len(pred)},
        {"concepto": "registros_aprobacion", "n": int((pred["clase_real"] == APROBACION).sum())},
        {"concepto": "registros_perdida", "n": int((pred["clase_real"] == PERDIDA).sum())},
        {"concepto": "registros_retiro", "n": int((pred["clase_real"] == RETIRO).sum())},
        {"concepto": "asignaturas_por_nombre", "n": pred["Descripcion_Materia"].str.strip().nunique()},
        {"concepto": "nrc_evaluados", "n": pred.groupby(["Cod materia curso", "nrc"]).ngroups},
        {"concepto": "con_registros_validos_sin_filtrar", "n": len(modeladas & con_registros_validos)},
    ])


# ---------------------------------------------------------------------------
# Formato largo, como los archivos de estadísticas de la corrida anterior
# ---------------------------------------------------------------------------
def estadisticas_formato_anterior(cv_asig: pd.DataFrame, por_asig: pd.DataFrame, nombres: pd.DataFrame) -> pd.DataFrame:
    """Columnas: mean, std, metric, Conjunto, CV o Directo, Descripcion_Materia, Cod materia curso,
    Modelo, Tipo_Modelo. Validación cruzada (media y desviación entre folds) y 2025-30 (directo)."""
    tipo = {"regresion": "Regresion", "clasificacion": "Clasificacion"}
    cv = pd.DataFrame({
        "mean": cv_asig["media"], "std": cv_asig["desviacion"], "metric": cv_asig["metrica"] + " (CV)",
        "Conjunto": "Desarrollo", "CV o Directo": "CV", "Cod materia curso": cv_asig["asignatura"],
        "Modelo": cv_asig["algoritmo"], "Tipo_Modelo": cv_asig["tarea"].map(tipo)})
    metricas = [m for m in METRICAS_REGRESION + METRICAS_CLASIFICACION + ["n", "n_retiros_reales"] if m in por_asig]
    prueba = por_asig.melt(id_vars=["asignatura", "algoritmo", "tarea"], value_vars=metricas,
                           var_name="metrica", value_name="mean").dropna(subset=["mean"])
    prueba = pd.DataFrame({
        "mean": prueba["mean"], "std": np.nan, "metric": prueba["metrica"] + " (directo)",
        "Conjunto": "2025-30", "CV o Directo": "Directo", "Cod materia curso": prueba["asignatura"],
        "Modelo": prueba["algoritmo"], "Tipo_Modelo": prueba["tarea"].map(tipo)})
    largo = pd.concat([cv, prueba], ignore_index=True)
    largo = largo.merge(nombres, on="Cod materia curso", how="left")
    return largo[["mean", "std", "metric", "Conjunto", "CV o Directo", "Descripcion_Materia",
                  "Cod materia curso", "Modelo", "Tipo_Modelo"]]
