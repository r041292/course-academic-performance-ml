"""Pruebas estadísticas del documento.

- Friedman + W de Kendall + Wilcoxon con Holm (punto 12, Tablas 16 y 17).
- Bootstrap por asignaturas en 2025-30 (punto 13, Tabla 18).
- Cambio de distribución de notas: Kolmogórov–Smirnov por permutación, Benjamini–Hochberg
  y distancia de Wasserstein, con grupo de control (punto 9).
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, friedmanchisquare, mannwhitneyu, spearmanr, wasserstein_distance, wilcoxon

from .inferencia import APROBACION, PERDIDA, RETIRO

MENOR_ES_MEJOR = {"mae", "rmse", "log_loss"}


# ---------------------------------------------------------------------------
# Correcciones por comparaciones múltiples
# ---------------------------------------------------------------------------
def holm(p_valores):
    p = np.asarray(p_valores, dtype=float)
    orden = np.argsort(p)
    ajustados = np.empty_like(p)
    acumulado = 0.0
    for posicion, indice in enumerate(orden):
        acumulado = max(acumulado, (len(p) - posicion) * p[indice])
        ajustados[indice] = min(acumulado, 1.0)
    return ajustados


def benjamini_hochberg(p_valores):
    p = np.asarray(p_valores, dtype=float)
    n = len(p)
    orden = np.argsort(p)
    q = p[orden] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    resultado = np.empty_like(q)
    resultado[orden] = np.minimum(q, 1.0)
    return resultado


# ---------------------------------------------------------------------------
# Punto 12: Friedman sobre las medias de validación cruzada por asignatura
# ---------------------------------------------------------------------------
def comparacion_friedman(cv_asig: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (tarea, metrica), grupo in cv_asig.groupby(["tarea", "metrica"]):
        tabla = grupo.pivot_table(index="asignatura", columns="algoritmo", values="media").dropna()
        if tabla.shape[1] < 3 or len(tabla) < 2:
            continue
        algoritmos = list(tabla.columns)
        chi2, p = friedmanchisquare(*[tabla[a] for a in algoritmos])
        n, k = tabla.shape
        rangos = tabla.rank(axis=1, ascending=metrica in MENOR_ES_MEJOR).mean()
        pares = list(combinations(algoritmos, 2))
        p_pares = [wilcoxon(tabla[a], tabla[b]).pvalue for a, b in pares]
        fila = {"tarea": tarea, "metrica": metrica, "n_asignaturas": n, "chi2_friedman": chi2, "p_friedman": p,
                "w_kendall": chi2 / (n * (k - 1))}
        fila.update({f"rango_medio_{a}": rangos[a] for a in algoritmos})
        fila.update({f"p_holm_{a}_{b}": v for (a, b), v in zip(pares, holm(p_pares))})
        filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Punto 13: bootstrap por asignaturas (con estadísticos suficientes por asignatura)
# ---------------------------------------------------------------------------
def _resumen_por_asignatura(pred: pd.DataFrame, algoritmo: str) -> pd.DataFrame:
    """Conteos por asignatura que permiten recalcular cualquier métrica sumando asignaturas."""
    filas = []
    for asignatura, g in pred.groupby("Cod materia curso"):
        real = g["clase_real"].astype(int).to_numpy()
        est = g[f"clase_{algoritmo}"].astype(int).to_numpy()
        fila = {"asignatura": asignatura}
        for r in (0, 1, 2):
            for e in (0, 1, 2):
                fila[f"c{r}{e}"] = int(((real == r) & (est == e)).sum())
        con_nota = g["retiro_real"] == 0
        fila["suma_error_abs"] = float((g.loc[con_nota, "nota_real"] - g.loc[con_nota, f"nota_estimada_{algoritmo}"]).abs().sum())
        fila["n_nota"] = int(con_nota.sum())
        filas.append(fila)
    return pd.DataFrame(filas).set_index("asignatura")


def _metricas_desde_conteos(c: np.ndarray, error_abs: float, n_nota: float) -> dict:
    """c es la matriz de confusión 3×3 (filas real, columnas estimada)."""
    vp = np.diag(c)
    precision = np.divide(vp, c.sum(axis=0), out=np.zeros(3), where=c.sum(axis=0) > 0)
    recall = np.divide(vp, c.sum(axis=1), out=np.zeros(3), where=c.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
    total = c.sum()
    na_real, na_est = c[1:, :].sum(), c[:, 1:].sum()
    na_detectados = c[1:, 1:].sum()
    aciertos_inst = c[0, 0] + c[1, 1] + c[2, 1] + c[2, 2]
    return {
        "recall_no_aprobacion": na_detectados / na_real if na_real else np.nan,
        "precision_no_aprobacion": na_detectados / na_est if na_est else np.nan,
        "mae": error_abs / n_nota if n_nota else np.nan,
        "f1_macro": f1.mean(),
        "f1_aprobacion": f1[0], "f1_perdida": f1[1], "f1_retiro": f1[2],
        "acierto_institucional": aciertos_inst / total if total else np.nan,
    }


def bootstrap_por_asignaturas(pred: pd.DataFrame, algoritmos, repeticiones: int, semilla: int) -> pd.DataFrame:
    resumen = {a: _resumen_por_asignatura(pred, a) for a in algoritmos}
    asignaturas = resumen[algoritmos[0]].index
    columnas_c = [f"c{r}{e}" for r in range(3) for e in range(3)]

    arreglos = {a: (resumen[a][columnas_c].to_numpy(float), resumen[a]["suma_error_abs"].to_numpy(),
                    resumen[a]["n_nota"].to_numpy(float)) for a in algoritmos}

    def metricas(a, pesos):
        conteos, error_abs, n_nota = arreglos[a]
        c = (pesos @ conteos).reshape(3, 3)
        return _metricas_desde_conteos(c, pesos @ error_abs, pesos @ n_nota)

    puntuales = {a: metricas(a, np.ones(len(asignaturas))) for a in algoritmos}
    rng = np.random.default_rng(semilla)
    replicas = {a: [] for a in algoritmos}
    for _ in range(repeticiones):
        pesos = np.bincount(rng.integers(0, len(asignaturas), len(asignaturas)), minlength=len(asignaturas)).astype(float)
        for a in algoritmos:
            replicas[a].append(metricas(a, pesos))
    replicas = {a: pd.DataFrame(v) for a, v in replicas.items()}

    filas = []
    for a, b in combinations(algoritmos, 2):
        for metrica in puntuales[a]:
            diferencias = replicas[a][metrica] - replicas[b][metrica]
            filas.append({"algoritmo_a": a, "algoritmo_b": b, "metrica": metrica,
                          "valor_a": puntuales[a][metrica], "valor_b": puntuales[b][metrica],
                          "diferencia_a_menos_b": puntuales[a][metrica] - puntuales[b][metrica],
                          "ic95_inferior": diferencias.quantile(0.025), "ic95_superior": diferencias.quantile(0.975),
                          "asignaturas": len(asignaturas), "registros": len(pred), "repeticiones": repeticiones})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Punto 9: cambio de distribución de notas entre desarrollo y prueba
# ---------------------------------------------------------------------------
def ks_permutacion(a: np.ndarray, b: np.ndarray, permutaciones: int, rng) -> tuple[float, float]:
    """Estadístico D de Kolmogórov–Smirnov y p-valor por permutación (notas discretas, con empates).

    p = (1 + #{D_perm >= D_obs}) / (permutaciones + 1)   (Phipson y Smyth, 2010)
    """
    combinados = np.concatenate([a, b])
    valores, codigos = np.unique(np.round(combinados, 2), return_inverse=True)
    n_a, n = len(a), len(combinados)

    def distancia(etiquetas_a):
        cuenta_a = np.bincount(codigos[etiquetas_a], minlength=len(valores))
        cuenta_total = np.bincount(codigos, minlength=len(valores))
        cdf_a = np.cumsum(cuenta_a) / n_a
        cdf_b = np.cumsum(cuenta_total - cuenta_a) / (n - n_a)
        return np.max(np.abs(cdf_a - cdf_b))

    d_observado = distancia(np.arange(n_a))
    mayores = 0
    for _ in range(permutaciones):
        mayores += distancia(rng.permutation(n)[:n_a]) >= d_observado - 1e-12
    return d_observado, (1 + mayores) / (permutaciones + 1)


def cambio_distribucion_notas(historial: pd.DataFrame, por_asig: pd.DataFrame, asignaturas, cfg,
                              algoritmo_referencia: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """KS y Wasserstein por asignatura (todas las evaluadas) y comparación R² < 0 frente a R² >= 0."""
    notas = historial[historial["Cod materia curso"].isin(asignaturas) & (historial["Periodo"] <= cfg.periodo_prueba)]
    notas = notas.assign(nota=pd.to_numeric(notas["Calificacion_Final"], errors="coerce"))
    notas = notas[notas["nota"].between(1.5, 5.0)]
    rng = np.random.default_rng(cfg.semilla)

    reg = por_asig[por_asig["tarea"] == "regresion"]
    r2 = reg.pivot_table(index="asignatura", columns="algoritmo", values="r2")
    mae = reg.pivot_table(index="asignatura", columns="algoritmo", values="mae")

    filas = []
    for asignatura, g in notas.groupby("Cod materia curso"):
        antes = g.loc[g["Periodo"] <= cfg.periodo_max_desarrollo, "nota"].to_numpy()
        despues = g.loc[g["Periodo"] == cfg.periodo_prueba, "nota"].to_numpy()
        if len(antes) == 0 or len(despues) == 0 or asignatura not in r2.index:
            continue
        d, p = ks_permutacion(antes, despues, cfg.ks_permutaciones, rng)
        filas.append({"asignatura": asignatura, "n_desarrollo": len(antes), "n_prueba": len(despues),
                      "media_desarrollo": antes.mean(), "media_prueba": despues.mean(),
                      "ks_d": d, "p_permutacion": p, "wasserstein": wasserstein_distance(antes, despues),
                      **{f"r2_{a}": r2.loc[asignatura].get(a) for a in r2.columns},
                      f"mae_{algoritmo_referencia}": mae.loc[asignatura].get(algoritmo_referencia)})
    detalle = pd.DataFrame(filas)
    detalle["q_bh"] = benjamini_hochberg(detalle["p_permutacion"])
    detalle["cambio_significativo"] = detalle["q_bh"] < 0.05
    columnas_r2 = [c for c in detalle if c.startswith("r2_")]
    detalle["r2_negativo_alguno"] = (detalle[columnas_r2] < 0).any(axis=1)

    # Comparación entre grupos (grupo de control = asignaturas con R² >= 0)
    resumen = []
    for nombre, mascara in (("alguno", detalle["r2_negativo_alguno"]),
                            (algoritmo_referencia, detalle[f"r2_{algoritmo_referencia}"] < 0)):
        neg, pos = detalle[mascara], detalle[~mascara]
        tabla = [[neg["cambio_significativo"].sum(), (~neg["cambio_significativo"]).sum()],
                 [pos["cambio_significativo"].sum(), (~pos["cambio_significativo"]).sum()]]
        razon, p_fisher = fisher_exact(tabla)
        resumen.append({"criterio_r2_negativo": nombre, "n_r2_negativo": len(neg), "n_r2_no_negativo": len(pos),
                        "sig_en_r2_negativo": int(neg["cambio_significativo"].sum()),
                        "pct_sig_r2_negativo": neg["cambio_significativo"].mean(),
                        "sig_en_r2_no_negativo": int(pos["cambio_significativo"].sum()),
                        "pct_sig_r2_no_negativo": pos["cambio_significativo"].mean(),
                        "odds_ratio_fisher": razon, "p_fisher": p_fisher,
                        "wasserstein_media_r2_negativo": neg["wasserstein"].mean(),
                        "wasserstein_media_r2_no_negativo": pos["wasserstein"].mean(),
                        "p_mann_whitney_wasserstein": mannwhitneyu(neg["wasserstein"], pos["wasserstein"]).pvalue})
    rho_r2 = spearmanr(detalle["wasserstein"], detalle[f"r2_{algoritmo_referencia}"], nan_policy="omit")
    rho_mae = spearmanr(detalle["wasserstein"], detalle[f"mae_{algoritmo_referencia}"], nan_policy="omit")
    resumen.append({"criterio_r2_negativo": f"spearman_{algoritmo_referencia}",
                    "rho_wasserstein_r2": rho_r2.statistic, "p_wasserstein_r2": rho_r2.pvalue,
                    "rho_wasserstein_mae": rho_mae.statistic, "p_wasserstein_mae": rho_mae.pvalue,
                    "n_asignaturas": len(detalle)})
    return detalle, pd.DataFrame(resumen)
