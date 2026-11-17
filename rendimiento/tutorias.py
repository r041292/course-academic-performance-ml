"""Propuesta de asignación de tutorías (puntos 7 y 8 del checklist).

Reglas (las mismas de la corrida anterior, ahora explícitas):
  1. Unidad: número estimado de estudiantes que no aprobarán (pérdida o retiro) según la
     clase integrada del algoritmo elegido. Las asignaturas se agrupan por nombre.
  2. Se ordenan de mayor a menor número de casos (desempate: tasa estimada) y se eligen las TOP_N.
  3. Asignatura de repitencia alta: casos > media + K desviaciones de todas las asignaturas.
  4. NRC sobresaliente: casos > media + K desviaciones de los NRC de su asignatura;
     se prioriza uno por asignatura (el de más casos; desempate por tasa).
  5. Tipo de apoyo en las TOP_N: "Ambas" (alta + NRC), "Asignatura" (alta sin NRC, o sin
     superar el umbral y sin un NRC que destaque) y "NRC" (solo NRC sobresaliente).
Validación: se compara con la repitencia real de 2025-30 y con el criterio actual
(repitencia 2024-30 de la universidad) a nivel de asignatura; a nivel NRC solo con la realidad.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .inferencia import APROBACION

COL_REPITENCIA_INSTITUCIONAL = "Repitencia institucional usada para planear 2025-30 (LLENAR)"


def _preparar(pred: pd.DataFrame, algoritmo: str) -> pd.DataFrame:
    datos = pred.copy()
    datos["asignatura_nombre"] = datos["Descripcion_Materia"].astype(str).str.strip()
    datos["no_aprueba_estimado"] = (datos[f"clase_{algoritmo}"].astype(int) != APROBACION).astype(int)
    datos["no_aprueba_real"] = (datos["clase_real"].astype(int) != APROBACION).astype(int)
    return datos


def _ordenar(tabla: pd.DataFrame, casos: str, tasa: str) -> pd.DataFrame:
    return tabla.sort_values([casos, tasa], ascending=False).reset_index(drop=True)


def construir_propuesta(pred: pd.DataFrame, algoritmo: str, top_n: int, k: float):
    datos = _preparar(pred, algoritmo)

    asignaturas = datos.groupby("asignatura_nombre").agg(
        registros=("no_aprueba_estimado", "size"), casos_estimados=("no_aprueba_estimado", "sum"),
        casos_reales=("no_aprueba_real", "sum")).reset_index()
    asignaturas["tasa_estimada"] = asignaturas["casos_estimados"] / asignaturas["registros"]
    asignaturas["tasa_real"] = asignaturas["casos_reales"] / asignaturas["registros"]
    umbral_asignatura = asignaturas["casos_estimados"].mean() + k * asignaturas["casos_estimados"].std()
    asignaturas["repitencia_alta"] = asignaturas["casos_estimados"] > umbral_asignatura

    nrc = datos.groupby(["asignatura_nombre", "nrc"]).agg(
        registros=("no_aprueba_estimado", "size"), casos_estimados=("no_aprueba_estimado", "sum"),
        casos_reales=("no_aprueba_real", "sum")).reset_index()
    nrc["tasa_estimada"] = nrc["casos_estimados"] / nrc["registros"]
    por_asig = nrc.groupby("asignatura_nombre")["casos_estimados"]
    nrc["umbral_nrc"] = por_asig.transform("mean") + k * por_asig.transform("std")
    nrc["sobresaliente"] = nrc["casos_estimados"] > nrc["umbral_nrc"]
    por_asig_real = nrc.groupby("asignatura_nombre")["casos_reales"]
    nrc["umbral_nrc_real"] = por_asig_real.transform("mean") + k * por_asig_real.transform("std")

    priorizados = _ordenar(nrc[nrc["sobresaliente"]], "casos_estimados", "tasa_estimada") \
        .drop_duplicates("asignatura_nombre")[["asignatura_nombre", "nrc"]] \
        .rename(columns={"nrc": "nrc_priorizado"})
    asignaturas = asignaturas.merge(priorizados, on="asignatura_nombre", how="left")
    asignaturas["tiene_nrc_sobresaliente"] = asignaturas["nrc_priorizado"].notna()

    asignaturas = _ordenar(asignaturas, "casos_estimados", "tasa_estimada")
    asignaturas["rango"] = np.arange(1, len(asignaturas) + 1)
    asignaturas["seleccionada"] = asignaturas["rango"] <= top_n
    asignaturas["tipo_apoyo"] = np.select(
        [~asignaturas["seleccionada"],
         asignaturas["repitencia_alta"] & asignaturas["tiene_nrc_sobresaliente"],
         asignaturas["tiene_nrc_sobresaliente"] & ~asignaturas["repitencia_alta"]],
        ["Sin apoyo", "Ambas", "NRC"], default="Asignatura")

    sel = asignaturas[asignaturas["seleccionada"]]
    conteos = pd.DataFrame([
        ("asignaturas_evaluadas", len(asignaturas)),
        ("asignaturas_con_casos_estimados", int((asignaturas["casos_estimados"] > 0).sum())),
        ("umbral_repitencia_alta_casos", umbral_asignatura),
        ("asignaturas_repitencia_alta", int(asignaturas["repitencia_alta"].sum())),
        ("nrc_evaluados", len(nrc)),
        ("nrc_sobresalientes", int(nrc["sobresaliente"].sum())),
        ("asignaturas_con_nrc_sobresaliente", int(asignaturas["tiene_nrc_sobresaliente"].sum())),
        ("apoyo_ambas", int((sel["tipo_apoyo"] == "Ambas").sum())),
        ("apoyo_nrc", int((sel["tipo_apoyo"] == "NRC").sum())),
        ("apoyo_asignatura_total", int((sel["tipo_apoyo"] == "Asignatura").sum())),
        ("apoyo_asignatura_por_repitencia_alta", int(((sel["tipo_apoyo"] == "Asignatura") & sel["repitencia_alta"]).sum())),
        ("apoyo_asignatura_por_estar_en_top", int(((sel["tipo_apoyo"] == "Asignatura") & ~sel["repitencia_alta"]).sum())),
        ("asignaturas_con_nrc_sobresaliente_fuera_del_top", int((asignaturas["tiene_nrc_sobresaliente"] & ~asignaturas["seleccionada"]).sum())),
        ("empatadas_en_el_corte", int((asignaturas["casos_estimados"] == sel["casos_estimados"].min()).sum()) if len(sel) else 0),
    ], columns=["concepto", "valor"])
    return asignaturas, nrc, conteos


def validar_propuesta(asignaturas: pd.DataFrame, nrc: pd.DataFrame, pred: pd.DataFrame,
                      repitencia_institucional: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Compara la selección del modelo con el criterio actual y con la realidad de 2025-30."""
    # Criterio actual: tasa 2024-30 de la universidad × matrícula 2025-30 (conocida al inicio del semestre)
    inst = repitencia_institucional[["Cod materia curso", COL_REPITENCIA_INSTITUCIONAL]].rename(
        columns={COL_REPITENCIA_INSTITUCIONAL: "tasa_institucional"})
    por_codigo = pred.groupby("Cod materia curso").agg(registros=("clase_real", "size"),
                                                      nombre=("Descripcion_Materia", "first")).reset_index()
    por_codigo = por_codigo.merge(inst, on="Cod materia curso", how="left")
    por_codigo["asignatura_nombre"] = por_codigo["nombre"].astype(str).str.strip()
    por_codigo["casos_institucional"] = por_codigo["tasa_institucional"] * por_codigo["registros"]
    inst_nombre = por_codigo.groupby("asignatura_nombre")["casos_institucional"].sum(min_count=1).rename("casos_institucional")
    tabla = asignaturas.merge(inst_nombre, on="asignatura_nombre", how="left")
    tabla["tasa_institucional"] = tabla["casos_institucional"] / tabla["registros"]
    tabla = tabla.sort_values(["casos_institucional", "tasa_institucional"], ascending=False, na_position="last")
    seleccion_inst = set(tabla["asignatura_nombre"].head(top_n))
    seleccion_modelo = set(asignaturas.loc[asignaturas["seleccionada"], "asignatura_nombre"])

    total_real = asignaturas["casos_reales"].sum()
    cobertura = lambda nombres: asignaturas.loc[asignaturas["asignatura_nombre"].isin(nombres), "casos_reales"].sum() / total_real
    validos = tabla.dropna(subset=["casos_institucional"])

    filas = [
        ("cobertura_no_aprobados_reales_modelo", cobertura(seleccion_modelo)),
        ("cobertura_no_aprobados_reales_criterio_actual", cobertura(seleccion_inst)),
        ("asignaturas_en_ambas_selecciones", len(seleccion_modelo & seleccion_inst)),
        ("spearman_casos_modelo_vs_reales", spearmanr(asignaturas["casos_estimados"], asignaturas["casos_reales"]).statistic),
        ("spearman_casos_actual_vs_reales", spearmanr(validos["casos_institucional"], validos["casos_reales"]).statistic),
        ("spearman_tasa_modelo_vs_real", spearmanr(asignaturas["tasa_estimada"], asignaturas["tasa_real"]).statistic),
        ("spearman_tasa_actual_vs_real", spearmanr(validos["tasa_institucional"], validos["tasa_real"]).statistic),
        ("asignaturas_sin_dato_institucional", int(tabla["casos_institucional"].isna().sum())),
    ]

    # Nivel NRC: solo contra la realidad (no existe un criterio histórico de NRC)
    con_nrc = asignaturas.loc[asignaturas["seleccionada"] & asignaturas["tiene_nrc_sobresaliente"],
                              ["asignatura_nombre", "nrc_priorizado"]]
    nrc_sel = nrc[nrc["asignatura_nombre"].isin(con_nrc["asignatura_nombre"])]
    marcado = nrc_sel.merge(con_nrc, on="asignatura_nombre")
    marcado = marcado[marcado["nrc"] == marcado["nrc_priorizado"]]
    if len(nrc_sel) and nrc_sel["casos_reales"].sum():
        pct_casos = marcado["casos_reales"].sum() / nrc_sel["casos_reales"].sum()
        pct_matricula = marcado["registros"].sum() / nrc_sel["registros"].sum()
        filas += [
            ("nrc_priorizados", len(marcado)),
            ("nrc_pct_no_aprobados_reales_en_priorizados", pct_casos),
            ("nrc_pct_matricula_en_priorizados", pct_matricula),
            ("nrc_lift", pct_casos / pct_matricula if pct_matricula else np.nan),
            ("nrc_precision_supera_umbral_real", (marcado["casos_reales"] > marcado["umbral_nrc_real"]).mean()),
            ("nrc_spearman_casos_estimados_vs_reales", spearmanr(nrc["casos_estimados"], nrc["casos_reales"]).statistic),
        ]
    return pd.DataFrame(filas, columns=["indicador", "valor"])


def control_sensatez(asignaturas: pd.DataFrame, factor: float = 2.0, diferencia_minima: float = 0.20) -> pd.DataFrame:
    """Asignaturas con repitencia estimada muy distinta de la real (p. ej., el caso de Algoritmia I)."""
    t = asignaturas.copy()
    extremas = (t["tasa_estimada"] > factor * t["tasa_real"]) & ((t["tasa_estimada"] - t["tasa_real"]) > diferencia_minima)
    return t.loc[extremas, ["asignatura_nombre", "registros", "casos_estimados", "casos_reales",
                            "tasa_estimada", "tasa_real", "rango"]].sort_values("rango")
