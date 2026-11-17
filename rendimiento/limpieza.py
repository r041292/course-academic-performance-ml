"""Carga y limpieza del historial académico.

Las reglas son las mismas de la corrida anterior (notebook rev_multiples_asig...);
aquí solo se reorganizan en funciones cortas y se reemplazan los print por logging.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

from .utilidades import log

PREFIJO = "_ Matricula detalle para analisis."
COL_NOTA_RETIROS = PREFIJO + "Calif Final _ Retiros"   # nota final; -1 cuando hubo retiro
COL_RETIRO = "Retiro_Asignatura_Cat"                    # 1 = retiro, 0 = no retiro
OBSERVACIONES_VALIDAS = ("Prerrequisito cumplido", "No tiene pre requisito")
COLUMNAS_SIN_CONVERSION_DECIMAL = [
    "Nombre_Programa", PREFIJO + "Prof_Codigo", PREFIJO + "Sexo", PREFIJO + "Procedencia Categoria",
]

DEPARTAMENTOS_EXCLUIDOS = {
    "Esp. Der. de Sociedades", "Esp. Gcia. Finan. Serv. Salud", "Esp. Gcia. Proy. Ingeniería",
    "Esp. Neg. y Manejo. de Confli.", "Maestría en Matemáticas", "Esp. en Finanzas",
    "Centro para Excelencia Docente", "Esp.Gcia.Empresas Comerc.", "Dirección de Biblioteca",
    "Enf. Cuidado Crít Adulto", "Esp. En Responsab y Seguro", "Esp. Derecho Comercial",
    "Maestría en Epidemiología", "Maestría en Salud Pública", "Esp. en Derecho Penal",
    "Movilidad Nacional", "Sostenibilidad Ambiental", "Centro Recur Exito Estudiantil",
    "Semillero Investigadores-CIUN", "Intercambios Estudiantil", "Actividades Culturales",
    "Calidad y Proyectos Académicos", "Deportes", "Centro de Acompañamiento Estud",
    "Bienestar Universitario",
}

# Columnas que no se revisan en la eliminación global de filas con nulos.
COLUMNAS_IGNORADAS_ELIMINACION_GLOBAL = [
    PREFIJO + c for c in (
        "Puntaje_icfes_recalificado", "Dif anios icfes clase", "PCN_Puntaje_Ciencias_Naturales",
        "PIN_Puntaje_en_Ingles", "PLC_Puntaje_para_Lectura_Critica", "PMA_Puntaje_para_Matematicas",
        "PSC_Puntaje_Sociales_y_Ciudadanas", "Tipo_Colegio", "Tipo_Calendario", "pga inicial",
        "prom semestral t_1", "Asistencia CREE t_1",
    )
]


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------
def normalizar_texto(valor):
    """Minúsculas, sin tildes y con espacios simples; para comparar textos."""
    if pd.isna(valor):
        return None
    texto = str(valor).strip().lower()
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texto)


def nombre_snake_case(nombre: str) -> str:
    """Convierte un nombre de asignatura en snake_case sin tildes (para nombrar prerrequisitos)."""
    nombre = "".join(c for c in unicodedata.normalize("NFD", nombre) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-zA-Z0-9]+", "_", nombre).strip("_").lower()


# ---------------------------------------------------------------------------
# Limpieza inicial (sobre todo el historial)
# ---------------------------------------------------------------------------
def limpieza_inicial(df: pd.DataFrame) -> pd.DataFrame:
    """Homologaciones y exclusiones generales del historial."""
    df = df.copy()
    # El parquet guarda algunas columnas como categóricas. Se pasan a texto para que las
    # agrupaciones (p. ej., por asignatura) no generen grupos vacíos de categorías no presentes.
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(object)

    if "Nombre_Division" in df:
        es_docencia = df["Nombre_Division"].map(normalizar_texto) == normalizar_texto("Dirección de Docencia")
        df.loc[es_docencia, "Nombre_Division"] = "IESE-Inst.de Estudios en Educ."

    if "Descripcion_Materia" in df:
        texto = df["Descripcion_Materia"].map(normalizar_texto).fillna("")
        excluir = texto.str.contains(r"examen|practic|intercambi|seminario|ex\.?|turno|turnos|internado|refuerzo", regex=True)
        excluir |= texto.eq(normalizar_texto("PROGRAMA EDUCACION FINANCIERA"))
        df = df.loc[~excluir]

    col = PREFIJO + "DPTO Asignatura"
    if col in df:
        valores = df[col].map(normalizar_texto)
        excluidos = {normalizar_texto(v) for v in DEPARTAMENTOS_EXCLUIDOS}
        df = df.loc[~(valores.isna() | valores.isin(excluidos))]

    if "Nombre_Programa" in df:
        df = df.copy()
        valores = df["Nombre_Programa"].map(normalizar_texto).fillna("")
        lenguas = valores.str.contains("lenguas modernas y cultura") & valores.str.contains("barranquilla")
        df.loc[lenguas, "Nombre_Programa"] = "Lenguas Modernas y Cultura - BARRANQUILLA"
        licenciatura = valores == normalizar_texto("Licenciatura en Educación Infantil - BARRANQUILLA")
        df.loc[licenciatura, "Nombre_Programa"] = "Licenciat. pedagogía infantil"

    reemplazos = {
        PREFIJO + "Sexo": (lambda s: s.astype(str).str.strip().isin(["-99", "N"]), "-99"),
        PREFIJO + "Tipo_Colegio": (lambda s: s.astype(str).str.strip().eq("I"), "Institución Privada"),
        PREFIJO + "Tipo_Calendario": (lambda s: s.astype(str).str.strip().eq("I"), "Calendario B"),
    }
    for col, (condicion, valor) in reemplazos.items():
        if col in df:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype(object)
            df.loc[condicion(df[col]), col] = valor

    col = PREFIJO + "Asistencia CREE t_1"
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = _depurar_cohorte(df)

    col = PREFIJO + "Dif anios icfes clase"
    if col in df:
        negativos = pd.to_numeric(df[col], errors="coerce") < 0
        df.loc[negativos, col] = 0

    col = PREFIJO + "PLC_Puntaje_para_Lectura_Critica"
    if col in df:
        df = df.loc[~(pd.to_numeric(df[col], errors="coerce") > 100)]

    log.info("Limpieza inicial: %d registros", len(df))
    return df.copy()


def _depurar_cohorte(df: pd.DataFrame) -> pd.DataFrame:
    """Cohorte -99: se asigna el último periodo a estudiantes nuevos de ese periodo; el resto se elimina."""
    if "cohorte" not in df:
        return df
    cohorte = pd.to_numeric(df["cohorte"], errors="coerce")
    sin_cohorte = cohorte.eq(-99) | df["cohorte"].astype(str).str.strip().eq("-99")
    if "Periodo" in df and "Pidm" in df and sin_cohorte.any():
        periodo = pd.to_numeric(df["Periodo"], errors="coerce")
        ultimo = int(periodo.max())
        primer_periodo = df["Pidm"].map(periodo.groupby(df["Pidm"]).min())
        rellenar = sin_cohorte & primer_periodo.eq(ultimo) & periodo.eq(ultimo)
        df.loc[rellenar, "cohorte"] = ultimo
        sin_cohorte &= ~rellenar
    return df.loc[~sin_cohorte]


def nombres_de_asignaturas(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla código → nombre (primera descripción encontrada)."""
    nombres = df[["Cod materia curso", "Descripcion_Materia"]].dropna().drop_duplicates()
    nombres["Descripcion_Materia"] = nombres["Descripcion_Materia"].str.strip()
    return nombres.drop_duplicates("Cod materia curso").sort_values("Cod materia curso").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Selección de asignaturas
# ---------------------------------------------------------------------------
def seleccionar_asignaturas(historial: pd.DataFrame, max_asignaturas: int, matricula_minima: int) -> list:
    """Las asignaturas de mayor matrícula histórica con al menos `matricula_minima` registros."""
    conteo = historial.groupby("Cod materia curso").size().sort_values(ascending=False)
    conteo = conteo[conteo >= matricula_minima]
    return conteo.index[:max_asignaturas].tolist()


# ---------------------------------------------------------------------------
# Limpieza de registros para modelar
# ---------------------------------------------------------------------------
def convertir_comas_decimales(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas de texto con coma decimal → número cuando es posible."""
    df = df.copy()
    for col in df.columns:
        if col in COLUMNAS_SIN_CONVERSION_DECIMAL or df[col].dtype != object:
            continue
        df[col] = df[col].str.replace(",", ".", regex=False)
        try:
            df[col] = df[col].astype(float)
        except (ValueError, TypeError):
            pass
    return df


def calcular_intentos_y_semestres_profesor(df: pd.DataFrame) -> pd.DataFrame:
    """Número de intento del estudiante en la asignatura y semestres del profesor dictándola.

    Ambos cuentan hasta el periodo actual inclusive (se conocen al inicio del semestre).
    """
    intentos = df[["Pidm", "Cod materia curso", "Periodo"]].drop_duplicates().sort_values(
        ["Pidm", "Cod materia curso", "Periodo"])
    intentos["num_intentos_asignatura"] = intentos.groupby(["Pidm", "Cod materia curso"]).cumcount() + 1
    df = df.merge(intentos, on=["Pidm", "Cod materia curso", "Periodo"], how="left")

    col_prof = PREFIJO + "Prof_Codigo"
    if col_prof in df:
        semestres = df[[col_prof, "Cod materia curso", "Periodo"]].drop_duplicates().sort_values(
            [col_prof, "Cod materia curso", "Periodo"])
        semestres["num_semestres_profesor_asignatura"] = semestres.groupby([col_prof, "Cod materia curso"]).cumcount() + 1
        df = df.merge(semestres, on=[col_prof, "Cod materia curso", "Periodo"], how="left")
    else:
        df["num_semestres_profesor_asignatura"] = np.nan
    return df


def imputar_valores_basicos(df: pd.DataFrame) -> pd.DataFrame:
    """Reglas de relleno de la corrida anterior y creación de la variable de retiro."""
    df = df.copy()
    for col in (PREFIJO + "repitencia profesor referencia", PREFIJO + "repitencia profesor referencia 1ano",
                PREFIJO + "Asistencia CREE t_1"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    col = PREFIJO + "Procedencia Categoria"
    if col in df:
        df[col] = df[col].astype(object).replace("6 No registra", np.nan)

    col = PREFIJO + "Sexo"
    if col in df:
        df[col] = df[col].astype(object).replace([-99, "-99"], np.nan)

    if "Calificacion_Final" in df:
        nota = pd.to_numeric(df["Calificacion_Final"], errors="coerce").fillna(-1)
        df["Calificacion_Final"] = nota.where(nota >= 0, -1)

    df[COL_RETIRO] = (pd.to_numeric(df[COL_NOTA_RETIROS], errors="coerce") < 0).astype(int)
    return calcular_intentos_y_semestres_profesor(df)


def eliminar_columnas_vacias(df: pd.DataFrame) -> pd.DataFrame:
    vacias = df.columns[df.isna().all()].tolist()
    return df.drop(columns=vacias)


def eliminar_filas_con_pocos_nulos(df: pd.DataFrame, ignorar: list | None = None,
                                   ignorar_prerrequisitos: bool = False, umbral: float = 0.20):
    """Elimina filas con nulos en las columnas que tienen menos de `umbral` de vacíos.

    Columnas con muchos vacíos (≥ umbral) se conservan tal cual; los modelos las manejan.
    """
    ignorar = set(ignorar or [])
    for col in list(df.columns):
        if col in ignorar or (ignorar_prerrequisitos and col.startswith("Prereq_")):
            continue
        proporcion = df[col].isna().mean() if len(df) else 0
        if 0 < proporcion < umbral:
            df = df.dropna(subset=[col])
    return df


def preparar_registros(historial: pd.DataFrame, asignaturas: list, eliminar_filas: bool) -> pd.DataFrame:
    """Registros de las asignaturas elegidas, con las reglas de limpieza aplicadas.

    `eliminar_filas=True` se usa para el conjunto de desarrollo; en la inferencia no se
    eliminan filas, para predecir a todos los estudiantes matriculados.
    """
    df = historial[
        historial["Observacion_Prerrequisito"].isin(OBSERVACIONES_VALIDAS)
        & historial["Cod materia curso"].isin(asignaturas)
    ]
    df = convertir_comas_decimales(df)
    df = imputar_valores_basicos(df)
    df = eliminar_columnas_vacias(df)
    if eliminar_filas:
        df = eliminar_filas_con_pocos_nulos(df, ignorar=COLUMNAS_IGNORADAS_ELIMINACION_GLOBAL,
                                            ignorar_prerrequisitos=True)
    return df.reset_index(drop=True)
