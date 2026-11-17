"""Definición de las variables de cada asignatura.

Cada asignatura tiene su propio conjunto de predictores. La definición se decide UNA
vez con los datos de desarrollo y se guarda junto al modelo, para aplicarla igual en
la validación cruzada y en el periodo de prueba.
"""

from dataclasses import asdict, dataclass, field

import pandas as pd

from .limpieza import COL_RETIRO, PREFIJO, eliminar_filas_con_pocos_nulos, nombre_snake_case

OBJETIVO_NOTA = "resultado_final"
OBJETIVO_RETIRO = COL_RETIRO
COL_SABER11_GLOBAL = "puntaje_saber11"
COL_PGA = "pga_inicial"
COL_INDICADOR_IMPUTADO = "promedio_imputado"

RENOMBRAR = {
    PREFIJO + "repitencia profesor referencia 1ano": "repitencia_prof_ref",
    "Nombre_Programa": "programa",
    PREFIJO + "Prof_Codigo": "profesor_codigo",
    PREFIJO + "pga inicial": "pga_inicial",
    PREFIJO + "prom semestral t_1": "promedio_sem_t1",
    PREFIJO + "Sexo": "sexo",
    PREFIJO + "Asistencia CREE t_1": "asistencia_cree_t1",
    PREFIJO + "Procedencia Categoria": "procedencia_categoria",
    PREFIJO + "Edad cursan asignatura": "edad_curso",
    PREFIJO + "Calif Final _ Retiros": "resultado_final",
    PREFIJO + "Puntaje estrato": "estrato",
    PREFIJO + "Puntaje_icfes_recalificado": "puntaje_saber11",
    PREFIJO + "Dif anios icfes clase": "años_saber11_vs_clase",
    PREFIJO + "PCN_Puntaje_Ciencias_Naturales": "saber11_Ciencias_Naturales",
    PREFIJO + "PIN_Puntaje_en_Ingles": "saber11_ingles",
    PREFIJO + "PLC_Puntaje_para_Lectura_Critica": "saber11_lectura_critica",
    PREFIJO + "PMA_Puntaje_para_Matematicas": "saber11_matematicas",
    PREFIJO + "PSC_Puntaje_Sociales_y_Ciudadanas": "saber11_sociales",
    PREFIJO + "Tipo_Colegio": "Tipo_colegio",
    PREFIJO + "Tipo_Calendario": "Tipo_calendario",
}

CATEGORICAS = ["programa", "profesor_codigo", "sexo", "procedencia_categoria", "Tipo_colegio", "Tipo_calendario"]

BASE_CON_PRERREQUISITOS = [
    "repitencia_prof_ref", "programa", "profesor_codigo", "pga_inicial", "promedio_sem_t1", "sexo",
    "asistencia_cree_t1", "procedencia_categoria", "edad_curso", "num_intentos_asignatura",
    "num_semestres_profesor_asignatura", "estrato",
]
BASE_SIN_PRERREQUISITOS = [
    "repitencia_prof_ref", "programa", "profesor_codigo", "sexo", "asistencia_cree_t1",
    "procedencia_categoria", "edad_curso", "num_intentos_asignatura",
    "num_semestres_profesor_asignatura", "estrato", "Tipo_colegio", "Tipo_calendario",
]
SABER11_COMPLETO = [
    "puntaje_saber11", "años_saber11_vs_clase", "saber11_Ciencias_Naturales", "saber11_ingles",
    "saber11_lectura_critica", "saber11_matematicas", "saber11_sociales",
]
SABER11_SIN_GLOBAL = [c for c in SABER11_COMPLETO if c != COL_SABER11_GLOBAL]


@dataclass
class DefinicionAsignatura:
    """Qué predictores usa una asignatura y cómo se trata su promedio acumulado."""
    asignatura: str
    tiene_prerrequisitos: bool
    regimen_promedio: str                 # "con_prerrequisitos", "usar_pga", "imputar_pga" o "saber11"
    predictores: list                     # columnas que entran al modelo (antes del preprocesamiento)
    categoricas: list
    prerrequisitos: dict = field(default_factory=dict)  # nombre_columna -> (columna fuente, código)

    @property
    def imputa_promedio(self) -> bool:
        return self.regimen_promedio == "imputar_pga"

    @property
    def columnas_entrada(self) -> list:
        """Columnas que necesita el pipeline (incluye el puntaje global si se imputa el PGA)."""
        extra = [COL_SABER11_GLOBAL] if self.imputa_promedio and COL_SABER11_GLOBAL not in self.predictores else []
        return self.predictores + extra

    def a_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def desde_dict(cls, datos: dict) -> "DefinicionAsignatura":
        return cls(**datos)


# ---------------------------------------------------------------------------
# Prerrequisitos
# ---------------------------------------------------------------------------
def _columnas_prerrequisito(df: pd.DataFrame, nombres: pd.DataFrame, umbral: float) -> dict:
    """Prerrequisitos con nota disponible en al menos `umbral` de los registros.

    Las columnas Prereq_i_* cambian de significado entre estudiantes (la posición i no
    siempre es la misma asignatura), por eso se renombran con el nombre de la asignatura
    prerrequisito: Prereq_<nombre>_Nota y Prereq_<nombre>_Intentos.
    Devuelve {nombre_nuevo: (columna_fuente_i, codigo_prerrequisito)} en orden de aparición.
    """
    mapa_nombres = dict(zip(nombres["Cod materia curso"], nombres["Descripcion_Materia"]))
    seleccion = {}
    for col_nota in [c for c in df.columns if c.startswith("Prereq_") and c.endswith("_Nota")]:
        col_intentos = col_nota.replace("_Nota", "_Intentos")
        col_codigo = col_nota.replace("_Nota", "_Codigo")
        if col_intentos not in df or col_codigo not in df:
            continue
        if df[col_nota].notna().mean() < umbral:
            continue
        for codigo in df[col_codigo].dropna().unique():
            base = nombre_snake_case(str(mapa_nombres.get(codigo, codigo)))
            for sufijo, fuente in (("Nota", col_nota), ("Intentos", col_intentos)):
                nombre = f"Prereq_{base}_{sufijo}"
                seleccion.setdefault(nombre, (fuente, codigo))
    return seleccion


def agregar_columnas_prerrequisito(df: pd.DataFrame, prerrequisitos: dict) -> pd.DataFrame:
    """Crea las columnas Prereq_<nombre>_* en cualquier conjunto (desarrollo o prueba)."""
    df = df.copy()
    for nombre, (fuente, codigo) in prerrequisitos.items():
        col_codigo = fuente.replace("_Nota", "_Codigo").replace("_Intentos", "_Codigo")
        if fuente in df and col_codigo in df:
            valores = pd.to_numeric(df[fuente], errors="coerce").where(df[col_codigo] == codigo)
        else:
            valores = pd.Series(float("nan"), index=df.index)
        df[nombre] = valores if nombre not in df else df[nombre].fillna(valores)
    return df


# ---------------------------------------------------------------------------
# Definición por asignatura
# ---------------------------------------------------------------------------
def renombrar(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RENOMBRAR)


def _regimen_promedio(df: pd.DataFrame, umbral_saber11: float, umbral_imputar: float) -> str:
    """Asignaturas sin prerrequisitos: el % de registros sin PGA indica si el curso se toma
    sobre todo en primer semestre (la universidad no registra esa ubicación)."""
    if COL_PGA not in df or len(df) == 0:
        return "saber11"
    vacios = df[COL_PGA].isna().mean()
    if vacios > umbral_saber11:
        return "saber11"
    if vacios > umbral_imputar:
        if COL_SABER11_GLOBAL in df and pd.to_numeric(df[COL_SABER11_GLOBAL], errors="coerce").nunique() > 1:
            return "imputar_pga"
        return "saber11"
    return "usar_pga"


def definir_asignatura(df_desarrollo: pd.DataFrame, asignatura: str, nombres: pd.DataFrame, cfg) -> DefinicionAsignatura:
    """Decide los predictores de la asignatura con sus datos de desarrollo (sin mirar el objetivo)."""
    df = renombrar(df_desarrollo)
    observacion = df["Observacion_Prerrequisito"].dropna()
    tiene_prerrequisitos = len(observacion) > 0 and observacion.iloc[0] == "Prerrequisito cumplido"

    if tiene_prerrequisitos:
        regimen = "con_prerrequisitos"
        base = list(BASE_CON_PRERREQUISITOS)
        prerrequisitos = _columnas_prerrequisito(df, nombres, cfg.umbral_completitud_prerrequisito)
    else:
        regimen = _regimen_promedio(df, cfg.umbral_pga_sin_promedio, cfg.umbral_pga_imputar)
        base = list(BASE_SIN_PRERREQUISITOS)
        base += {"saber11": SABER11_COMPLETO,
                 "imputar_pga": [COL_PGA] + SABER11_SIN_GLOBAL,
                 "usar_pga": [COL_PGA, "promedio_sem_t1"]}[regimen]
        # Corrida anterior: en asignaturas sin prerrequisitos se usaban las columnas Prereq_i_* originales
        # que superaran el umbral; en la práctica estas asignaturas no las tienen.
        prerrequisitos = {}

    predictores = [c for c in base if c in df.columns] + list(prerrequisitos)
    return DefinicionAsignatura(
        asignatura=asignatura,
        tiene_prerrequisitos=tiene_prerrequisitos,
        regimen_promedio=regimen,
        predictores=predictores,
        categoricas=[c for c in CATEGORICAS if c in predictores],
        prerrequisitos=prerrequisitos,
    )


def construir_tabla(df: pd.DataFrame, definicion: DefinicionAsignatura, eliminar_filas: bool) -> pd.DataFrame:
    """Tabla lista para modelar: predictores, objetivos y columnas de identificación.

    En desarrollo se eliminan filas con nulos en columnas casi completas (regla de la
    corrida anterior); en prueba se conservan todas.
    """
    tabla = renombrar(df)
    tabla = agregar_columnas_prerrequisito(tabla, definicion.prerrequisitos)
    for col in definicion.columnas_entrada:
        if col not in tabla:
            tabla[col] = float("nan")
    if eliminar_filas:
        revisar = definicion.columnas_entrada + [OBJETIVO_NOTA, OBJETIVO_RETIRO]
        tabla = tabla.loc[eliminar_filas_con_pocos_nulos(tabla[revisar]).index]
    return tabla


def _texto_categoria(valor):
    """Texto estable para una categoría: 123, 123.0 y '123' quedan como '123'."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return float("nan")
    if isinstance(valor, float) and valor.is_integer():
        return str(int(valor))
    texto = str(valor).strip()
    return texto or float("nan")


def matriz_predictores(tabla: pd.DataFrame, definicion: DefinicionAsignatura) -> pd.DataFrame:
    """Solo las columnas de entrada, con las categóricas como texto y el resto numérico."""
    X = tabla[definicion.columnas_entrada].copy()
    for col in X.columns:
        if col in definicion.categoricas:
            X[col] = X[col].astype(object).map(_texto_categoria)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    return X
