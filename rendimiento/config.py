"""Configuración única de la corrida final.

Todo lo que cambia entre ejecuciones (rutas, periodos, umbrales, presupuesto de
Optuna) vive aquí, para que el resto del código no tenga valores escondidos.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Configuracion:
    # --- Rutas -------------------------------------------------------------
    # Carpeta base en Google Drive (o local). Debe contener el parquet del historial.
    carpeta_base: Path = Path("/content/drive/MyDrive/ProyectoGrado")
    nombre_parquet: str = "historia_todos_2019_202610_dpto_asig_poblado.parquet"
    # Excel con la repitencia institucional 2024-30 (punto 7 del checklist).
    nombre_excel_repitencia: str = "Repitencia_asignaturas_punto7.xlsx"
    # Subcarpetas de salida (se crean dentro de carpeta_base).
    carpeta_modelos: str = "modelos_v3"
    carpeta_resultados: str = "resultados_v3"

    # --- Periodos ----------------------------------------------------------
    periodo_max_desarrollo: int = 202510   # último periodo usado para entrenar y validar
    periodo_prueba: int = 202530           # periodo reservado (evaluación final)

    # --- Selección de asignaturas -----------------------------------------
    max_asignaturas: int = 600             # las de mayor matrícula histórica
    matricula_minima: int = 200            # registros mínimos por asignatura
    # Para pruebas rápidas: lista de códigos a usar (None = todas).
    asignaturas_a_usar: list | None = None

    # --- Reglas de modelado -----------------------------------------------
    umbral_retiro_clasificador: float = 0.03   # % mínimo de retiros para entrenar clasificador
    n_folds: int = 5
    semilla: int = 42
    umbral_aprobacion: float = 3.0
    umbral_completitud_prerrequisito: float = 0.8
    # Regímenes del promedio acumulado (pga_inicial) según % de registros sin PGA.
    umbral_pga_sin_promedio: float = 0.80      # > 80 % vacío: se usan las pruebas Saber 11
    umbral_pga_imputar: float = 0.30           # entre 30 % y 80 %: se imputa desde Saber 11
    escala_promedio: tuple = (1.5, 5.0)        # escala institucional del promedio

    # --- Algoritmos y Optuna ----------------------------------------------
    algoritmos: tuple = ("RF", "XGB", "CAT")
    optuna_ensayos: int = 20                  # ensayos por modelo (calibrar con una prueba de tiempos)
    optuna_tiempo_max_seg: int | None = 90    # tope de la búsqueda por modelo; el ensayo en curso siempre termina
    # Métrica que optimiza Optuna en clasificación de retiro: "pr_auc", "f1_retiro" o "log_loss".
    metrica_optuna_clasificacion: str = "pr_auc"
    usar_gpu: bool = False                    # True si el entorno de Colab tiene GPU y se quiere usar
    # Parámetros fijos adicionales de CatBoost. En la prueba local, la codificación por
    # estadísticas de categorías (CTR) hizo cada ajuste ~40 veces más lento; si en Colab ocurre
    # lo mismo, usar {"one_hot_max_size": 255} (one-hot para categóricas de hasta 255 niveles).
    catboost_parametros_fijos: dict = field(default_factory=dict)

    # --- Línea base (regresión lineal) ------------------------------------
    asignaturas_linea_base: tuple = (
        "FIS1023", "IST2089", "MAT1031", "MAT1101",
        "MAT4258", "FIS1033", "CAS3030", "CAS3020",
    )

    # --- Análisis posteriores ---------------------------------------------
    algoritmo_elegido: str = "CAT"            # el que alimenta la propuesta de tutorías
    tutorias_top_n: int = 100
    tutorias_k_desviaciones: float = 1.0
    bootstrap_repeticiones: int = 2000
    ks_permutaciones: int = 9999
    medir_tiempo_shap: bool = True
    shap_asignaturas_muestra: int = 5
    shap_registros_muestra: int = 200

    # Columnas de identificación que se conservan en las predicciones.
    columnas_id: tuple = field(default=(
        "Periodo", "Pidm", "nrc", "Cod materia curso", "Descripcion_Materia",
    ))

    # --- Rutas derivadas ---------------------------------------------------
    @property
    def ruta_parquet(self) -> Path:
        return self.carpeta_base / self.nombre_parquet

    @property
    def ruta_excel_repitencia(self) -> Path:
        return self.carpeta_base / self.nombre_excel_repitencia

    @property
    def ruta_modelos(self) -> Path:
        return self.carpeta_base / self.carpeta_modelos

    @property
    def ruta_resultados(self) -> Path:
        return self.carpeta_base / self.carpeta_resultados

    def crear_carpetas(self) -> None:
        self.ruta_modelos.mkdir(parents=True, exist_ok=True)
        self.ruta_resultados.mkdir(parents=True, exist_ok=True)
