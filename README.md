# Código de la corrida final

Predicción del rendimiento académico por asignatura: nota final (regresión) y retiro (clasificación) con Random Forest, XGBoost y CatBoost, evaluados en el periodo reservado 2025-30.

## Cómo ejecutarlo en Google Colab

1. Sube a una carpeta de Google Drive, por ejemplo `MyDrive/ProyectoGrado`:
   - esta carpeta completa, `codigo_corrida_final`;
   - el parquet del historial, `historia_todos_2019_202610_dpto_asig_poblado.parquet`;
   - el Excel `Repitencia_asignaturas_punto7.xlsx` (criterio institucional para validar las tutorías).
2. Abre `ejecutar_corrida_colab.ipynb` en Colab y ajusta `CARPETA_BASE` en la celda de configuración.
3. Ejecuta primero la **calibración** (3 asignaturas). Con los tiempos que muestra, ajusta `optuna_ensayos` y `optuna_tiempo_max_seg`.
4. Ejecuta la corrida completa, etapa por etapa. **El entrenamiento es reanudable:** si Colab se desconecta, vuelve a ejecutar las celdas de configuración, `etapa_datos` y `etapa_entrenamiento`; las asignaturas terminadas se omiten.

**Estimación de tiempos** (prueba local, 12 núcleos, 2 ensayos de Optuna por modelo):

| Algoritmo | Búsqueda + CV, por modelo | Ajuste final |
|---|---|---|
| XGBoost | ~3 s | 0,3 s |
| Random Forest | ~8 s | 0,8 s |
| CatBoost (con `one_hot_max_size=255`) | ~36 s | 6 s |

Con 15 ensayos el tiempo crece en proporción, aunque la poda de Optuna lo reduce, y `optuna_tiempo_max_seg` pone un tope por modelo. Para 513 asignaturas, conviene repartir la corrida en varias sesiones de Colab, porque es reanudable.

## Estructura

```
codigo_corrida_final/
├── ejecutar_corrida_colab.ipynb   # notebook de ejecución
└── rendimiento/
    ├── config.py            # todos los parámetros de la corrida
    ├── limpieza.py          # carga, limpieza y selección de asignaturas
    ├── variables.py         # predictores de cada asignatura (prerrequisitos, régimen del promedio)
    ├── preprocesamiento.py  # pipelines de scikit-learn (imputación del PGA, categóricas, modelo)
    ├── optimizacion.py      # Optuna dentro de la validación cruzada de 5 folds
    ├── entrenamiento.py     # entrenamiento por asignatura y línea base lineal
    ├── inferencia.py        # predicciones 2025-30 y clase integrada
    ├── metricas.py          # métricas con nombres explícitos (clase retiro, no retiro, macro)
    ├── evaluacion.py        # tablas del documento
    ├── estadistica.py       # Friedman, bootstrap, KS y Wasserstein
    ├── tutorias.py          # propuesta de tutorías y su validación
    ├── corrida.py           # orquestador por etapas
    └── utilidades.py        # tiempos, CSV y versiones
```

## Decisiones de diseño

- **Validación cruzada real.** 5 folds sobre todos los datos anteriores a 2025-30: `KFold` con mezcla en regresión y `StratifiedKFold` por retiro en clasificación. El modelo se reentrena en cada fold, y los folds son los mismos para los tres algoritmos. No hay división 80/20. El modelo final se ajusta con todos los datos de desarrollo.
- **Optuna.** Muestreador TPE (semilla 42) y poda por mediana. Optimiza el MAE en regresión y la PR-AUC en clasificación (configurable). Las métricas de validación que se reportan son las del mejor ensayo, por lo que son algo optimistas; la evaluación no sesgada es 2025-30.
- **Pipelines.** Todo lo que se aprende de los datos queda dentro del pipeline y se ajusta solo con el fold de entrenamiento:
  - la escala de la imputación del PGA (Min-Max de Saber 11 a [1,50; 5,00], más el indicador `promedio_imputado`);
  - las categorías conocidas. XGBoost usa categorías fijas y una categoría nueva queda como faltante. CatBoost usa sus categóricas nativas, incluido el clasificador, con lo que se corrige la nota N2. Random Forest usa `OrdinalEncoder`, que asigna −1 a lo desconocido.
- **Regímenes del promedio** en asignaturas sin prerrequisitos, según el % de registros sin PGA en los datos de desarrollo:
  - más del 80 %: pruebas de Saber 11;
  - entre 30 % y 80 %: PGA imputado más las pruebas de Saber 11 sin el puntaje global;
  - 30 % o menos: PGA y promedio semestral.
- **Clasificador de retiro.** Solo si el retiro supera el 3 % y hay al menos 5 registros de cada clase. Umbral de decisión 0,5. XGBoost y CatBoost ponderan la clase retiro; Random Forest usa `class_weight="balanced"`.
- **Diferencia con la corrida anterior.** La eliminación de filas con pocos nulos se aplica solo a los datos de desarrollo (antes también miraba 2025-30). En la inferencia no se eliminan filas.

## Correspondencia CSV → documento

| CSV (`resultados_v3/`) | Uso en el documento |
|---|---|
| `datos_resumen`, `asignaturas_modeladas` | Datos y alcance (513 asignaturas, registros) |
| `versiones` | Párrafo de reproducibilidad (punto 19) |
| `tiempos_ejecucion` | Tiempos por etapa, algoritmo y asignatura (R4); costo de SHAP (C8) |
| `estado_modelos` | Asignaturas con y sin clasificador, y motivo |
| `optuna_mejores_parametros` | Registro de hiperparámetros por modelo (asignatura × algoritmo × tarea). Incluye `parametros_optuna` (los elegidos), `parametros_efectivos` (todos los del modelo final, incluida la ponderación de clases), ensayos completos y podados, métrica optimizada, n de registros y de predictores. Base del Anexo B |
| `predicciones_desarrollo_oof` | Datos de entrenamiento por registro: identificación, **variables tal como entran al modelo**, resultado real y predicción fuera de muestra (del fold en que el registro fue validación) de cada algoritmo |
| `predicciones_2025_30` | Inferencia por registro: identificación, estado real, **variables usadas en la inferencia**, nota estimada, probabilidad y predicción de retiro, y clase integrada de cada algoritmo |
| `estadisticas_formato_anterior` | Formato largo como el de las corridas anteriores (`mean`, `std`, `metric`, `Conjunto`, `CV o Directo`, `Descripcion_Materia`, `Cod materia curso`, `Modelo`, `Tipo_Modelo`), con la validación cruzada y 2025-30 |
| `cv_por_fold`, `cv_por_asignatura` | Base de la Tabla 8 y de Friedman |
| `tabla08_validacion_cruzada` | Tabla 8 |
| `linea_base_por_fold` | Tablas 6 y 7; línea base en 2025-30 |
| `prueba_por_asignatura`, `tabla09_prueba_por_asignatura` | Tabla 9 (clase retiro; excluye asignaturas sin retiros) |
| `tabla_nueva_distribucion_r2`, `caso_r2_extremo` | Tabla nueva de R² y ejemplo tipo LEY4625 (punto 10) |
| `tabla10_global` | Tabla 10 |
| `tablas11_12_matrices_3_clases` | Tablas 11 y 12 |
| `tabla13_metricas_por_clase` | Tabla 13 (incluye accuracy) |
| `tablas14_15_acierto_institucional` | Tablas 14 y 15, y la sensibilidad con regla simétrica (punto 1) |
| `punto2_anexoC_no_aprobacion` | Punto 2 y Anexo C (matrices binarias) |
| `punto17_retiros_sin_clasificador` | Punto 17 |
| `punto24_conteo_asignaturas` | Punto 24 (asignaturas evaluadas y no evaluadas) |
| `tablas16_17_friedman` | Tablas 16 y 17 (Friedman, W de Kendall, Wilcoxon-Holm) |
| `tabla18_bootstrap` | Tabla 18 ampliada (punto 13) |
| `punto9_ks_por_asignatura`, `punto9_ks_resumen` | Cambio de distribución con grupo de control (punto 9) |
| `tutorias_asignaturas`, `tutorias_nrc`, `tutorias_conteos_para_texto` | Propuesta de tutorías y Figura 10 (punto 8) |
| `tutorias_validacion` | Comparación con el criterio 2024-30 y lift de NRC (punto 7) |
| `tutorias_control_sensatez` | Asignaturas con repitencia estimada muy distinta de la real |

Además, cada asignatura deja en `resultados_v3/parciales/` sus archivos `cv_`, `optuna_`, `estado_` y `desarrollo_`, que permiten reanudar el entrenamiento. Los modelos quedan en `modelos_v3/<ALGORITMO>/<tarea>/<asignatura>.joblib` (pipeline completo) y la definición de cada asignatura en `modelos_v3/definiciones/<asignatura>.json`: predictores, categóricas, prerrequisitos y régimen del promedio.
