"""Orquestador de la corrida completa.

Etapas (cada una deja sus CSV en carpeta_resultados):
  1. datos         – carga, limpieza y selección de asignaturas.
  2. entrenamiento – Optuna + validación cruzada + modelo final por asignatura (reanudable).
  3. linea_base    – regresión lineal en las 8 asignaturas de referencia.
  4. inferencia    – predicciones en 2025-30.
  5. evaluacion    – tablas 8 a 15, R², no aprobación, alcance del clasificador.
  6. estadistica   – Friedman, bootstrap y cambio de distribución de notas.
  7. tutorias      – propuesta de apoyos y su validación.
  8. tiempo_shap   – (opcional) costo de SHAP por algoritmo en una muestra.
Rutina adicional (se ejecuta aparte, después de la corrida):
  inferencia_con_shap – predicciones 2025-30 con la interpretación SHAP por registro.
"""

import time

import joblib
import pandas as pd

from . import estadistica, evaluacion, tutorias
from .entrenamiento import entrenar_asignatura, linea_base, ruta_definicion, ruta_modelo
from .inferencia import predecir_periodo
from .interpretacion import calcular_shap, inferencia_con_shap
from .limpieza import (COLUMNAS_IGNORADAS_ELIMINACION_GLOBAL, eliminar_filas_con_pocos_nulos,
                       limpieza_inicial, nombres_de_asignaturas, preparar_registros, seleccionar_asignaturas)
from .utilidades import RegistroTiempos, guardar_csv, leer_csv, log, tabla_versiones


class Corrida:
    def __init__(self, cfg):
        self.cfg = cfg
        cfg.crear_carpetas()
        self.tiempos = RegistroTiempos()
        self.resultados = cfg.ruta_resultados

    # -- utilidades ---------------------------------------------------------
    def _guardar(self, df, nombre):
        guardar_csv(df, self.resultados / f"{nombre}.csv")

    def _guardar_tiempos(self):
        ruta = self.resultados / "tiempos_ejecucion.csv"
        nuevos = self.tiempos.como_tabla()
        if ruta.exists():
            nuevos = pd.concat([leer_csv(ruta), nuevos], ignore_index=True)
        guardar_csv(nuevos, ruta)
        self.tiempos = RegistroTiempos()

    # -- 1. datos -----------------------------------------------------------
    def etapa_datos(self):
        cfg = self.cfg
        with self.tiempos.medir("carga_y_limpieza"):
            historial = limpieza_inicial(pd.read_parquet(cfg.ruta_parquet))
            self.nombres = nombres_de_asignaturas(historial)
            asignaturas = cfg.asignaturas_a_usar or seleccionar_asignaturas(
                historial, cfg.max_asignaturas, cfg.matricula_minima)
            registros = preparar_registros(historial, asignaturas, eliminar_filas=False)

            desarrollo = registros[registros["Periodo"] <= cfg.periodo_max_desarrollo]
            desarrollo = eliminar_filas_con_pocos_nulos(desarrollo, ignorar=COLUMNAS_IGNORADAS_ELIMINACION_GLOBAL,
                                                        ignorar_prerrequisitos=True)
            conteo = desarrollo.groupby("Cod materia curso").size()
            self.asignaturas = [a for a in asignaturas if conteo.get(a, 0) >= cfg.matricula_minima]
            self.historial = historial
            self.desarrollo = desarrollo
            self.prueba = registros[registros["Periodo"] == cfg.periodo_prueba]
        resumen = pd.DataFrame([
            ("registros_historial_limpio", len(historial)),
            ("asignaturas_candidatas", len(asignaturas)),
            ("asignaturas_modeladas", len(self.asignaturas)),
            ("registros_desarrollo", int(desarrollo["Cod materia curso"].isin(self.asignaturas).sum())),
            ("registros_prueba", int(self.prueba["Cod materia curso"].isin(self.asignaturas).sum())),
            ("periodo_max_desarrollo", cfg.periodo_max_desarrollo),
            ("periodo_prueba", cfg.periodo_prueba),
        ], columns=["concepto", "valor"])
        self._guardar(resumen, "datos_resumen")
        self._guardar(pd.DataFrame({"asignatura": self.asignaturas}), "asignaturas_modeladas")
        self._guardar(tabla_versiones(cfg.usar_gpu), "versiones")
        self._guardar_tiempos()
        log.info("Asignaturas a modelar: %d", len(self.asignaturas))

    # -- 2. entrenamiento ---------------------------------------------------
    def etapa_entrenamiento(self):
        """Reanudable: si una asignatura ya tiene definición y resultados parciales, se omite."""
        parciales = self.resultados / "parciales"
        inicio = time.perf_counter()
        for numero, asignatura in enumerate(self.asignaturas, start=1):
            archivo_cv = parciales / f"cv_{asignatura}.csv"
            if archivo_cv.exists() and ruta_definicion(self.cfg, asignatura).exists():
                continue
            log.info("[%d/%d] Entrenando %s", numero, len(self.asignaturas), asignatura)
            try:
                datos = self.desarrollo[self.desarrollo["Cod materia curso"] == asignatura]
                salida = entrenar_asignatura(datos, asignatura, self.nombres, self.cfg, self.tiempos)
                guardar_csv(salida["optuna"], parciales / f"optuna_{asignatura}.csv")
                guardar_csv(salida["estado"], parciales / f"estado_{asignatura}.csv")
                guardar_csv(salida["registros"], parciales / f"desarrollo_{asignatura}.csv")
                guardar_csv(salida["cv"], archivo_cv)          # último: marca la asignatura como terminada
            except Exception as error:
                log.warning("Entrenamiento fallido en %s: %s", asignatura, error)
            self._guardar_tiempos()
        self.tiempos.filas.append({"etapa": "entrenamiento_total_esta_sesion",
                                   "segundos": round(time.perf_counter() - inicio, 1)})
        self._unir_parciales()

    def _unir_parciales(self):
        parciales = self.resultados / "parciales"
        for prefijo, nombre in (("cv", "cv_por_fold"), ("optuna", "optuna_mejores_parametros"),
                                ("estado", "estado_modelos"), ("desarrollo", "predicciones_desarrollo_oof")):
            archivos = sorted(parciales.glob(f"{prefijo}_*.csv"))
            if archivos:
                self._guardar(pd.concat([leer_csv(a) for a in archivos], ignore_index=True), nombre)
        self._guardar_tiempos()

    # -- 3. línea base ------------------------------------------------------
    def etapa_linea_base(self):
        partes = []
        for asignatura in self.cfg.asignaturas_linea_base:
            if asignatura not in self.asignaturas:
                continue
            with self.tiempos.medir("linea_base", asignatura, "LIN", "regresion"):
                partes.append(linea_base(self.desarrollo[self.desarrollo["Cod materia curso"] == asignatura],
                                         self.prueba[self.prueba["Cod materia curso"] == asignatura],
                                         asignatura, self.nombres, self.cfg))
        if partes:
            self._guardar(pd.concat(partes, ignore_index=True), "linea_base_por_fold")
        self._guardar_tiempos()

    # -- 4. inferencia ------------------------------------------------------
    def etapa_inferencia(self):
        with self.tiempos.medir("inferencia_total"):
            self.predicciones = predecir_periodo(self.prueba, self.asignaturas, self.cfg, self.tiempos)
        self._guardar(self.predicciones, "predicciones_2025_30")
        self._guardar_tiempos()

    # -- 5. evaluación ------------------------------------------------------
    def etapa_evaluacion(self):
        cfg, pred = self.cfg, self.predicciones
        algoritmos = list(cfg.algoritmos)
        with self.tiempos.medir("evaluacion"):
            cv = leer_csv(self.resultados / "cv_por_fold.csv")
            self.cv_asig = evaluacion.cv_por_asignatura(cv)
            self.por_asig = evaluacion.metricas_prueba_por_asignatura(pred, algoritmos)
            tablas = {
                "cv_por_asignatura": self.cv_asig,
                "tabla08_validacion_cruzada": evaluacion.tabla8_validacion_cruzada(self.cv_asig),
                "prueba_por_asignatura": self.por_asig,
                "tabla09_prueba_por_asignatura": evaluacion.tabla9_prueba_por_asignatura(self.por_asig),
                "tabla_nueva_distribucion_r2": evaluacion.tabla_distribucion_r2(self.por_asig),
                "caso_r2_extremo": evaluacion.caso_r2_extremo(self.por_asig, pred, cfg.algoritmo_elegido),
                "tabla10_global": evaluacion.tabla10_global(pred, algoritmos),
                "tablas11_12_matrices_3_clases": evaluacion.matrices_confusion_3_clases(pred, algoritmos),
                "tabla13_metricas_por_clase": evaluacion.tabla13_metricas_por_clase(pred, algoritmos),
                "tablas14_15_acierto_institucional": evaluacion.tablas14_15_acierto_institucional(pred, algoritmos),
                "punto2_anexoC_no_aprobacion": evaluacion.no_aprobacion(pred, algoritmos),
                "punto17_retiros_sin_clasificador": evaluacion.retiros_sin_clasificador(pred),
                "punto24_conteo_asignaturas": evaluacion.conteo_asignaturas(
                    self.asignaturas, self.prueba, pred, self.historial, cfg.periodo_prueba),
                "estadisticas_formato_anterior": evaluacion.estadisticas_formato_anterior(
                    self.cv_asig, self.por_asig, self.nombres),
            }
        for nombre, tabla in tablas.items():
            self._guardar(tabla, nombre)
        self._guardar_tiempos()

    # -- 6. estadística -----------------------------------------------------
    def etapa_estadistica(self):
        cfg = self.cfg
        with self.tiempos.medir("friedman"):
            self._guardar(estadistica.comparacion_friedman(self.cv_asig), "tablas16_17_friedman")
        with self.tiempos.medir("bootstrap"):
            self._guardar(estadistica.bootstrap_por_asignaturas(self.predicciones, list(cfg.algoritmos),
                                                                cfg.bootstrap_repeticiones, cfg.semilla),
                          "tabla18_bootstrap")
        with self.tiempos.medir("cambio_distribucion_notas"):
            detalle, resumen = estadistica.cambio_distribucion_notas(
                self.historial, self.por_asig, sorted(self.predicciones["Cod materia curso"].unique()),
                cfg, cfg.algoritmo_elegido)
        self._guardar(detalle, "punto9_ks_por_asignatura")
        self._guardar(resumen, "punto9_ks_resumen")
        self._guardar_tiempos()

    # -- 7. tutorías --------------------------------------------------------
    def etapa_tutorias(self):
        cfg = self.cfg
        with self.tiempos.medir("propuesta_tutorias"):
            asignaturas, nrc, conteos = tutorias.construir_propuesta(
                self.predicciones, cfg.algoritmo_elegido, cfg.tutorias_top_n, cfg.tutorias_k_desviaciones)
            self._guardar(asignaturas, "tutorias_asignaturas")
            self._guardar(nrc, "tutorias_nrc")
            self._guardar(conteos, "tutorias_conteos_para_texto")
            self._guardar(tutorias.control_sensatez(asignaturas), "tutorias_control_sensatez")
            if cfg.ruta_excel_repitencia.exists():
                institucional = pd.read_excel(cfg.ruta_excel_repitencia, sheet_name="Asignaturas")
                self._guardar(tutorias.validar_propuesta(asignaturas, nrc, self.predicciones, institucional,
                                                         cfg.tutorias_top_n), "tutorias_validacion")
            else:
                log.warning("No se encontró %s; se omite la validación de la propuesta.", cfg.ruta_excel_repitencia)
        self._guardar_tiempos()

    # -- 8. tiempo de SHAP (respaldo del argumento sobre Random Forest) ------
    def etapa_tiempo_shap(self):
        cfg = self.cfg
        if not cfg.medir_tiempo_shap:
            return
        from .entrenamiento import cargar_definicion
        from .variables import construir_tabla, matriz_predictores

        muestra = [a for a in self.asignaturas if ruta_definicion(cfg, a).exists()][:cfg.shap_asignaturas_muestra]
        for asignatura in muestra:
            definicion = cargar_definicion(cfg, asignatura)
            datos = self.desarrollo[self.desarrollo["Cod materia curso"] == asignatura]
            X = matriz_predictores(construir_tabla(datos, definicion, eliminar_filas=True), definicion)
            X = X.sample(min(cfg.shap_registros_muestra, len(X)), random_state=cfg.semilla)
            for algoritmo in cfg.algoritmos:
                pipeline = joblib.load(ruta_modelo(cfg, algoritmo, "regresion", asignatura))
                X_modelo = pipeline[:-1].transform(X)
                try:
                    with self.tiempos.medir("tiempo_shap_regresion", asignatura, algoritmo, "regresion", len(X_modelo)):
                        calcular_shap(algoritmo, pipeline[-1], X_modelo, definicion.categoricas)
                except Exception as error:
                    log.warning("SHAP falló en %s/%s: %s", asignatura, algoritmo, error)
        self._guardar_tiempos()

    # -- rutina adicional: inferencia 2025-30 con SHAP ------------------------
    def etapa_inferencia_con_shap(self, algoritmos=("CAT", "XGB"), top_n=10, exportar_valores=False):
        """Predicciones 2025-30 con las columnas interpretacion_general_<ALG>, interpretacion_registro_<ALG>
        y shap_valor_base_<ALG>, como en las corridas anteriores. Usa los modelos ya guardados.

        Requiere haber ejecutado antes `etapa_datos()` en la misma sesión.
        `exportar_valores=True` guarda además los valores SHAP numéricos en formato largo.
        """
        with self.tiempos.medir("inferencia_con_shap_total"):
            predicciones, valores = inferencia_con_shap(self.prueba, self.asignaturas, self.cfg, self.tiempos,
                                                        algoritmos=tuple(algoritmos), top_n=top_n,
                                                        exportar_valores=exportar_valores)
        self._guardar(predicciones, "predicciones_2025_30_con_shap")
        if exportar_valores and len(valores):
            self._guardar(valores, "valores_shap_2025_30")
        self._guardar_tiempos()
        return predicciones

    # -- todo ---------------------------------------------------------------
    def ejecutar_todo(self):
        self.etapa_datos()
        self.etapa_entrenamiento()
        self.etapa_linea_base()
        self.etapa_inferencia()
        self.etapa_evaluacion()
        self.etapa_estadistica()
        self.etapa_tutorias()
        self.etapa_tiempo_shap()
        log.info("Corrida completa. Resultados en %s", self.resultados)
