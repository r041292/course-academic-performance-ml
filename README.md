# Course-level prediction of academic performance

Code for a master's thesis project that estimates, **at the start of the semester**, how each undergraduate student is likely to do in each course they are enrolled in. The estimates are used to decide where academic tutoring should go.

For every course (subject), the project trains two models:

- a **regression model** that estimates the student's **final grade** (0–5 scale; 3.0 is the passing mark);
- a **classification model** that estimates whether the student will **withdraw** from the course.

The two predictions are combined into a single outcome per student and course: **pass**, **fail** or **withdraw**. These outcomes are aggregated by course and by course section to propose which courses and sections should receive tutoring support.

Three algorithms are compared: **Random Forest**, **XGBoost** and **CatBoost**. Hyperparameters are tuned with **Optuna** inside a 5-fold cross-validation, and the final models are evaluated on a **held-out academic term** that is never used for training or tuning.

> ⚠️ **The data are not included and cannot be shared.**
> The code runs on an institutional dataset with individual academic records (grades, withdrawals, prior-course results and socio-demographic attributes). It is confidential university information and cannot be published. **Without that file the code cannot be executed.** The repository is provided so the method can be inspected and reproduced on similar data. The expected input structure is described below.

---

## What the pipeline does

1. **Data preparation.** Loads the student–course history, applies cleaning rules (excluded departments and course types, homogenised categories, invalid cohorts) and keeps the courses with the largest historical enrolment (by default, up to 600 courses with at least 200 records).
2. **Course-specific feature definition.** Each course gets its own set of predictors, decided only from the training periods:
   - **Courses with prerequisites:** cumulative GPA, previous-term GPA, the student's **grade and number of attempts in each prerequisite course**, instructor-related variables and socio-demographic variables.
   - **Courses without prerequisites:** the treatment of GPA depends on how many students in the course have no GPA yet, which identifies courses taken mostly in the first semester:
     - more than 80 % without GPA → national entrance-exam scores are used instead;
     - between 30 % and 80 % → the missing GPA is imputed from the entrance-exam score (min–max scaling to the 1.5–5.0 GPA range) and a binary "imputed" flag is added;
     - 30 % or less → GPA is used directly.
3. **Hyperparameter search and cross-validation.** For each course, algorithm and task, Optuna (TPE sampler, median pruning) searches the hyperparameters. Every candidate is evaluated with the same 5 folds for all algorithms: `KFold` for grades, `StratifiedKFold` for withdrawal. The model is retrained in each fold, and the search stops after a fixed number of trials or a time limit per model.
4. **Final models.** Refitted on all training periods with the best hyperparameters and saved as complete scikit-learn pipelines.
5. **Inference** on the held-out term.
6. **Evaluation:** per-course and pooled metrics, a three-class confusion analysis, and an "institutional accuracy" indicator (see below).
7. **Statistical comparison of algorithms:** Friedman test, Kendall's W, pairwise Wilcoxon tests with Holm correction, and course-level bootstrap confidence intervals.
8. **Distribution-shift analysis:** compares each course's grade distribution between the training periods and the held-out term (permutation Kolmogorov–Smirnov test with Benjamini–Hochberg correction, and Wasserstein distance).
9. **Tutoring allocation proposal** and its validation against the real outcomes of the held-out term.
10. **Interpretability:** optional SHAP explanations per student, and a timing benchmark of SHAP for each algorithm.

### Key modelling choices

- **Leakage control.** Everything learned from data lives inside a scikit-learn `Pipeline` and is fitted only on the training fold: the GPA imputation scale, the set of known categories and the categorical encoding. Unseen categories in the test term (for example, a new instructor) are handled explicitly.
  - CatBoost uses native categorical features.
  - XGBoost uses fixed pandas categories; unseen values become missing.
  - Random Forest uses an `OrdinalEncoder`; unseen values become −1.
- **Withdrawal classifier.** Trained only when withdrawals are more than 3 % of the course's records and there are at least 5 records of each class. Decision threshold: 0.5. XGBoost and CatBoost weight the minority class by the negative/positive ratio; Random Forest uses `class_weight="balanced"`.
- **Combining the two models.** If the classifier predicts withdrawal, the outcome is *withdraw*. Otherwise the estimated grade decides: *pass* if it is 3.0 or higher, *fail* if it is lower. In courses without a classifier, the grade alone decides.
- **Reproducibility.** Fixed random seed (42) for folds, models and the Optuna sampler. Library versions are written to `versiones.csv`. Because the search has a time limit per model, a slower machine may complete fewer trials; the number of trials per model is recorded.
- **Optimism of cross-validation.** The same folds are used to choose hyperparameters and to report validation metrics, so those metrics are slightly optimistic. The held-out term is the unbiased evaluation.

---

## Expected input data (not included)

| File | Content |
|---|---|
| `historia_todos_2019_202610_dpto_asig_poblado.parquet` | One row per student × course × term. Main fields: term (`Periodo`), anonymised student id (`Pidm`), section (`nrc`), course code and name, prerequisite status, final status (passed / failed / withdrawn), final grade, cumulative and previous-term GPA, instructor code and instructor history, entrance-exam scores (overall and by area), demographic attributes (sex, age, socio-economic stratum, origin, school type and calendar), academic-support attendance, and up to *n* prerequisite slots with each prerequisite's code, grade and number of attempts. |
| `Repitencia_asignaturas_punto7.xlsx` *(optional)* | Course-level failure-and-withdrawal rate from the previous equivalent term, used as the current institutional baseline to validate the tutoring proposal. |

Column names follow the institution's data warehouse; the renaming map is in `rendimiento/variables.py`.

---

## How to run (Google Colab)

1. Put this folder, the parquet file and (optionally) the Excel file in a Google Drive folder, for example `MyDrive/ProyectoGrado`.
2. Open `ejecutar_corrida_colab.ipynb` in Colab, choose a **CPU runtime** (high-RAM if available) and set `CARPETA_BASE` to that folder.
3. Run the **calibration** cell first (3 courses). It reports the time of each stage, so you can adjust `optuna_ensayos` (trials per model) and `optuna_tiempo_max_seg` (time limit per model).
4. Run the full pipeline stage by stage. **Training is resumable:** each finished course is saved, so after a disconnection you only need to re-run the setup cells, `etapa_datos()` and `etapa_entrenamiento()`.

The code can also run outside Colab: install `pandas pyarrow scikit-learn xgboost catboost optuna shap openpyxl`, create a `Configuracion(...)` pointing to local paths and call `Corrida(cfg).ejecutar_todo()`.

GPUs are not needed. The per-course datasets are small (a few hundred to a few thousand rows), and the running time is driven by the per-model search time limit, not by hardware speed.

---

## Repository structure

```
├── ejecutar_corrida_colab.ipynb   # notebook that runs every stage in Colab
└── rendimiento/
    ├── config.py            # all run parameters (paths, terms, thresholds, Optuna budget)
    ├── limpieza.py          # loading, cleaning and course selection
    ├── variables.py         # per-course predictor definition (prerequisites, GPA regime)
    ├── preprocesamiento.py  # scikit-learn pipelines: GPA imputation, categorical handling, model
    ├── optimizacion.py      # Optuna search inside 5-fold cross-validation
    ├── entrenamiento.py     # per-course training and linear-regression baseline
    ├── inferencia.py        # predictions for the held-out term and the combined outcome
    ├── interpretacion.py    # SHAP explanations per student (optional stage)
    ├── metricas.py          # metrics with explicit names (withdrawal class, non-withdrawal, macro)
    ├── evaluacion.py        # result tables
    ├── estadistica.py       # Friedman, bootstrap, Kolmogorov–Smirnov and Wasserstein
    ├── tutorias.py          # tutoring allocation proposal and validation
    ├── corrida.py           # stage orchestrator
    └── utilidades.py        # timing log, CSV writing, library versions
```

Code identifiers and output file names are in Spanish. The tables below explain each one.

---

## Outputs

All results are written as CSV files (`;` separator, decimal comma) to `resultados_v3/`. Models are written to `modelos_v3/<ALGORITHM>/<task>/<course>.joblib` (one complete pipeline each), with one JSON per course in `modelos_v3/definiciones/` describing its predictors, categorical variables, prerequisites and GPA regime.

### Run information

| File | What it contains |
|---|---|
| `datos_resumen` | Record counts after cleaning, number of candidate and modelled courses, training and test records, and the terms used. |
| `asignaturas_modeladas` | List of modelled courses. |
| `estado_modelos` | For each course and task: whether a model was trained and, if not, why (e.g., too few withdrawals). |
| `versiones` | Python and library versions, platform, and whether a GPU was used. |
| `tiempos_ejecucion` | Seconds spent in each stage, by course, algorithm and task: data loading, hyperparameter search with cross-validation, final fit, inference, statistics and SHAP timing. |

### Training and hyperparameters

| File | What it contains |
|---|---|
| `optuna_mejores_parametros` | One row per model (course × algorithm × task): number of records and predictors, GPA regime, optimised metric, trials run / completed / pruned, best score, the hyperparameters chosen by Optuna, and all effective parameters of the final model. |
| `cv_por_fold` | Cross-validation metrics of the best configuration, one row per fold. |
| `cv_por_asignatura` | Mean and standard deviation across folds of each metric, per course, algorithm and task. |
| `predicciones_desarrollo_oof` | One row per training record: identifiers, the predictors exactly as fed to the model, the real grade and withdrawal, and each algorithm's **out-of-fold** prediction (made when that record was in the validation fold). |
| `linea_base_por_fold` | Linear-regression baseline for a fixed set of reference courses: metrics per fold (same folds as the other models) and on the held-out term. |

### Predictions on the held-out term

| File | What it contains |
|---|---|
| `predicciones_2025_30` | One row per student × course in the test term: identifiers and section, real status, real grade and real three-class outcome, the predictors used, and for each algorithm the estimated grade, the withdrawal probability and prediction, and the combined outcome (0 = pass, 1 = fail, 2 = withdraw). |
| `predicciones_2025_30_con_shap` *(optional)* | The same, plus a SHAP summary for each student: the variables with the largest average impact in the course, the variables pushing that student's estimate up and down, and the model's base value. |
| `valores_shap_2025_30` *(optional)* | Long table with the numeric SHAP value of every variable for every student. |

### Evaluation tables

| File | What it contains |
|---|---|
| `tabla08_validacion_cruzada` | Cross-validation summary: for each algorithm and metric, the mean across courses of the per-course mean and of the per-course fold standard deviation. Grades: MAE, RMSE, R². Withdrawal: accuracy, precision, recall and F1 for the withdrawal class, macro F1, ROC-AUC, PR-AUC. |
| `prueba_por_asignatura` | Test-term metrics for each course and algorithm, including the confusion counts of the withdrawal classifier. |
| `tabla09_prueba_por_asignatura` | Mean and standard deviation across courses of the test-term metrics. Withdrawal-class metrics exclude courses with no real withdrawals in the term, where recall is undefined; the number excluded is reported. |
| `tabla_nueva_distribucion_r2` | Distribution of per-course R² on the test term (median, quartiles, minimum, number and share of negative values) for each algorithm, plus the median and quartiles of the MAE. |
| `caso_r2_extremo` | The course with the lowest R²: its grade spread, the error of predicting the term average, and the model's error, which explain why R² can be very negative when grades barely vary. |
| `tabla10_global` | Metrics computed once over all test records pooled together (not averaged by course). |
| `tablas11_12_matrices_3_clases` | Confusion matrix of the combined three-class outcome (pass / fail / withdraw) for each algorithm. |
| `tabla13_metricas_por_clase` | Precision, recall and F1 for each of the three classes, their macro and weighted averages, and the overall accuracy. |
| `tablas14_15_acierto_institucional` | **Institutional accuracy**: a prediction counts as correct when it matches the real pass/fail outcome of students who completed the course, or when it flags a withdrawn student as fail or withdraw (both unfavourable). A predicted withdrawal for a student who completed the course counts as an error. Each category is reported as a count, as a share of all records and as a share of its real outcome group. A symmetric variant is included as a sensitivity check. |
| `punto2_anexoC_no_aprobacion` | Binary view, pass vs. not pass (fail or withdraw): full confusion counts, recall and precision for "not pass", number of unfavourable predictions, and the real pass rate. |
| `punto17_retiros_sin_clasificador` | How many courses, records and real withdrawals fall in courses with and without a withdrawal classifier. Withdrawals in courses without one can only be detected indirectly through a low estimated grade. |
| `punto24_conteo_asignaturas` | How many modelled courses were evaluated in the test term, how many were not offered, how many had no valid records, and the number of records per real outcome. |
| `estadisticas_formato_anterior` | All cross-validation and test metrics in one long table (`mean`, `std`, `metric`, set, course, model, task), the format used in earlier versions of the project. |

### Statistical comparison

| File | What it contains |
|---|---|
| `tablas16_17_friedman` | For each cross-validation metric: Friedman χ² and p-value across courses, Kendall's W (effect size, 0–1), mean rank of each algorithm (1 = best) and Holm-adjusted p-values of the pairwise Wilcoxon tests. |
| `tabla18_bootstrap` | Pairwise differences between algorithms on the test term with 95 % confidence intervals from resampling whole courses: not-pass recall and precision, MAE, macro F1, per-class F1 and institutional accuracy. |
| `punto9_ks_por_asignatura` | For each course: number of grades in training and test, mean grades, Kolmogorov–Smirnov D, permutation p-value, Benjamini–Hochberg q-value, Wasserstein distance (in grade points) and each algorithm's R². |
| `punto9_ks_resumen` | Whether grade-distribution shifts are more frequent and larger in courses where the models fit poorly (R² < 0) than in the rest: proportions with a significant shift, Fisher's exact test, Mann–Whitney test on Wasserstein distances, and Spearman correlations between shift size and R² or MAE. |

### Tutoring allocation

| File | What it contains |
|---|---|
| `tutorias_asignaturas` | One row per course (grouped by name): enrolment, estimated and real number and rate of students who will not pass, whether the course is "high risk" (estimated count above the mean plus one standard deviation), its prioritised section if any, its rank, whether it is among the top-N selected courses, and the type of support assigned (whole course, specific section, or both). |
| `tutorias_nrc` | One row per section: enrolment, estimated and real counts, and the within-course thresholds used to flag outlier sections. |
| `tutorias_conteos_para_texto` | Summary counts of the proposal (high-risk courses, flagged sections, support types, ties at the cut-off). |
| `tutorias_validacion` | How well the proposal anticipated the real outcomes. **Course level:** share of all real non-passing students covered by the model's top-N courses vs. the top-N by the institutional baseline, overlap between both selections, and Spearman correlations. **Section level:** *lift* (share of non-passing students in flagged sections divided by their share of enrolment), precision and Spearman correlation. |
| `tutorias_control_sensatez` | Courses whose estimated non-pass rate is far from the real one, listed for manual review. |

---

## Data

The code is shared for transparency. The underlying data belong to the university and are not distributed.
