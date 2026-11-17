"""Métricas con nombres explícitos: siempre se dice de qué clase o promedio se trata."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, log_loss, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)

MIN_REGISTROS_R2 = 5


def metricas_regresion(y_real, y_pred) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2_calculable = len(y_real) >= MIN_REGISTROS_R2 and np.var(y_real) > 1e-12
    return {
        "mae": mean_absolute_error(y_real, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "r2": r2_score(y_real, y_pred) if r2_calculable else np.nan,
        "n": len(y_real),
    }


def metricas_clasificacion(y_real, y_pred, probabilidad=None) -> dict:
    """Retiro es la clase positiva (1). Se reportan la clase retiro, la clase no retiro y el macro."""
    y_real = np.asarray(y_real, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    hay_dos_clases = len(np.unique(y_real)) == 2
    resultado = {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision_retiro": precision_score(y_real, y_pred, pos_label=1, zero_division=0),
        "recall_retiro": recall_score(y_real, y_pred, pos_label=1, zero_division=0),
        "f1_retiro": f1_score(y_real, y_pred, pos_label=1, zero_division=0),
        "precision_no_retiro": precision_score(y_real, y_pred, pos_label=0, zero_division=0),
        "recall_no_retiro": recall_score(y_real, y_pred, pos_label=0, zero_division=0),
        "f1_no_retiro": f1_score(y_real, y_pred, pos_label=0, zero_division=0),
        "f1_macro": f1_score(y_real, y_pred, average="macro", labels=[0, 1], zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "log_loss": np.nan,
        "n": len(y_real),
        "n_retiros_reales": int((y_real == 1).sum()),
        "n_retiros_predichos": int((y_pred == 1).sum()),
        "vp_retiro": int(((y_real == 1) & (y_pred == 1)).sum()),
        "fp_retiro": int(((y_real == 0) & (y_pred == 1)).sum()),
        "fn_retiro": int(((y_real == 1) & (y_pred == 0)).sum()),
        "vn_retiro": int(((y_real == 0) & (y_pred == 0)).sum()),
    }
    if probabilidad is not None and hay_dos_clases:
        probabilidad = np.clip(np.asarray(probabilidad, dtype=float), 1e-6, 1 - 1e-6)
        resultado["roc_auc"] = roc_auc_score(y_real, probabilidad)
        resultado["pr_auc"] = average_precision_score(y_real, probabilidad)
        resultado["log_loss"] = log_loss(y_real, probabilidad, labels=[0, 1])
    return resultado


# Métrica que optimiza Optuna y su dirección.
OBJETIVOS_OPTUNA = {
    "regresion": ("mae", "minimize"),
    "pr_auc": ("pr_auc", "maximize"),
    "f1_retiro": ("f1_retiro", "maximize"),
    "log_loss": ("log_loss", "minimize"),
}
