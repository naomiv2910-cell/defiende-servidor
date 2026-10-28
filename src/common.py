"""Utilidades compartidas: hashes, metricas, preprocesamiento, umbral y prediccion con un paquete de modelo.

Todo el codigo que decide algo (metricas, umbral, preprocesamiento) vive aqui para que el cuaderno,
la ejecucion por linea de comandos y la prueba de recuperacion usen exactamente lo mismo.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "label"                      # 0 = normal, 1 = ataque (clase positiva = 1)
EXCLUDE = ["label", "attack_cat", "id"]  # attack_cat revela la respuesta; id solo identifica filas
POSITIVE = 1


# --------------------------------------------------------------------------- hashes y versiones
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_array(values) -> str:
    """Huella de una lista/array (p. ej. los ids de un split)."""
    arr = np.asarray(values)
    return hashlib.sha256(arr.astype(str).tobytes() if arr.dtype.kind in "OUS" else arr.tobytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return ""


def git_commit(root: str | Path = ".") -> str:
    return _git(Path(root), "rev-parse", "HEAD") or "sin-repositorio-git"


def git_dirty(root: str | Path = ".") -> bool:
    return bool(_git(Path(root), "status", "--porcelain"))


def dvc_reference(root: str | Path = ".") -> dict:
    """Lee el md5 que DVC guardo en data/raw.dvc y data/prepared.dvc (la 'version' de los datos)."""
    root = Path(root)
    ref = {}
    for name in ("raw", "prepared"):
        f = root / "data" / f"{name}.dvc"
        if f.exists():
            outs = yaml.safe_load(f.read_text()).get("outs", [])
            ref[name] = outs[0].get("md5") if outs else None
        else:
            ref[name] = None
    return ref


def data_version_string(root: str | Path = ".") -> str:
    ref = dvc_reference(root)
    if ref["raw"] or ref["prepared"]:
        return f"dvc:raw={ref['raw']};prepared={ref['prepared']}"
    return "dvc:pendiente(sin archivos .dvc)"


# --------------------------------------------------------------------------- datos
def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE]


def row_group_hash(df: pd.DataFrame) -> pd.Series:
    """Hash por fila usando SOLO las variables predictoras (sin id, label ni attack_cat).
    Dos filas con el mismo hash tienen entradas identicas."""
    return pd.Series(pd.util.hash_pandas_object(df[feature_columns(df)], index=False).values, index=df.index)


# --------------------------------------------------------------------------- preprocesamiento
def build_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    """Imputacion + escalado (opcional) + one-hot. Se ajusta SOLO con entrenamiento (via Pipeline.fit)."""
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)),
    ])
    return ColumnTransformer([("num", Pipeline(num_steps), num), ("cat", cat_pipe, cat)], remainder="drop")


# --------------------------------------------------------------------------- metricas
def _div(n: float, d: float, zero: float = 0.0) -> float:
    """Division segura. Convencion del proyecto: denominador cero -> 0.0 (y se documenta)."""
    return float(n / d) if d else zero


def confusion_counts(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def compute_metrics(y_true, y_pred) -> dict:
    """Clase positiva = ataque. Denominador cero -> 0.0."""
    c = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    f1_attack = _div(2 * tp, 2 * tp + fp + fn)
    f1_normal = _div(2 * tn, 2 * tn + fn + fp)
    return {
        **c,
        "recall": _div(tp, tp + fn),            # recall de ataque (TPR)
        "precision": _div(tp, tp + fp),         # precision de ataque
        "fpr": _div(fp, fp + tn),               # trafico normal marcado como ataque
        "f1_attack": f1_attack,
        "macro_f1": (f1_attack + f1_normal) / 2,
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
    }


def apply_threshold(score, threshold: float):
    """Regla: y_pred = 1 (ataque) si score_attack >= umbral; si no, 0."""
    return (np.asarray(score) >= threshold).astype(int)


def pick_threshold(y_true, score, max_fpr: float = 0.05) -> float:
    """Umbral que maximiza el recall de ataque manteniendo FPR <= max_fpr. Se llama SOLO con validacion."""
    fpr, tpr, thr = roc_curve(y_true, score, drop_intermediate=False)
    ok = np.where(fpr <= max_fpr)[0]
    t = float(thr[ok[-1]])
    return min(t, 1.0) if np.isfinite(t) else 1.0


# --------------------------------------------------------------------------- paquete de modelo
def predict_with_bundle(bundle_dir: str | Path, df: pd.DataFrame) -> pd.DataFrame:
    """Carga preprocesamiento+modelo (model.joblib) y umbral (config.json) y predice sobre filas crudas."""
    bundle_dir = Path(bundle_dir)
    cfg = json.loads((bundle_dir / "config.json").read_text())
    pipe = joblib.load(bundle_dir / "model.joblib")
    X = df[cfg["feature_columns"]]
    score = pipe.predict_proba(X)[:, 1]
    return pd.DataFrame({"score_attack": score, "y_pred": apply_threshold(score, cfg["threshold"])}, index=df.index)
