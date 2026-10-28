"""Construccion del paquete de modelo (lo que se sube a Hugging Face) y su model card."""
from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.common import EXCLUDE, apply_threshold, sha256_file

CARD = """---
license: other
library_name: scikit-learn
tags: [tabular-classification, intrusion-detection, unsw-nb15, mlops, educational]
metrics: [recall, precision, f1]
---

# Defiende el servidor - {version} ({name})

Clasificador binario **normal (0) / ataque (1)** sobre UNSW-NB15 (particiones publicadas training-set / testing-set).
Proyecto academico de MLOps (Universidad Externado de Colombia). **No** demuestra eficacia en redes actuales.

## Que cambia en esta version
{change_text}

## Trazabilidad (codigo -> datos -> experimento -> modelo)
| Elemento | Valor |
|---|---|
| Commit de codigo (GitHub) | `{git_commit}` |
| Version de datos (DVC) | `{data_version}` |
| Run de MLflow | `{run_id}` (experimento {exp_id}) |
| SHA-256 de `model.joblib` | `{model_sha256}` |
| Umbral | `{threshold:.6f}` (regla: `y_pred = 1` si `score_attack >= umbral`) |
| Semilla | {seed} |

## Datos
Entrenamiento y validacion salen **solo** de `UNSW_NB15_training-set.csv`, con division por grupos de entradas identicas
(sin fuga entre ambos). Entradas excluidas: `label`, `attack_cat`, `id`. Preprocesamiento (imputacion, escalado, one-hot)
ajustado unicamente con entrenamiento y guardado dentro del mismo `Pipeline`.

## Metricas de validacion (clase positiva = ataque)
{metrics_table}

Matriz de confusion (validacion): TN={tn}, FP={fp}, FN={fn}, TP={tp}.
Denominador cero -> 0.0.

## Uso
```python
import json, joblib, pandas as pd
from huggingface_hub import snapshot_download
path = snapshot_download("{repo_id}", revision="<SHA_COMPLETO_DE_ESTA_REVISION>")
cfg = json.load(open(f"{{path}}/config.json"))
pipe = joblib.load(f"{{path}}/model.joblib")
df = pd.read_csv("entradas.csv")                       # columnas crudas del dataset
score = pipe.predict_proba(df[cfg["feature_columns"]])[:, 1]
y_pred = (score >= cfg["threshold"]).astype(int)       # 1 = ataque
```
Instala las dependencias de `requirements.txt` (mismas versiones de scikit-learn). `joblib` usa pickle: carga solo modelos de fuentes confiables.

## Limitaciones
- Dataset historico (2015), trafico sintetico de laboratorio; puede no generalizar a redes actuales.
- El conjunto de test publicado se solapa con entrenamiento/validacion (ver `auditoria.json` del repositorio de codigo).
- Se priorizo {priority}; revisa FPR y recall por tipo de ataque antes de usarlo en cualquier decision real.
- Modelo educativo, no apto para produccion.
"""


def metrics_table(m: dict) -> str:
    rows = [("Recall (ataque)", m["recall"]), ("Precision (ataque)", m["precision"]), ("FPR", m["fpr"]),
            ("Macro-F1", m["macro_f1"]), ("Accuracy (complemento)", m["accuracy"])]
    return "| Metrica | Valor |\n|---|---|\n" + "\n".join(f"| {k} | {v:.4f} |" for k, v in rows)


def build_bundle(out_dir: Path, version: str, run_info: dict, cfg: dict, threshold: float, val_metrics: dict,
                 feats: list[str], repo_id: str, seed: int, change_text: str, priority: str, model_src: Path) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copy(model_src, out_dir / "model.joblib")
    model_hash = sha256_file(out_dir / "model.joblib")
    config = {
        "version": version, "exp_id": cfg["id"], "model": cfg["model"], "threshold": float(threshold),
        "positive_class": 1, "rule": "y_pred = 1 if score_attack >= threshold else 0",
        "feature_columns": feats, "excluded_columns": EXCLUDE, "seed": seed,
        "model_sha256": model_hash, "run_id": run_info["run_id"], "git_commit": run_info["git_commit"],
        "data_version": run_info["data_version"], "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(), "validation_metrics": val_metrics,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "requirements.txt").write_text(
        f"numpy=={np.__version__}\npandas=={pd.__version__}\nscikit-learn=={sklearn.__version__}\n"
        f"joblib=={joblib.__version__}\nhuggingface_hub\n")
    card = CARD.format(version=version, name=cfg["name"], change_text=change_text, git_commit=run_info["git_commit"],
                       data_version=run_info["data_version"], run_id=run_info["run_id"], exp_id=cfg["id"],
                       model_sha256=model_hash, threshold=threshold, seed=seed, repo_id=repo_id,
                       metrics_table=metrics_table(val_metrics), tn=val_metrics["tn"], fp=val_metrics["fp"],
                       fn=val_metrics["fn"], tp=val_metrics["tp"], priority=priority)
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    return config


def publish_version(api, repo_id: str, folder: Path, version: str, message: str) -> str:
    """Sube la carpeta como UNA revision (borrando lo anterior para que cada revision sea un modelo completo)
    y crea el tag. Devuelve el SHA completo (inmutable) del commit."""
    commit = api.upload_folder(repo_id=repo_id, folder_path=str(folder), repo_type="model",
                               commit_message=message, delete_patterns="*")
    sha = commit.oid
    api.create_tag(repo_id=repo_id, tag=version, revision=sha, repo_type="model", exist_ok=True)
    info = api.model_info(repo_id, revision=sha)
    assert info.sha == sha, "El SHA publicado no coincide con el de la revision consultada"
    return sha
