"""Ejecuta UNA configuracion de configs/experimentos.yaml y la registra en MLflow.

Uso desde la raiz del proyecto:
    python -m src.run_experiment --id E1
    python -m src.run_experiment --id E4 --max-fpr 0.05

Requiere data/prepared/train.csv y data/prepared/val.csv (los genera el cuaderno, seccion 3).
Nunca toca el archivo de test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.common import (EXCLUDE, SEED, TARGET, apply_threshold, build_preprocessor, compute_metrics,
                        data_version_string, feature_columns, git_commit, pick_threshold, sha256_file)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = {"LogisticRegression": LogisticRegression,
          "DecisionTreeClassifier": DecisionTreeClassifier,
          "RandomForestClassifier": RandomForestClassifier}
EXPERIMENT_NAME = "defiende-el-servidor"


def setup_mlflow(root: Path) -> None:
    """Seguimiento local (SQLite + carpeta de artefactos). No necesita servidor."""
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(root / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def load_prepared(root: Path):
    train = pd.read_csv(root / "data/prepared/train.csv")
    val = pd.read_csv(root / "data/prepared/val.csv")
    return train, val


def build_pipeline(cfg: dict, X_train: pd.DataFrame, seed: int) -> Pipeline:
    params = dict(cfg["params"])
    cls = MODELS[cfg["model"]]
    if cfg["model"] != "LogisticRegression":
        params["random_state"] = seed
    else:
        params["random_state"] = seed
    return Pipeline([("prep", build_preprocessor(X_train, scale=cfg["scale"])), ("clf", cls(**params))])


def run_experiment(cfg: dict, root: str | Path = ".", seed: int = SEED, max_fpr: float = 0.05) -> dict:
    root = Path(root)
    setup_mlflow(root)
    train, val = load_prepared(root)
    feats = feature_columns(train)
    X_tr, y_tr, X_va, y_va = train[feats], train[TARGET], val[feats], val[TARGET]

    commit, dver = git_commit(root), data_version_string(root)
    out_dir = root / "models" / cfg["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"{cfg['id']}_{cfg['name']}") as run:
        mlflow.set_tags({"exp_id": cfg["id"], "family": cfg["family"], "hypothesis": " ".join(cfg["hypothesis"].split()),
                         "change": cfg["change"], "git_commit": commit, "data_version": dver})
        mlflow.log_params({"model": cfg["model"], "seed": seed, "scale": cfg["scale"],
                           "n_train": len(train), "n_val": len(val), "n_features": len(feats),
                           "max_fpr_for_threshold": max_fpr, **{f"p_{k}": v for k, v in cfg["params"].items()}})

        pipe = build_pipeline(cfg, X_tr, seed)
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        fit_s = time.time() - t0
        score = pipe.predict_proba(X_va)[:, 1]

        # Metricas con el umbral por defecto (0.5) y con el umbral elegido en validacion
        m05 = compute_metrics(y_va, apply_threshold(score, 0.5))
        thr = pick_threshold(y_va, score, max_fpr=max_fpr)
        mtuned = compute_metrics(y_va, apply_threshold(score, thr))
        mlflow.log_metric("threshold", 0.5)
        for k, v in m05.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("tuned_threshold", thr)
        for k, v in mtuned.items():
            mlflow.log_metric(f"tuned_{k}", v)
        mlflow.log_metric("fit_seconds", fit_s)

        # Artefactos: modelo completo (preprocesamiento + estimador), scores de validacion y graficas
        model_path = out_dir / "model.joblib"
        joblib.dump(pipe, model_path, compress=3)
        model_hash = sha256_file(model_path)
        pd.DataFrame({"id": val["id"], "y_true": y_va, "score_attack": score}).to_csv(out_dir / "val_scores.csv", index=False)

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        ConfusionMatrixDisplay.from_predictions(y_va, apply_threshold(score, 0.5), display_labels=["normal", "ataque"],
                                                ax=ax[0], colorbar=False)
        ax[0].set_title(f"{cfg['id']} umbral 0.5")
        ConfusionMatrixDisplay.from_predictions(y_va, apply_threshold(score, thr), display_labels=["normal", "ataque"],
                                                ax=ax[1], colorbar=False)
        ax[1].set_title(f"{cfg['id']} umbral {thr:.3f} (FPR<={max_fpr})")
        RocCurveDisplay.from_predictions(y_va, score, ax=ax[2])
        ax[2].set_title("ROC (validacion)")
        fig.tight_layout()
        fig.savefig(out_dir / "validacion.png", dpi=110)
        plt.close(fig)

        info = {"run_id": run.info.run_id, "exp_id": cfg["id"], "model_sha256": model_hash, "git_commit": commit,
                "data_version": dver, "tuned_threshold": thr, "metrics_threshold_0.5": m05, "metrics_tuned": mtuned}
        (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
        for f in ("model.joblib", "val_scores.csv", "validacion.png", "run_info.json"):
            mlflow.log_artifact(str(out_dir / f), artifact_path="modelo" if f == "model.joblib" else "evidencia")
        mlflow.log_dict(cfg, "evidencia/config_experimento.json")

    print(f"[{cfg['id']}] run_id={info['run_id']} recall={m05['recall']:.4f} fpr={m05['fpr']:.4f} macroF1={m05['macro_f1']:.4f}"
          f" | umbral_val={thr:.4f} recall={mtuned['recall']:.4f} fpr={mtuned['fpr']:.4f}")
    return {**info, "family": cfg["family"], "hypothesis": cfg["hypothesis"], "name": cfg["name"], "model": cfg["model"],
            "params": cfg["params"], "change": cfg["change"], "seed": seed, "fit_seconds": fit_s}


def load_configs(root: str | Path = ".") -> dict:
    return yaml.safe_load((Path(root) / "configs/experimentos.yaml").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="E1, E2, E3 o E4")
    ap.add_argument("--root", default=".")
    ap.add_argument("--max-fpr", type=float, default=0.05)
    a = ap.parse_args()
    conf = load_configs(a.root)
    cfg = next(e for e in conf["experiments"] if e["id"] == a.id)
    run_experiment(cfg, a.root, seed=conf.get("seed", SEED), max_fpr=a.max_fpr)
