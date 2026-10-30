"""Revisión local de la estructura y las evidencias del proyecto."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = [
    "README.md",
    "entrega.md",
    "requirements.txt",
    "environment.yml",
    "Defiende_el_servidor.ipynb",
    "configs/experimentos.yaml",
    "src/common.py",
    "src/run_experiment.py",
    "src/bundle.py",
    "src/recuperar.py",
    "evidencias",
]
FINAL_EVIDENCE = [
    "seleccion.json",
    "metricas-test.json",
    "predicciones-test.csv",
    "mlflow",
    "recuperacion",
]
EXPERIMENT_COLUMNS = {
    "ID",
    "run_id",
    "referencia_dvc",
    "model_sha256",
    "recall",
    "precision",
    "fpr",
    "macro_f1",
}


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).exists():
            errors.append(f"Falta: {relative}")

    delivery_path = ROOT / "entrega.md"
    if delivery_path.exists():
        delivery = delivery_path.read_text(encoding="utf-8")
        if "[PENDIENTE]" in delivery or "[PENDIENTE:" in delivery:
            warnings.append("entrega.md aún contiene marcadores PENDIENTE")

    audit_path = ROOT / "evidencias" / "auditoria.json"
    if not audit_path.exists():
        warnings.append("Aún no existe evidencias/auditoria.json")
    else:
        try:
            json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"JSON inválido en evidencias/auditoria.json: {exc}")

    experiments_path = ROOT / "evidencias" / "experimentos.csv"
    if not experiments_path.exists():
        warnings.append("Aún no existe evidencias/experimentos.csv")
    else:
        try:
            with experiments_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                missing_columns = EXPERIMENT_COLUMNS.difference(reader.fieldnames or [])
                rows = list(reader)
            if missing_columns:
                errors.append(
                    "Faltan columnas en evidencias/experimentos.csv: "
                    + ", ".join(sorted(missing_columns))
                )
            if len(rows) != 4:
                errors.append(
                    f"evidencias/experimentos.csv debe tener cuatro filas E1–E4; tiene {len(rows)}"
                )
        except (OSError, csv.Error) as exc:
            errors.append(f"No se pudo leer evidencias/experimentos.csv: {exc}")

    for relative in FINAL_EVIDENCE:
        path = ROOT / "evidencias" / relative
        if relative in {"mlflow", "recuperacion"}:
            if not path.is_dir():
                warnings.append(f"Aún no existe evidencias/{relative}/")
            elif not any(candidate.is_file() for candidate in path.rglob("*")):
                warnings.append(f"evidencias/{relative}/ todavía no contiene archivos")
        elif not path.exists():
            warnings.append(f"Aún no existe evidencias/{relative}")

    if not (ROOT / ".dvc" / "config").exists():
        warnings.append("DVC no está configurado; no se puede recuperar ni publicar datos con dvc pull/push")
    for name in ("raw", "prepared"):
        if not (ROOT / "data" / f"{name}.dvc").exists():
            warnings.append(f"Aún no existe la referencia DVC data/{name}.dvc")

    for message in warnings:
        print(f"ADVERTENCIA: {message}")
    for message in errors:
        print(f"ERROR: {message}")

    if errors:
        print(f"\nResultado: {len(errors)} error(es), {len(warnings)} advertencia(s).")
        return 1
    print(f"\nResultado: estructura válida, {len(warnings)} advertencia(s) por completar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
