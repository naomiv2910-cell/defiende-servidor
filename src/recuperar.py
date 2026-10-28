"""Prueba de recuperacion: descarga v1 y v2 desde Hugging Face POR SHA, predice sobre entradas fijas
y compara con las salidas guardadas al entrenar.

Uso (en un entorno y carpeta NUEVOS, con la cache vacia):
    python -m src.recuperar --repo-id USUARIO/defiende-el-servidor-unsw \
        --v1-sha <SHA_COMPLETO_V1> --v2-sha <SHA_COMPLETO_V2>

Modo local (solo para depurar, no cuenta como prueba): --local-v1 hf_v1 --local-v2 hf_v2
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.common import predict_with_bundle, sha256_file


def fetch(repo_id: str | None, sha: str | None, local: str | None, cache_dir: str) -> Path:
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download
    if not sha or len(sha) != 40:
        raise SystemExit("Debes indicar el SHA COMPLETO (40 caracteres) de la revision.")
    return Path(snapshot_download(repo_id=repo_id, revision=sha, cache_dir=cache_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id")
    ap.add_argument("--v1-sha")
    ap.add_argument("--v2-sha")
    ap.add_argument("--local-v1")
    ap.add_argument("--local-v2")
    ap.add_argument("--dir", default="evidencias/recuperacion")
    ap.add_argument("--tol", type=float, default=1e-9, help="tolerancia numerica para los scores")
    a = ap.parse_args()

    d = Path(a.dir)
    entradas = pd.read_csv(d / "entradas_validacion.csv")
    assert len(entradas) >= 20, "Se requieren al menos 20 filas fijas de validacion"
    cache = tempfile.mkdtemp(prefix="hf_cache_nueva_")  # cache vacia: nada se reutiliza de la ejecucion original

    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    es_local = bool(a.local_v1 or a.local_v2)
    sufijo = "_local" if es_local else ""   # el ensayo local NO cuenta como prueba de recuperacion
    log(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | cache HF: {cache}")
    resultado = {"python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__,
                 "n_entradas": len(entradas), "tolerancia": a.tol, "hf_cache_dir": cache, "versiones": {}}
    ok_global = True
    for v, sha, local in (("v1", a.v1_sha, a.local_v1), ("v2", a.v2_sha, a.local_v2)):
        bundle = fetch(a.repo_id, sha, local, cache)
        cfg = json.loads((bundle / "config.json").read_text())
        model_hash = sha256_file(bundle / "model.joblib")
        pred = predict_with_bundle(bundle, entradas)
        out = pd.DataFrame({"id": entradas["id"], **pred.to_dict("list")})
        out.to_csv(d / f"salidas_recuperadas_{v}{sufijo}.csv", index=False)
        esperado = pd.read_csv(d / f"salidas_originales_{v}.csv")
        max_diff = float(np.max(np.abs(out["score_attack"].values - esperado["score_attack"].values)))
        pred_ok = bool((out["y_pred"].values == esperado["y_pred"].values).all())
        hash_ok = model_hash == cfg["model_sha256"]
        ok = pred_ok and max_diff <= a.tol and hash_ok
        ok_global &= ok
        resultado["versiones"][v] = {"revision": sha or f"local:{local}", "model_sha256": model_hash,
                                     "hash_coincide_con_config": hash_ok, "umbral": cfg["threshold"],
                                     "predicciones_identicas": pred_ok, "max_dif_score": max_diff, "ok": ok}
        log(f"{v}: revision={sha or local} hash_ok={hash_ok} predicciones_identicas={pred_ok} max_dif_score={max_diff:.3e} -> {'OK' if ok else 'FALLA'}")
    resultado["resultado_global"] = "OK" if ok_global else "FALLA"
    (d / f"resultado_recuperacion{sufijo}.json").write_text(json.dumps(resultado, indent=2))
    log("RESULTADO GLOBAL: " + resultado["resultado_global"])
    (d / f"log_recuperacion{sufijo}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0 if ok_global else 1)


if __name__ == "__main__":
    main()
