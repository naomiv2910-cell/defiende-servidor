# Defiende el servidor — Versionado de código, datos, experimentos y modelos
Clasificación binaria normal/ataque (UNSW-NB15) con Git/GitHub, DVC, MLflow y Hugging Face Hub. Curso MLOps 2026-II.

**Python 3.11 o superior** (probado con 3.12). Dependencias fijadas en `requirements.txt`.

## 1. Instalación

### Con Conda (recomendado)
```bash
conda env create -f environment.yml
conda activate defiende-servidor
```

Para registrar el ambiente como kernel de Jupyter/VS Code:
```bash
python -m ipykernel install --user --name defiende-servidor --display-name "Python (defiende-servidor)"
```

Si actualizas las dependencias, sincroniza `environment.yml` con `requirements.txt`. Para recrear el ambiente después de esos cambios:
```bash
conda env update -f environment.yml --prune
```

### Alternativa con `venv`
```bash
git clone https://github.com/<usuario>/<repo>.git && cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Recuperar los datos (DVC)
```bash
dvc pull                           # descarga data/raw (originales) y data/prepared (preparados + splits) desde el remote DVC
```
Remote DVC: `<COMPLETAR: ubicación y forma de acceso>` (ver `entrega.md`). Verifica huellas: `sha256sum data/raw/*.csv` (Windows: `Get-FileHash data\raw\*.csv`) y compáralas con `evidencias/auditoria.json`.
Si empiezas sin DVC: copia `UNSW_NB15_training-set.csv` y `UNSW_NB15_testing-set.csv` en `data/raw/` y ejecuta las secciones 2–4 del cuaderno.

## 3. Ejecutar un experimento
```bash
python -m src.run_experiment --id E1      # E1..E4, definidos en configs/experimentos.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db     # opcional: ver los runs en http://127.0.0.1:5000
```
Todo el flujo (auditoría, splits, 4 experimentos, v1/v2, test) está en `Defiende_el_servidor.ipynb`; ejecútalo en orden desde la raíz del proyecto.

## 4. Descargar un modelo y predecir
```python
import json, joblib, pandas as pd
from huggingface_hub import snapshot_download
path = snapshot_download("<usuario>/<repo-hf>", revision="<SHA_COMPLETO_V1_o_V2>")   # SHA de 40 caracteres
cfg  = json.load(open(f"{path}/config.json")); pipe = joblib.load(f"{path}/model.joblib")
df   = pd.read_csv("evidencias/recuperacion/entradas_validacion.csv")
score = pipe.predict_proba(df[cfg["feature_columns"]])[:, 1]
y_pred = (score >= cfg["threshold"]).astype(int)     # 1 = ataque
```

## 5. Prueba de recuperación (entorno nuevo)
```bash
python -m src.recuperar --repo-id <usuario>/<repo-hf> --v1-sha <SHA_V1> --v2-sha <SHA_V2>
```
Escribe `evidencias/recuperacion/log_recuperacion.txt`, `resultado_recuperacion.json` y las salidas recuperadas de v1 y v2.

## Estructura
```
src/ (common.py, run_experiment.py, bundle.py, recuperar.py)   configs/experimentos.yaml
data/raw.dvc  data/prepared.dvc  .dvc/  evidencias/  Defiende_el_servidor.ipynb  entrega.md
```
Cadena de trazabilidad: commit Git → referencia DVC → run ID MLflow → SHA de Hugging Face (ver `entrega.md`).
