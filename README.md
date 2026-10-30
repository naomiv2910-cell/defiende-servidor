# Defiende el servidor

Clasificación binaria normal (`0`) / ataque (`1`) sobre UNSW-NB15. El proyecto registra experimentos con MLflow y prepara modelos para Hugging Face Hub.

Repositorio: <https://github.com/naomiv2910-cell/defiende-servidor>

## Requisitos e instalación

El entorno Conda del proyecto usa Python 3.12. Para reproducirlo:

```powershell
conda env create -f environment.yml
conda activate defiende-servidor
python -m ipykernel install --user --name defiende-servidor --display-name "Python (defiende-servidor)"
```

Alternativa con `venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si cambias las dependencias, mantén sincronizados `environment.yml` y `requirements.txt`.

## Datos y DVC

En la copia de trabajo actual, los archivos originales están en `data/` y los datos preparados y particiones en `data/prepared/`. Aún no hay una configuración de remote DVC ni archivos `.dvc`; por eso `dvc pull` no está listo y los metadatos de los experimentos muestran que falta registrar la versión DVC.

Antes de publicar una entrega reproducible, mueve/versiona los datos originales y preparados con DVC, configura un almacenamiento accesible para el equipo y actualiza `entrega.md`. No incluyas los CSV grandes directamente en GitHub. Comprueba los SHA-256 con:

```powershell
Get-FileHash data\UNSW_NB15_training-set.csv -Algorithm SHA256
Get-FileHash data\UNSW_NB15_testing-set.csv -Algorithm SHA256
```

## Ejecutar experimentos

El cuaderno principal está en la raíz y contiene el flujo de auditoría, preparación, comparación de modelos y evaluación:

```text
Defiende_el_servidor.ipynb
```

Ejecuta sus celdas en orden desde la raíz del proyecto. Para ejecutar una configuración ya preparada y registrarla en MLflow:

```powershell
python -m src.run_experiment --id E1
python -m src.run_experiment --id E4 --max-fpr 0.05
```

Los identificadores `E1`–`E4` y sus parámetros están en `configs/experimentos.yaml`. Este comando requiere `data/prepared/train.csv` y `data/prepared/val.csv`; no lee ni modifica el conjunto de test.

Para consultar los runs locales:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Publicar y recuperar modelos

`src/bundle.py` contiene la creación del paquete del modelo y su publicación en Hugging Face Hub; usa una sesión autenticada y un repositorio de modelos que pertenezca al equipo. No guardes tokens en el repositorio.

La prueba de recuperación de `src.recuperar` requiere un directorio con `entradas_validacion.csv` y las salidas originales de v1/v2. Cuando esos archivos existan:

```powershell
python -m src.recuperar --repo-id USUARIO/REPOSITORIO-HF --v1-sha SHA_COMPLETO_V1 --v2-sha SHA_COMPLETO_V2
```

Usa SHA completos de 40 caracteres. La prueba descarga ambas revisiones en una caché temporal, comprueba los hashes y compara predicciones y scores. La prueba local con `--local-v1` y `--local-v2` sirve para depurar, pero no sustituye la recuperación desde Hugging Face.

## Evidencias y entrega

`evidencias/auditoria.json`, `evidencias/experimentos.csv` y las figuras actuales documentan el trabajo disponible. La selección final, las métricas de test y los archivos para la prueba de recuperación deben completarse antes de considerar cerrada la entrega. Los contratos de esos archivos están descritos en [evidencias/README.md](evidencias/README.md), y [entrega.md](entrega.md) funciona como índice.

Ejecuta la revisión estructural antes de entregar:

```powershell
python scripts/check_delivery.py
```

La revisión informa como advertencias sobre datos/evidencias que aún falten; no inventa resultados ni configura servicios externos.

## Estructura principal

```text
configs/experimentos.yaml        Configuraciones E1–E4
data/                            Datos originales y preparados de la copia local
evidencias/                      Auditoría, resultados y figuras
src/common.py                    Métricas, preprocesamiento y utilidades
src/run_experiment.py            Entrenamiento y registro en MLflow
src/bundle.py                    Paquetes y publicación de modelos
src/recuperar.py                 Prueba de recuperación de v1/v2
Defiende_el_servidor.ipynb       Flujo de trabajo principal
```

La trazabilidad esperada es `commit Git → referencia DVC → run ID MLflow → SHA de Hugging Face`. La referencia DVC seguirá pendiente hasta que el equipo configure DVC y versione los datos.
