# Contratos de evidencia

Los resultados finales deben ser reales y coherentes con MLflow, los modelos publicados y los archivos usados en la recuperación. El estado actual incluye auditoría, comparación de experimentos y figuras; los artefactos listados como pendientes deben añadirse al completar el flujo.

## `auditoria.json`

Registra la fuente y fecha de descarga, SHA-256, filas y columnas, clases, faltantes, tipos, duplicados por variables predictoras, grupos con etiquetas contradictorias, política aplicada, variables incluidas/excluidas, semilla, tamaños y hashes de particiones, y cruces/solapamientos entre train, validación y test. El archivo actual conserva además los campos `fuentes`, `duplicados_train`, `duplicados_test`, `politica_limpieza`, `variables_usadas`, `variables_excluidas`, `particiones`, `cruces_entre_particiones` y `esquema_train_igual_test`.

## `experimentos.csv`

Una fila por configuración (`E1`–`E4`), con hipótesis, familia/modelo, parámetros, semilla, commit Git, referencia DVC, run ID, umbrales, recall, precision, FPR, macro-F1, hash del modelo, tiempo de entrenamiento y conclusión de validación. El archivo existente contiene estas columnas; no rellenes una referencia DVC que no haya sido creada.

## Evidencias finales pendientes

- `seleccion.json`: versión elegida, run ID, referencias Git/DVC/Hugging Face, hash del modelo, umbral, razón de selección y evidencia de congelación antes de consultar el test.
- `metricas-test.json`: versión y hash del modelo, hash del test, cantidad de filas, umbral, clase positiva, matriz de confusión y métricas de la versión v2 congelada.
- `predicciones-test.csv`: una fila por registro del test, con identificador de fila, etiqueta real, score de ataque y predicción; documenta la regla `score_attack >= threshold`.
- `mlflow/`: exportación estructurada de los cuatro runs, sus parámetros, métricas, metadatos y artefactos.
- `recuperacion/`: al menos 20 entradas fijas de validación, predicciones originales y recuperadas de v1/v2, hashes, comandos, logs y comparación de scores.

Evita guardar credenciales, tokens, cachés o copias descargables de los modelos en esta carpeta.
