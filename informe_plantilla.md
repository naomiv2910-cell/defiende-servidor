# Informe — Defiende el servidor (máx. 5 páginas, referencias aparte)
Integrantes y grupo: ...

## 1. Datos y limpieza
Fuentes, SHA-256, filas/columnas (tabla de auditoria.json). Duplicados por entradas, contradicciones y política. Variables excluidas (label, attack_cat, id) y por qué.
Split por grupos (semilla, tamaños, huellas). Solapamiento del test con train/val. *Figuras: 01_clases, 04_particiones.*

## 2. Experimentos (E1–E4)
Tabla de experimentos.csv (hipótesis, cambio, parámetros, métricas con umbral 0.5 y con umbral de validación). *Figuras: 05_comparacion_modelos, 06_roc_pr.* Interpretación de errores (10_error_por_categoria).

## 3. Versiones v1 y v2
Qué cambia, qué mejora, qué empeora, qué errores se aceptan. Tabla v1 vs v2 (validación). *Figura: 09_v1_vs_v2.*
Trazabilidad: commit Git, referencia DVC, run MLflow, SHA Hugging Face (tabla de entrega.md).

## 4. Elección del umbral
Criterio (recall con FPR ≤ 5 % en validación), curva recall/FPR vs umbral, evidencia de congelación (tag `seleccion-congelada`).

## 5. Test final y recuperación
Métricas y matriz de confusión en test (una sola evaluación). Prueba de recuperación en entorno nuevo (resultado_recuperacion.json).

## 6. Limitaciones
Dataset histórico; solapamiento test/train; contradicciones; sin garantía en redes actuales; declaración de uso de IA y contribución de cada integrante.
