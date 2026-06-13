# Comparación de resultados: Python (sklearn) vs WEKA (SMO)

> **Plantilla lista para pegar resultados.** Completa las celdas de la columna *WEKA (SMO)* con los valores que obtuviste al ejecutar el experimento según [`INSTRUCCIONES_WEKA.md`](INSTRUCCIONES_WEKA.md).

---

## 1. Contexto de la comparación

- **Dataset:** `outputs/dataset_2020_preprocesado.arff` (3,152 condados, 3 features numéricas + clase nominal `winner`).
- **Algoritmo:** SVM con kernel lineal (en sklearn: `SVC(kernel='linear')`; en WEKA: `SMO` con `PolyKernel` exponent=1).
- **Valor de C:** `100.0` (mejor modelo en Python; ajustar si en WEKA se usa otro C).
- **Escalado:** `StandardScaler` en Python; filtro `Standardize` en WEKA.
- **Split:** 80/20 con semilla `42` en ambos (los algoritmos de muestreo difieren → comparación **ambiental**, no bit-a-bit).

---

## 2. Métricas de Python (referencia, ya obtenidas del notebook)

| Métrica | Python (sklearn) |
|---|---|
| Accuracy global | **0.9826** |
| Precision GOP | 0.9792 |
| Recall GOP | 1.0000 |
| F1 GOP | 0.9895 |
| Precision DEM | 1.0000 |
| Recall DEM | 0.9018 |
| F1 DEM | 0.9484 |
| N° vectores de soporte | 227 |

**Matriz de confusión (Python):**

|  | pred_GOP | pred_DEM |
|---|---|---|
| **real_GOP** | 519 | 0 |
| **real_DEM** | 11 | 101 |

---

## 3. Métricas de WEKA (pegar aquí)

> Copia del panel *Classifier output* de WEKA Explorer los valores correspondientes.

| Métrica | WEKA (SMO) |
|---|---|
| Accuracy global | _pegar aquí_ |
| Precision GOP (F-Measure columna "Precision" fila "GOP") | _pegar aquí_ |
| Recall GOP (TP Rate GOP) | _pegar aquí_ |
| F-Measure GOP | _pegar aquí_ |
| Precision DEM | _pegar aquí_ |
| Recall DEM (TP Rate DEM) | _pegar aquí_ |
| F-Measure DEM | _pegar aquí_ |
| N° vectores de soporte (sección "Support vector machining") | _pegar aquí_ |

**Matriz de confusión (WEKA):**

|  | a (=GOP) | b (=DEM) | ← classified as |
|---|---|---|---|
| **a (=GOP)** | _pegar_ | _pegar_ | a = GOP |
| **b (=DEM)** | _pegar_ | _pegar_ | b = DEM |

> WEKA usa el formato `a b <- classified as`, donde la columna a la izquierda es la clase real. La esquina superior-izquierda es `TN` para clase `a` (GOP) en este caso (519 esperado).

---

## 4. Tabla comparativa (Δ = WEKA − Python)

| Métrica | Python | WEKA | Δ |
|---|---|---|---|
| Accuracy | 0.9826 | _-_ | _-_ |
| F1 GOP | 0.9895 | _-_ | _-_ |
| F1 DEM | 0.9484 | _-_ | _-_ |
| N° SV | 227 | _-_ | _-_ |

---

## 5. Observaciones

_Pegar aquí las observaciones tras comparar:_

- ¿La diferencia de accuracy es ≤ 1%? (esperado)
- ¿La diferencia de F1-DEM es consistente con la variabilidad de muestreo?
- ¿Hay clases (GOP/DEM) en las que la diferencia sea mayor?
- ¿WEKA y Python coinciden en la cantidad de vectores de soporte? (Si difieren mucho, sospechar diferencia de implementación interna de SMO o de la condición de parada.)

---

## 6. Conclusiones

_Completar tras la comparación. Algunos puntos a discutir:_

- **Origen de las diferencias esperables:**
  1. Algoritmo de muestreo aleatorio de sklearn vs WEKA (filas ligeramente distintas en train/test).
  2. Implementación interna de SMO: la de WEKA usa SMO clásico de Platt con `PolyKernel` exponent=1; sklearn usa una versión con dual coordinate descent. La solución puede diferir ligeramente cuando hay múltiples óptimos del dual.
  3. Manejo del sesgo (`b`): sklearn usa `decision_function_shape='ovr'` por defecto en multiclase; aquí es binario así que no aplica, pero el intercept puede tener tratamientos numéricos distintos.
  4. `random_state` interno de sklearn para SVC: solo afecta el `shrinking` heuristic, no la solución, pero el orden de procesamiento sí puede variar.

- **Conclusión general:** la comparación busca confirmar que **ambos entornos producen el mismo tipo de solución** (accuracy > 0.95, alta sensibilidad para GOP, recall DEM > 0.85). Variaciones de ±0.02 en accuracy son normales y no invalidan la comparación.

- **Próximos pasos (si hubiera tiempo):**
  - Probar con C=1.0 (default) para ver si las diferencias se reducen.
  - Probar exportando los splits exactos (`train.arff` + `test.arff`) para eliminar la variabilidad de muestreo.
  - Comparar también con kernel RBF en ambos entornos.
