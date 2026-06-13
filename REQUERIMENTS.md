# Requerimientos del Proyecto: Modelo SVM para Clasificación de Resultados Electorales

> **Estado:** requerimientos validados con el usuario. Decisiones de implementación detalladas en [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md). Este archivo describe **qué** se quiere construir; el `IMPLEMENTATION-PLAN.md` describe **cómo**.

## 1. Objetivo general

Construir un sistema en Python que:
1. Entrene un modelo de **Support Vector Machine (SVM)** usando `sklearn.svm` con un dataset de resultados electorales.
2. Permita ingresar **nuevos datos** (de un condado hipotético) y que el modelo entrenado **clasifique** el resultado (GOP o DEM).
3. Exporte el dataset preprocesado a formato `.arff` para validar el mismo modelo en **WEKA** (función SMO, clase nominal) y comparar resultados entre WEKA y Python.

> Nota: La regresión lineal **NO** forma parte de este proyecto. El enfoque es exclusivamente SVM.

---

## 2. Dataset

- **Fuente:** [tonmcg/US_County_Level_Election_Results_08-24](https://github.com/tonmcg/US_County_Level_Election_Results_08-24) (GitHub).
- **Archivo a usar:** `2020_US_County_Level_Presidential_Results.csv`
- **Tamaño:** ~3,100 registros (condados de EE.UU.), cumple ampliamente el mínimo de 500 registros requerido para SVM.
- **Columnas originales:**
  - `state_name`
  - `county_fips`
  - `county_name`
  - `votes_gop`
  - `votes_dem`
  - `total_votes`
  - `diff`
  - `per_gop`
  - `per_dem`
  - `per_point_diff`

---

## 3. Definición de variables para el modelo

### 3.1 Variable objetivo (clase nominal)
- **Nombre:** `winner`
- **Definición:**
  - `GOP` si `per_point_diff > 0`
  - `DEM` si `per_point_diff <= 0`
- Es una clasificación **binaria**, ideal para SVM con clase nominal en WEKA (SMO).

### 3.2 Variables predictoras (features) — **DECIDIDO**

Se utilizarán **únicamente las 3 variables crudas** que no están matemáticamente derivadas de la clase:

- `votes_gop`
- `votes_dem`
- `total_votes`

> ⚠️ **Decisión sobre fuga de datos (data leakage):** las columnas `diff`, `per_gop`, `per_dem` y `per_point_diff` se excluyen de las features porque son derivadas directas de `votes_gop`, `votes_dem` y `total_votes`, e incluyen el `per_point_diff` que define la clase. Usarlas produciría una separación trivial (accuracy ≈ 100%) que no aporta valor pedagógico al ejercicio SVM. En una iteración futura podrían reconsiderarse junto con variables demográficas externas (ver §8).

---

## 4. Pipeline de trabajo (Python)

### 4.1 Carga y exploración de datos
- Descargar el CSV desde GitHub (con cache local en `data/`).
- Revisar valores nulos, tipos de datos, distribución de clases (`GOP` vs `DEM`).
- Estadísticas descriptivas básicas de las 3 features.

### 4.2 Preprocesamiento
- Crear la columna `winner` a partir de `per_point_diff`.
- Seleccionar las 3 features finales según §3.2.
- División en conjuntos de **entrenamiento y prueba**: **80/20 estratificado** con `random_state=42` (reproducibilidad).
- Escalado de variables con `StandardScaler` (fit solo en train, transform en test).

### 4.3 Entrenamiento del modelo SVM
- Usar `sklearn.svm.SVC` con `kernel='linear'`.
- Evaluar **5 valores de C** para ilustrar margen duro vs blando: `C ∈ {0.01, 0.1, 1.0, 10, 100}`.
- Documentar el efecto de C en accuracy y número de vectores de soporte.

### 4.4 Evaluación del modelo
- Métricas: *accuracy*, matriz de confusión, precisión, recall, F1-score **por clase** (no solo accuracy global, por el desbalance ~75/25 GOP/DEM).
- Comparar resultados entre los 5 valores de `C`.
- Identificar el `C` con mejor balance (sugerencia: mejor F1 de la clase DEM).

### 4.5 Sistema de proyección (predicción de nuevos datos)
- Función `predict_county(votes_gop, votes_dem, total_votes, model, scaler) -> str` que:
  - Recibe valores crudos.
  - Aplica el `StandardScaler` entrenado.
  - Devuelve `'GOP'` o `'DEM'`.
- Demostración con 3 condados inventados (GOP claro, DEM claro, reñido).

### 4.6 Exportación a formato ARFF para WEKA
- Exportar el dataset **sin escalar** (3 features crudas + clase `winner`) a `outputs/dataset_2020_preprocesado.arff`.
- Estructura mínima:
  - `@ATTRIBUTE votes_gop   NUMERIC`
  - `@ATTRIBUTE votes_dem   NUMERIC`
  - `@ATTRIBUTE total_votes NUMERIC`
  - `@ATTRIBUTE winner      {GOP, DEM}`
- WEKA aplicará el filtro `Standardize` internamente para mantener paridad de escala con Python.

---

## 5. Validación cruzada con WEKA

1. Cargar `outputs/dataset_2020_preprocesado.arff` en WEKA.
2. Aplicar filtro `weka.filters.unsupervised.attribute.Standardize` a las 3 features numéricas (NO a `winner`).
3. Ir a la pestaña **Classify**.
4. Seleccionar el clasificador: `weka.classifiers.functions.SMO` con kernel polinómico de grado 1 (equivalente a lineal).
5. Configurar la clase nominal: `(Nom) winner`.
6. **Test options:** *Percentage split* = 80%, *Random seed* = 42.
7. *Start* y obtener accuracy, matriz de confusión, precision, recall, F-measure.
8. **Comparación "ambiental":** estos resultados se contrastan con los de Python. Los splits NO son idénticos al 100% (los algoritmos de muestreo aleatorio de sklearn y WEKA difieren), por lo que variaciones de ~0.5-1% en accuracy son esperables. La comparación es válida como "mismo dataset + mismo algoritmo + mismo C + misma escala" (ver [`outputs/comparacion_python_vs_weka.md`](outputs/comparacion_python_vs_weka.md)).

---

## 6. Entregables esperados

1. **`notebook.ipynb`** — carga, preprocesamiento, entrenamiento SVM, evaluación y sistema de predicción. Ejecutable de inicio a fin.
2. **`outputs/dataset_2020_preprocesado.arff`** — insumo para WEKA.
3. **`outputs/INSTRUCCIONES_WEKA.md`** — pasos exactos para replicar el experimento en WEKA.
4. **`outputs/comparacion_python_vs_weka.md`** — tabla comparativa (plantilla) + observaciones.
5. **`requirements.txt`** + **`README.md`** — instrucciones de entorno (venv) y uso del repositorio.
6. **(Opcional)** Visualización del hiperplano SVM con 2 features (`votes_gop` vs `votes_dem`).

---

## 7. Conceptos clave a demostrar/explicar

- **Margen duro vs margen blando** en SVM, y cómo el parámetro `C` controla este comportamiento.
- **Data leakage** cuando las features están matemáticamente relacionadas con la clase.
- Diferencias prácticas entre implementación en **Python (sklearn)** y **WEKA (SMO)** para el mismo dataset y clase nominal.
- Interpretación de las métricas de clasificación (accuracy, matriz de confusión, precision/recall/F1 por clase).

---

## 8. Próximas iteraciones (fuera de alcance inicial)

- Incorporar variables demográficas por condado (población, ingreso, % urbano/rural) si se encuentra una fuente accesible, para enriquecer el modelo.
- Evaluar otros kernels de SVM (RBF, polinomial) si el resultado lineal es insuficiente.
- Re-evaluar la inclusión de `per_gop`/`per_dem` como features una vez que se tengan más variables externas (con menos riesgo de leakage).
