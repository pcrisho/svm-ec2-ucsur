# Requerimientos del Proyecto: Modelo SVM para Clasificación de Resultados Electorales

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

### 3.2 Variables predictoras (features)
Usar variables numéricas del dataset original (sin datos demográficos por ahora; se evaluará agregarlos en una iteración futura):
- `votes_gop`
- `votes_dem`
- `total_votes`
- `diff`
- `per_gop`
- `per_dem`
- `per_point_diff` *(opcional: evaluar si excluirla de las features ya que define directamente la clase y podría causar separación trivial/perfecta; discutir si se usa solo para construir la clase y no como feature)*

> ⚠️ Importante: revisar si `per_point_diff`, `diff`, `per_gop` y `per_dem` generan **fuga de datos (data leakage)** respecto a la clase `winner`, ya que están matemáticamente relacionadas. Si la separación es trivial, considerar usar un subconjunto de features menos directo (ej. solo `votes_gop`, `votes_dem`, `total_votes`) para que el ejercicio de SVM sea pedagógicamente interesante.

---

## 4. Pipeline de trabajo (Python)

1. **Carga y exploración de datos**
   - Descargar el CSV desde GitHub.
   - Revisar valores nulos, tipos de datos, distribución de clases (`GOP` vs `DEM`).
   - Estadísticas descriptivas básicas.

2. **Preprocesamiento**
   - Crear la columna `winner` a partir de `per_point_diff`.
   - Seleccionar features finales (ver sección 3.2 y advertencia de leakage).
   - División en conjuntos de **entrenamiento y prueba** (ej. 80/20).
   - Escalado de variables (ej. `StandardScaler`), necesario para SVM.

3. **Entrenamiento del modelo SVM**
   - Usar `sklearn.svm.SVC`.
   - Probar **kernel lineal**.
   - Experimentar con el parámetro `C` para ilustrar:
     - **Margen duro (hard margin):** valores altos de `C` (penaliza fuertemente errores de clasificación, busca separación perfecta).
     - **Margen blando (soft margin):** valores bajos de `C` (permite errores, busca un margen más amplio/generalizable).
   - (Opcional, no obligatorio) Visualizar el hiperplano de decisión con `sklearn.svm` usando 2 features.

4. **Evaluación del modelo**
   - Métricas: *accuracy*, matriz de confusión, precisión, recall, F1-score.
   - Comparar resultados entre distintos valores de `C` (margen duro vs blando).

5. **Sistema de proyección (predicción de nuevos datos)**
   - Crear una función o script que:
     - Reciba como entrada nuevos valores para las features seleccionadas (ej. `votes_gop`, `votes_dem`, `total_votes`, etc. de un condado nuevo).
     - Aplique el mismo escalado usado en entrenamiento.
     - Devuelva la predicción de clase (`GOP` o `DEM`) usando el modelo SVM entrenado.

6. **Exportación a formato ARFF para WEKA**
   - Exportar el dataset preprocesado (features + clase `winner`) a `.arff`.
   - Asegurar que la clase `winner` esté correctamente definida como atributo **nominal** (`{GOP, DEM}`).

---

## 5. Validación cruzada con WEKA

1. Cargar el archivo `.arff` exportado en WEKA.
2. Ir a la pestaña **Classify**.
3. Seleccionar el clasificador: `weka.classifiers.functions.SMO`.
4. Configurar la clase nominal (`(Nom) winner`).
5. Ejecutar y obtener:
   - Accuracy
   - Matriz de confusión
   - Otras métricas relevantes (precision, recall, F-measure)
6. **Comparar** estos resultados con los obtenidos en Python (sklearn), documentando cualquier diferencia y posibles causas (ej. diferencias en kernel, parámetros por defecto, escalado de datos, manejo de `C`/`gamma`, etc.).

---

## 6. Entregables esperados

1. Script(s) o notebook en Python que cubran: carga de datos, preprocesamiento, entrenamiento SVM, evaluación, y sistema de proyección de nuevos datos.
2. Archivo `.arff` exportado para WEKA.
3. Documento/resumen comparando resultados Python vs WEKA (tabla de métricas + observaciones sobre diferencias).
4. (Opcional) Visualización del hiperplano SVM con `sklearn.svm` para 2 features seleccionadas.

---

## 7. Conceptos clave a demostrar/explicar

- **Margen duro vs margen blando** en SVM, y cómo el parámetro `C` controla este comportamiento.
- Diferencias prácticas entre implementación en **Python (sklearn)** y **WEKA (SMO)** para el mismo dataset y clase nominal.
- Interpretación de las métricas de clasificación (accuracy, matriz de confusión, etc.).

---

## 8. Próximas iteraciones (fuera de alcance inicial)

- Incorporar variables demográficas por condado (población, ingreso, % urbano/rural) si se encuentra una fuente accesible, para enriquecer el modelo.
- Evaluar otros kernels de SVM (RBF, polinomial) si el resultado lineal es insuficiente.