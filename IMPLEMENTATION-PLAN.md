# Plan de Implementación — SVM EC2 UCSUR

> **Archivo maestro del proyecto.** Mientras `REQUERIMENTS.md` declara **qué** se quiere, este documento define **cómo** se construye: arquitectura, decisiones técnicas, dependencias, estructura de carpetas y el detalle paso a paso de cada fase.
>
> Toda duda o ambigüedad que aparezca durante la implementación se resuelve contra este archivo. Si una decisión nueva contradice lo aquí escrito, este archivo se actualiza primero y luego el código.

---

## 0. Resumen de decisiones técnicas (cerradas)

| Aspecto | Decisión | Justificación |
|---|---|---|
| Variable objetivo | `winner ∈ {GOP, DEM}` desde `per_point_diff` | Definida en REQUERIMENTS §3.1 |
| Features (predictoras) | **`votes_gop`, `votes_dem`, `total_votes`** (solo estas 3) | Evita data leakage; las demás son derivadas matemáticas y producen separación trivial. Decidido en validación. |
| Algoritmo | `sklearn.svm.SVC` con `kernel='linear'` | Comparable 1:1 con `weka.classifiers.functions.SMO` |
| Valores de `C` a evaluar | `[0.01, 0.1, 1.0, 10, 100]` | Demuestra transición margen blando → duro |
| Partición | `train_test_split(test_size=0.20, random_state=42, stratify=y)` | Reproducibilidad + balance de clases |
| Escalado | `StandardScaler` (fit solo en `X_train`, transform en `X_test`) | SVM es sensible a la escala |
| Formato del código | Un único `notebook.ipynb` ejecutado de inicio a fin | Entrega académica, revisión celda a celda |
| Exportación ARFF | **Sin escalar**, 3 atributos numéricos + 1 nominal `winner` | WEKA aplica `Standardize` en su flujo |
| Comparación Python vs WEKA | **Ambiental**: mismo dataset, mismo kernel, mismo `C`, misma escala. Los splits difieren ~0-1% (algoritmos de muestreo distintos) | Decidido en validación; documentado en `comparacion_python_vs_weka.md` |
| Entorno | `python3 -m venv .venv` + `pip install -r requirements.txt` | Aislamiento, reproducibilidad |
| Versión Python | ≥ 3.10 | Requerida por `scikit-learn` 1.5+ |

---

## 1. Estructura de archivos (objetivo final)

```
svm-ec2-ucsur/
├── .venv/                                # entorno virtual (ignorado por git)
├── data/
│   └── 2020_US_County_Level_Presidential_Results.csv   # cacheado por el notebook
├── outputs/
│   ├── dataset_2020_preprocesado.arff    # entregable: insumo para WEKA
│   ├── INSTRUCCIONES_WEKA.md             # guía paso a paso para WEKA
│   └── comparacion_python_vs_weka.md     # plantilla para pegar resultados
├── IMPLEMENTATION-PLAN.md                # este archivo (masterfile)
├── REQUERIMENTS.md                       # requerimientos originales (actualizado)
├── README.md                             # guía de uso del repo + venv
├── requirements.txt                      # dependencias pinned
└── notebook.ipynb                        # entregable principal (código)
```

---

## 2. Dependencias (`requirements.txt`)

```
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
matplotlib>=3.8
jupyter>=1.0
```

Notas:
- No se añade `liac-arff` ni `scipy.io.arff`: el `.arff` se escribe a mano (cabecera `@RELATION`, `@ATTRIBUTE ... NUMERIC`, `@ATTRIBUTE winner {GOP,DEM}`, `@DATA`) — son ~20 líneas y evita una dependencia.
- `matplotlib` solo se usa para la visualización opcional de la Fase 9.
- Las versiones son mínimas; pip resolverá a las últimas compatibles.

---

## 3. Setup del entorno (venv)

Documentado paso a paso en `README.md`. Resumen:

```bash
python3 -m venv .venv
source .venv/bin/activate           # Linux/macOS
pip install -r requirements.txt
jupyter notebook                    # o jupyter lab
```

---

## 4. Fases de implementación

Cada fase es **independiente y validable**. No se avanza a la siguiente sin validar la anterior.

### Fase 0 — Setup del proyecto

**Objetivo:** dejar el repo arrancable.

**Tareas:**
1. Crear estructura de carpetas: `data/`, `outputs/`.
2. Escribir `requirements.txt`.
3. Crear y poblar el venv (`python3 -m venv .venv` + `pip install -r requirements.txt`).
4. Crear `notebook.ipynb` vacío con el kernel del venv seleccionado.
5. Marcar `.venv/`, `data/*.csv` (cache descargado) y `__pycache__/` en `.gitignore` (verificar que ya están; si no, agregar).

**Validación:** `jupyter notebook` abre el archivo y el kernel reporta Python del venv.

---

### Fase 1 — Carga y exploración de datos

**Objetivo:** confirmar que el dataset está sano y dimensionado para SVM.

**Tareas en celdas del notebook:**
1. **Celda de imports:** `pandas`, `numpy`, `os`, `pathlib.Path`.
2. **Celda de descarga/lectura:**
   - URL raw: `https://raw.githubusercontent.com/tonmcg/US_County_Level_Election_Results_08-24/main/2020_US_County_Level_Presidential_Results.csv`
   - Si `data/2020_US_County_Level_Presidential_Results.csv` no existe → descargar con `urllib.request.urlretrieve` y guardar; si existe → leer local (cache).
   - Leer con `pd.read_csv`.
3. **Inspección inicial:** `df.shape`, `df.dtypes`, `df.head()`, `df.isnull().sum()`.
4. **Distribución de la clase provisional:** `df['per_point_diff'].describe()`, conteo de `per_point_diff > 0` vs `<= 0` (esto anticipa el desbalance).
5. **Stats descriptivas de las 3 features candidatas:** `df[['votes_gop','votes_dem','total_votes']].describe()`.
6. **Markdown explicativo:** comentar que la proporción esperada es ~75% GOP / ~25% DEM a nivel condado (sesgo rural), lo que justifica el split estratificado de la Fase 2.

**Salidas esperadas:**
- `df.shape` ≈ `(~3112, 10)`.
- `votes_gop`, `votes_dem`, `total_votes` como `int64` o `float64`, sin nulos.
- Distribución GOP/DEM: reportar el % exacto.

**Validación:** las cifras son consistentes con la documentación del repo fuente (tonmcg).

---

### Fase 2 — Preprocesamiento

**Objetivo:** producir `X_train`, `X_test`, `y_train`, `y_test` listos para SVM.

**Tareas:**
1. **Crear la clase `winner`:**
   ```python
   df['winner'] = np.where(df['per_point_diff'] > 0, 'GOP', 'DEM')
   ```
   - Verificar con `df['winner'].value_counts(normalize=True)`.
2. **Verificar nulos en las 3 features y en `per_point_diff`.** Si hay, documentar la decisión (drop o imputar; lo más probable: no hay).
3. **Definir `X` e `y`:**
   ```python
   feature_cols = ['votes_gop', 'votes_dem', 'total_votes']
   X = df[feature_cols].values
   y = df['winner'].values
   ```
4. **Split 80/20 estratificado:**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.20, random_state=42, stratify=y
   )
   ```
5. **Escalado (fit en train, transform en test):**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test  = scaler.transform(X_test)
   ```
6. **Imprimir shapes** y proporción de clases en `y_train` vs `y_test` (deben ser prácticamente iguales por estratificación).

**Validación:**
- `X_train.shape == (2490, 3)`, `X_test.shape == (~622, 3)` (aprox).
- Estratificación confirmada: `y_train` y `y_test` con el mismo % GOP/DEM.

---

### Fase 3 — Entrenamiento SVM (5 valores de C)

**Objetivo:** entrenar 5 modelos y mostrar el efecto del parámetro `C`.

**Tareas:**
1. Importar `SVC` y definir:
   ```python
   C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
   models = {f'C={c}': SVC(kernel='linear', C=c, random_state=42) for c in C_values}
   ```
2. Entrenar cada uno: `models[name].fit(X_train, y_train)`.
3. Guardar cada modelo entrenado en un dict (no reentrenar).

**Validación:** los 5 modelos entrenan sin error; warning esperado si `C=0.01` y datos no son perfectamente separables (no es problema).

---

### Fase 4 — Evaluación

**Objetivo:** comparar el efecto de `C` con métricas estándar.

**Tareas:**
1. Para cada modelo: predecir `y_pred = model.predict(X_test)`.
2. Calcular `accuracy_score`, `confusion_matrix`, `classification_report`.
3. Construir un `pd.DataFrame` resumen con columnas: `C`, `accuracy`, `precision_GOP`, `recall_GOP`, `f1_GOP`, `precision_DEM`, `recall_DEM`, `f1_DEM`, `n_support_vectors`.
4. **Markdown interpretativo:** comentar que con datos linealmente separables las métricas serán altas (~0.95+) en todos los `C`, pero el número de vectores de soporte y la ubicación del hiperplano cambian. Explicar que el SVM lineal con `votes_gop` y `votes_dem` es casi trivial porque un condado con más votos GOP que DEM siempre será GOP.

**Validación:** tabla con 5 filas, accuracy reportado.

---

### Fase 5 — Sistema de predicción

**Objetivo:** función reutilizable que reciba votos crudos de un condado hipotético y devuelva `GOP` o `DEM`.

**Tareas:**
1. Definir:
   ```python
   def predict_county(votes_gop: int, votes_dem: int, total_votes: int,
                      model, scaler) -> str:
       x = scaler.transform([[votes_gop, votes_dem, total_votes]])
       return model.predict(x)[0]
   ```
   - Usar el **mejor modelo** de la Fase 4 (sugerencia: `C=1.0` o el que tenga mejor F1 de la clase minoritaria DEM, dado el desbalance).
2. Probar con 3 condados inventados que cubran casos extremos:
   - **GOP claro:** votos_gop=30000, votos_dem=10000, total_votes=45000.
   - **DEM claro:** votos_gop=5000, votos_dem=25000, total_votes=32000.
   - **Reñido:** votos_gop=15000, votos_dem=14800, total_votes=30500.
3. Imprimir predicción para cada uno y verificar coherencia.

**Validación:** las 3 predicciones son `GOP`, `DEM`, y la tercera depende del modelo (documentar el resultado).

---

### Fase 6 — Exportación a ARFF

**Objetivo:** producir `outputs/dataset_2020_preprocesado.arff` listo para WEKA.

**Tareas:**
1. **Importante:** el `.arff` se construye sobre el `df` **crudo** (sin escalar) para que WEKA aplique su propio `Standardize` (consistente con la decisión "Sin escalar + filtro WEKA" de la validación).
2. Estructura del archivo:
   ```
   @RELATION us_county_election_2020_svm

   @ATTRIBUTE votes_gop     NUMERIC
   @ATTRIBUTE votes_dem     NUMERIC
   @ATTRIBUTE total_votes   NUMERIC
   @ATTRIBUTE winner        {GOP, DEM}

   @DATA
   12345,6789,19500,GOP
   ...
   ```
3. **Reglas de formato:**
   - Nombre de relación sin espacios (usar guion bajo).
   - `@ATTRIBUTE winner {GOP, DEM}` con la lista exacta entre llaves, separada por coma+espacio.
   - Una instancia por línea: 3 números enteros + clase, separados por coma, sin espacios.
   - Codificación UTF-8, fin de línea `\n`.
4. **Validación post-escritura:** leer el archivo de vuelta con `open()` y verificar que el header tiene 5 líneas (`@RELATION` + 4 `@ATTRIBUTE`) y que el conteo de líneas `@DATA` coincide con `len(df)`.

**Salida:** `outputs/dataset_2020_preprocesado.arff`.

---

### Fase 7 — `INSTRUCCIONES_WEKA.md`

**Objetivo:** guía reproducible paso a paso.

**Contenido obligatorio:**
1. Abrir WEKA Explorer.
2. `Open file…` → seleccionar `outputs/dataset_2020_preprocesado.arff`.
3. **Aplicar filtro de estandarización:**
   - Pestaña *Preprocess* → *Filter* → *Choose* → `weka.filters.unsupervised.attribute.Standardize`.
   - *Apply* a las 3 columnas numéricas (`votes_gop`, `votes_dem`, `total_votes`); **NO** aplicarlo a `winner`.
4. Pestaña *Classify* → *Choose* → `weka.classifiers.functions.SMO`.
5. Configurar la clase: click en la caja de texto bajo *More options* y verificar que la clase es `(Nom) winner`.
6. **Test options:** seleccionar **Percentage split**, valor `80%`. Fijar el seed en `42` (campo *Random seed for XVAL / Percentage Split* — WEKA lo llama "Seed").
7. *Start*.
8. Copiar del panel de resultados: accuracy, matriz de confusión, precision/recall/F-measure por clase.
9. **Nota sobre la comparación:** aclarar que el split de WEKA **no es exactamente** el mismo que el de Python (los algoritmos de muestreo aleatorio difieren), por lo que pequeñas variaciones de accuracy (~0.5-1%) son esperables y no indican error.

---

### Fase 8 — `comparacion_python_vs_weka.md` (plantilla)

**Objetivo:** tabla lista para que el usuario pegue los resultados de WEKA.

**Contenido:**
- Cabecera con contexto (mismo dataset, mismo kernel lineal, mismo C, misma escala).
- Tabla Markdown con columnas: `Métrica | Python (sklearn) | WEKA (SMO) | Δ`.
- Filas: `Accuracy`, `Precision GOP`, `Recall GOP`, `F1 GOP`, `Precision DEM`, `Recall DEM`, `F1 DEM`, `Matriz confusión (texto)`.
- Sección "Observaciones" con bullets pre-armados sobre las posibles causas de diferencia (muestreo, parámetros por defecto, implementación interna de SMO, semilla).
- Sección "Conclusiones" vacía para que el usuario redacte.

---

### Fase 9 (opcional) — Visualización del hiperplano

**Objetivo:** ilustrar SVM con 2 features para una vista 2D.

**Tareas:**
1. Reentrenar SVM con solo 2 features: `votes_gop` y `votes_dem` (ignorando `total_votes`).
2. Pintar:
   - Puntos de entrenamiento coloreados por clase.
   - Hiperplano `w·x + b = 0`.
   - Márgenes `w·x + b = ±1`.
   - Vectores de soporte destacados.
3. Usar `matplotlib` con `DecisionBoundaryDisplay` o绘制 manual con `meshgrid` + `contourf`.

**Validación:** gráfico exportado a `outputs/svm_hiperplano_2d.png` o mostrado en el notebook.

---

## 5. Criterios de aceptación (DoD del proyecto)

- [ ] `notebook.ipynb` ejecuta de principio a fin sin errores (menú *Run All*).
- [ ] `outputs/dataset_2020_preprocesado.arff` se genera y WEKA lo carga sin warnings.
- [ ] Accuracy en test (Python) > 0.90 con al menos uno de los valores de C probados.
- [ ] `predict_county` devuelve predicciones coherentes para los 3 casos de prueba.
- [ ] `INSTRUCCIONES_WEKA.md` permite a una persona con WEKA básico replicar el experimento en ≤10 minutos.
- [ ] `README.md` permite a una persona nueva clonar el repo, crear el venv y ejecutar el notebook sin ayuda externa.

---

## 6. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Cambios en la URL del CSV fuente | Usar `try/except` y mensaje claro; cache local en `data/` |
| Versión de `scikit-learn` cambia la firma de `SVC` | Fijar versión mínima, no máxima, en `requirements.txt` |
| WEKA no disponible en Linux headless | El usuario corre WEKA en su máquina; yo entrego `.arff` + instrucciones |
| Discrepancia grande Python vs WEKA | Documentar la causa probable (muestreo, defaults) en el reporte |
| Clase minoritaria (DEM) subrepresentada | Reportar métricas por clase, no solo accuracy global |

---

## 7. Orden de ejecución

1. Fase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 (opcional).
2. Cada fase se valida antes de avanzar.
3. Cualquier desviación se documenta en este archivo.
