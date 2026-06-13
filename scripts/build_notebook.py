"""
Genera el notebook del proyecto con las celdas de las Fases 1 y 2.
Las fases siguientes se agregan luego extendiendo este script.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# ========================================================================
# FASE 1 — CARGA + EDA
# ========================================================================
cells.append(nbf.v4.new_markdown_cell(
    """# SVM para clasificación de resultados electorales 2020 (EE.UU. por condado)

**Proyecto:** SVM EC2 UCSUR  
**Plan:** [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md)  
**Requerimientos:** [`REQUERIMENTS.md`](REQUERIMENTS.md)

---

## Fase 1 — Carga y exploración de datos

**Objetivo:** confirmar que el dataset está sano y dimensionado para SVM (~3,100 condados).

Decisiones técnicas aplicadas (ver `IMPLEMENTATION-PLAN.md` §0):
- Features candidatas: `votes_gop`, `votes_dem`, `total_votes` (las únicas sin riesgo de data leakage).
- Clase objetivo `winner`: `GOP` si `per_point_diff > 0`, `DEM` en caso contrario.
- Fuente: [`tonmcg/US_County_Level_Election_Results_08-24`](https://github.com/tonmcg/US_County_Level_Election_Results_08-24)."""
))

cells.append(nbf.v4.new_markdown_cell("### 1.1 Imports y configuración"))
cells.append(nbf.v4.new_code_cell(
    """import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
pd.set_option('display.max_columns', 50)

RANDOM_STATE = 42
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / '2020_US_County_Level_Presidential_Results.csv'
CSV_URL = ('https://raw.githubusercontent.com/tonmcg/'
           'US_County_Level_Election_Results_08-24/master/'
           '2020_US_County_Level_Presidential_Results.csv')

print('Setup OK. Versiones:')
print(f'  pandas  : {pd.__version__}')
print(f'  numpy   : {np.__version__}')"""
))

cells.append(nbf.v4.new_markdown_cell(
    """### 1.2 Descarga (con cache) y lectura del CSV

Si el archivo ya existe localmente en `data/`, se reutiliza (cache). Si no, se descarga desde el raw de GitHub."""
))
cells.append(nbf.v4.new_code_cell(
    """if not CSV_PATH.exists():
    print(f'Descargando {CSV_URL} ...')
    urlretrieve(CSV_URL, CSV_PATH)
    print(f'Guardado en {CSV_PATH} ({CSV_PATH.stat().st_size / 1024:.1f} KB)')
else:
    print(f'Usando cache local: {CSV_PATH} ({CSV_PATH.stat().st_size / 1024:.1f} KB)')

df = pd.read_csv(CSV_PATH)
print(f'\\nDimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas')"""
))

cells.append(nbf.v4.new_markdown_cell("### 1.3 Inspección inicial: tipos, nulos, primeras filas"))
cells.append(nbf.v4.new_code_cell(
    """print('=== TIPOS DE DATOS ===')
print(df.dtypes)
print()
print('=== VALORES NULOS POR COLUMNA ===')
print(df.isnull().sum())
print()
print('=== PRIMERAS 5 FILAS ===')
df.head()"""
))

cells.append(nbf.v4.new_markdown_cell(
    """### 1.4 Distribución de la clase provisional `winner`

Construimos `winner` provisionalmente (sin guardar la columna todavía, solo para inspección) a partir de `per_point_diff` para confirmar la proporción esperada de condados GOP vs DEM."""
))
cells.append(nbf.v4.new_code_cell(
    """winner_provisional = np.where(df['per_point_diff'] > 0, 'GOP', 'DEM')
dist = pd.Series(winner_provisional).value_counts(normalize=True).mul(100).round(2)
print('=== DISTRIBUCIÓN DE CLASES (provisional) ===')
print(dist.astype(str) + ' %')
print()
print('Conteos absolutos:')
print(pd.Series(winner_provisional).value_counts())"""
))
cells.append(nbf.v4.new_markdown_cell(
    """**Interpretación esperada:** aproximadamente 75% GOP y 25% DEM a nivel condado (sesgo rural). Esto confirma el desbalance y justifica el uso de `stratify=y` en la partición de la Fase 2, así como reportar métricas por clase (no solo accuracy global)."""
))

cells.append(nbf.v4.new_markdown_cell("### 1.5 Estadísticas descriptivas de las 3 features candidatas"))
cells.append(nbf.v4.new_code_cell(
    """feature_cols = ['votes_gop', 'votes_dem', 'total_votes']
df[feature_cols].describe().T"""
))

cells.append(nbf.v4.new_markdown_cell(
    """### 1.6 Resumen de la Fase 1

**Lo que se valida aquí:**
- `df.shape` debe ser cercano a `(~3112, 10)`.
- Las 3 features (`votes_gop`, `votes_dem`, `total_votes`) deben ser numéricas y sin nulos.
- La proporción GOP/DEM debe ser coherente con la realidad electoral 2020 a nivel condado (~75/25).

Si los números anteriores son consistentes, se puede avanzar a la **Fase 2 (Preprocesamiento)**."""
))

# ========================================================================
# FASE 2 — PREPROCESAMIENTO
# ========================================================================
cells.append(nbf.v4.new_markdown_cell(
    """---

## Fase 2 — Preprocesamiento

**Objetivo:** producir `X_train`, `X_test`, `y_train`, `y_test` listos para SVM.

Pasos:
1. Crear la clase nominal `winner` desde `per_point_diff`.
2. Verificar nulos en features y clase.
3. Definir `X` (3 features crudas) e `y` (clase).
4. Split 80/20 **estratificado** con `random_state=42`.
5. `StandardScaler` (fit en train, transform en test)."""
))

cells.append(nbf.v4.new_markdown_cell("### 2.1 Crear la clase `winner`"))
cells.append(nbf.v4.new_code_cell(
    """df['winner'] = np.where(df['per_point_diff'] > 0, 'GOP', 'DEM')
print('Conteo de la clase winner:')
print(df['winner'].value_counts())
print()
print('Proporción:')
print(df['winner'].value_counts(normalize=True).mul(100).round(2).astype(str) + ' %')"""
))

cells.append(nbf.v4.new_markdown_cell("### 2.2 Verificar nulos en features y en la clase"))
cells.append(nbf.v4.new_code_cell(
    """nulos_features = df[feature_cols].isnull().sum().sum()
nulos_clase     = df['winner'].isnull().sum()
print(f'Nulos en features ({feature_cols}): {nulos_features}')
print(f'Nulos en winner: {nulos_clase}')
assert nulos_features == 0 and nulos_clase == 0, 'Hay nulos a tratar'
print('OK: sin nulos en features ni en winner.')"""
))

cells.append(nbf.v4.new_markdown_cell("### 2.3 Definir X e y, y split 80/20 estratificado"))
cells.append(nbf.v4.new_code_cell(
    """from sklearn.model_selection import train_test_split

X = df[feature_cols].values
y = df['winner'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f'X_train: {X_train.shape}  X_test: {X_test.shape}')
print(f'y_train: {y_train.shape}  y_test: {y_test.shape}')
print()
print('Proporción de clases en y_train:', np.round(np.mean(y_train == 'GOP') * 100, 2), '% GOP')
print('Proporción de clases en y_test :', np.round(np.mean(y_test  == 'GOP') * 100, 2), '% GOP')
print()
print('La estratificación garantiza que la proporción GOP/DEM sea ~idéntica en train y test.')"""
))

cells.append(nbf.v4.new_markdown_cell("### 2.4 Escalado con `StandardScaler` (fit en train, transform en test)"))
cells.append(nbf.v4.new_code_cell(
    """from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print('Medias de X_train (post-escalado):', np.round(X_train.mean(axis=0), 6).tolist())
print('Stds  de X_train (post-escalado):', np.round(X_train.std(axis=0),  6).tolist())
print()
print('X_train[:3] (muestra):')
print(X_train[:3])"""
))

cells.append(nbf.v4.new_markdown_cell(
    """### 2.5 Resumen de la Fase 2

- ✅ Clase `winner` creada: 82.33% GOP, 17.67% DEM.
- ✅ Sin nulos en features ni en la clase.
- ✅ Split 80/20 estratificado (`random_state=42`): ~2521 train / ~631 test.
- ✅ Features estandarizadas (media 0, std 1 en train; misma transformación aplicada a test).

**Listo para entrenar el SVM en la Fase 3.**"""
))

# ========================================================================
# FASE 3 — ENTRENAMIENTO SVM
# ========================================================================
cells.append(nbf.v4.new_markdown_cell(
    """---

## Fase 3 — Entrenamiento del SVM lineal

**Objetivo:** entrenar 5 modelos con diferentes valores de `C` para mostrar la transición margen blando → duro.

- Algoritmo: `sklearn.svm.SVC(kernel='linear')`.
- `C` controla la penalización por errores en entrenamiento:
  - **C bajo (margen blando):** permite más errores, margen más ancho, mejor generalización esperada.
  - **C alto (margen duro):** castiga mucho los errores, margen más estrecho, riesgo de overfitting."""
))

cells.append(nbf.v4.new_markdown_cell("### 3.1 Entrenar los 5 modelos"))
cells.append(nbf.v4.new_code_cell(
    """from sklearn.svm import SVC

C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
models = {f'C={c}': SVC(kernel='linear', C=c, random_state=RANDOM_STATE) for c in C_values}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f'Entrenado {name:>8}  |  n_support = {model.n_support_.tolist()}  |  total SV = {model.support_.shape[0]}')"""
))

# ========================================================================
# FASE 4 — EVALUACIÓN
# ========================================================================
cells.append(nbf.v4.new_markdown_cell(
    """---

## Fase 4 — Evaluación

Métricas por modelo:
- `accuracy` global.
- `precision`, `recall`, `f1` **por clase** (GOP y DEM) — importante por el desbalance.
- `confusion_matrix`.
- Número de vectores de soporte."""
))

cells.append(nbf.v4.new_markdown_cell("### 4.1 Métricas por modelo (tabla comparativa)"))
cells.append(nbf.v4.new_code_cell(
    """from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

rows = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    rows.append({
        'Modelo'      : name,
        'Accuracy'    : round(acc, 4),
        'Prec_GOP'    : round(report['GOP']['precision'], 4),
        'Recall_GOP'  : round(report['GOP']['recall'], 4),
        'F1_GOP'      : round(report['GOP']['f1-score'], 4),
        'Prec_DEM'    : round(report['DEM']['precision'], 4),
        'Recall_DEM'  : round(report['DEM']['recall'], 4),
        'F1_DEM'      : round(report['DEM']['f1-score'], 4),
        'N_SV'        : int(model.support_.shape[0]),
    })

df_results = pd.DataFrame(rows)
df_results"""
))

cells.append(nbf.v4.new_markdown_cell("### 4.2 Matriz de confusión del mejor modelo (mayor F1-DEM)"))
cells.append(nbf.v4.new_code_cell(
    """best_row = df_results.loc[df_results['F1_DEM'].idxmax()]
best_name = best_row['Modelo']
best_model = models[best_name]

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best, labels=['GOP', 'DEM'])

print(f'Mejor modelo según F1-DEM: {best_name}')
print(f'Accuracy = {best_row["Accuracy"]} | F1_DEM = {best_row["F1_DEM"]} | F1_GOP = {best_row["F1_GOP"]}')
print()
print('Matriz de confusión (filas = real, columnas = predicho):')
print(pd.DataFrame(cm, index=['real_GOP', 'real_DEM'], columns=['pred_GOP', 'pred_DEM']))
print()
print('Reporte completo:')
print(classification_report(y_test, y_pred_best, digits=4, zero_division=0))"""
))

cells.append(nbf.v4.new_markdown_cell(
    """**Interpretación esperada:** con las 3 features crudas (`votes_gop`, `votes_dem`, `total_votes`) y un kernel lineal, el problema es **casi linealmente separable**: el hiperplano `votes_gop = votes_dem` ya da una accuracy muy alta. Las métricas serán elevadas para todos los valores de `C`; el efecto de `C` se notará sobre todo en el **número de vectores de soporte** y en la ubicación exacta del hiperplano, más que en la accuracy global."""
))

# ========================================================================
# FASE 5 — SISTEMA DE PREDICCIÓN
# ========================================================================
cells.append(nbf.v4.new_markdown_cell(
    """---

## Fase 5 — Sistema de predicción para un condado nuevo

**Objetivo:** exponer una función reutilizable que reciba los votos crudos de un condado hipotético y devuelva `GOP` o `DEM`, aplicando internamente el `StandardScaler` entrenado."""
))

cells.append(nbf.v4.new_markdown_cell("### 5.1 Función `predict_county`"))
cells.append(nbf.v4.new_code_cell(
    """def predict_county(votes_gop: int, votes_dem: int, total_votes: int,
                      model, scaler) -> str:
    \"\"\"Clasifica un condado hipotético a partir de sus votos crudos.

    Parameters
    ----------
    votes_gop, votes_dem, total_votes : int
        Votos crudos del condado (antes de escalar).
    model : sklearn.svm.SVC
        Modelo SVM lineal ya entrenado.
    scaler : sklearn.preprocessing.StandardScaler
        Escalador ya ajustado sobre el conjunto de entrenamiento.

    Returns
    -------
    str
        'GOP' o 'DEM'.
    \"\"\"
    x = scaler.transform([[votes_gop, votes_dem, total_votes]])
    return model.predict(x)[0]

print('Función predict_county() definida.')"""
))

cells.append(nbf.v4.new_markdown_cell("### 5.2 Demostración con 3 condados inventados"))
cells.append(nbf.v4.new_code_cell(
    """examples = [
    ('GOP claro',     30000, 10000, 45000),
    ('DEM claro',      5000, 25000, 32000),
    ('Reñido',        15000, 14800, 30500),
]

print(f'Usando el mejor modelo: {best_name}\\n')
print(f'{\"Caso\":<12} {\"votes_gop\":>10} {\"votes_dem\":>10} {\"total\":>8}   {\"Predicción\":>10}')
print('-' * 60)
for label, vg, vd, vt in examples:
    pred = predict_county(vg, vd, vt, best_model, scaler)
    print(f'{label:<12} {vg:>10,} {vd:>10,} {vt:>8,}   {pred:>10}')"""
))

cells.append(nbf.v4.new_markdown_cell(
    """**Verificación cualitativa:** los casos "GOP claro" y "DEM claro" deberían predecirse correctamente. El caso "reñido" puede caer a uno u otro lado según el modelo y el escalado; el SVM lineal tiende a separar por la diferencia `votes_gop - votes_dem`, así que con una diferencia de +200 debería clasificar como `GOP`.

**Reutilización:** la función `predict_county` puede importarse y usarse en otro script siempre que se le pase el `model` y el `scaler` entrenados (ambos objetos de scikit-learn son serializables con `joblib.dump` si se quisiera persistir)."""
))

# ========================================================================
# Metadata y escritura
# ========================================================================
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3 (.venv)',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {
        'name': 'python',
        'version': '3.14',
    },
}

out = Path('notebook.ipynb')
nbf.write(nb, out)
print(f'Notebook creado: {out} ({out.stat().st_size} bytes, {len(cells)} celdas)')
