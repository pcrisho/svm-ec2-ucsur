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
# FASE 6 — EXPORTACIÓN A ARFF (PARA WEKA)
# ========================================================================
cells.append(nbf.v4.new_markdown_cell(
    """---

## Fase 6 — Exportación a ARFF para WEKA

**Objetivo:** producir `outputs/dataset_2020_preprocesado.arff` listo para abrir en WEKA Explorer.

**Decisiones de formato:**
- Se exportan los datos **crudos** (sin escalar) para que WEKA aplique el filtro `Standardize` en su propio flujo, garantizando paridad de escala con lo que ve Python.
- Solo 3 atributos numéricos + 1 atributo nominal `winner {GOP, DEM}`.
- Codificación UTF-8, fin de línea `\n`, sin espacios entre campos."""
))

cells.append(nbf.v4.new_markdown_cell("### 6.1 Escribir el archivo ARFF"))
cells.append(nbf.v4.new_code_cell(
    """OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
ARFF_PATH = OUTPUT_DIR / 'dataset_2020_preprocesado.arff'

relation_name = 'us_county_election_2020_svm'
attributes_block = '\\n'.join(
    f'@ATTRIBUTE {col:<13} NUMERIC' if col != 'winner'
    else f'@ATTRIBUTE {col:<13} {{GOP, DEM}}'
    for col in feature_cols + ['winner']
)

with open(ARFF_PATH, 'w', encoding='utf-8') as f:
    f.write(f'@RELATION {relation_name}\\n')
    f.write('\\n')
    f.write(attributes_block + '\\n')
    f.write('\\n')
    f.write('@DATA\\n')
    for _, row in df[feature_cols + ['winner']].iterrows():
        f.write(f\"{int(row['votes_gop'])},{int(row['votes_dem'])},\"\n                f\"{int(row['total_votes'])},{row['winner']}\\n\")

print(f'ARFF escrito en: {ARFF_PATH}')
print(f'Tamaño: {ARFF_PATH.stat().st_size / 1024:.1f} KB')"""
))

cells.append(nbf.v4.new_markdown_cell("### 6.2 Validar la estructura del ARFF"))
cells.append(nbf.v4.new_code_cell(
    """with open(ARFF_PATH, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

header_lines  = [l for l in lines if l.startswith('@')]
data_lines    = [l for l in lines if l and not l.startswith('@') and not l.isspace()]

print('=== ENCABEZADO ===')
for l in header_lines:
    print(l)
print()
print(f'Líneas de @DATA: {len(data_lines)} (esperado: {len(df)})')
assert len(data_lines) == len(df), 'Discrepancia en el número de filas'
assert '@RELATION ' in lines[0]
assert any('@ATTRIBUTE winner' in l and '{GOP, DEM}' in l for l in lines)
print('Validación OK: estructura ARFF correcta.')"""
))

cells.append(nbf.v4.new_markdown_cell("### 6.3 Primeras líneas del archivo (vista previa)"))
cells.append(nbf.v4.new_code_cell(
    """with open(ARFF_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 10:
            print(line.rstrip())
        else:
            break"""
))

cells.append(nbf.v4.new_markdown_cell(
    """### 6.4 Resumen de la Fase 6

- ✅ `outputs/dataset_2020_preprocesado.arff` generado (3 numéricas + `winner` nominal).
- ✅ 3,152 instancias (1 por condado), sin escalado.
- ✅ Codificación UTF-8 y sintaxis ARFF estándar (compatible con WEKA Explorer).

**Listo para la Fase 7 (instrucciones WEKA).**"""
))

# ========================================================================
# FASE 9 (OPCIONAL) — VISUALIZACIÓN DEL HIPERPLANO
# ========================================================================
cells.append(nbf.v4.new_markdown_cell(
    """---

## Fase 9 (opcional) — Visualización del hiperplano SVM en 2D

**Objetivo:** ilustrar el comportamiento del SVM lineal usando solo 2 features (`votes_gop`, `votes_dem`) para poder pintar el plano.

- Se reentrena un SVM con esas 2 features (sin `total_votes`).
- Se pinta:
  - Puntos de entrenamiento coloreados por clase.
  - **Hiperplano** de decisión: `w·x + b = 0`.
  - **Márgenes**: `w·x + b = ±1`.
  - **Vectores de soporte** destacados.
- Gráfico exportado a `outputs/svm_hiperplano_2d.png`."""
))

cells.append(nbf.v4.new_markdown_cell("### 9.1 Reentrenar SVM con 2 features"))
cells.append(nbf.v4.new_code_cell(
    """from sklearn.svm import SVC

# Tomar el mejor C de la Fase 4 (C=100)
C_best = float(best_name.split('=')[1])

# Subset: solo votes_gop y votes_dem (índices 0 y 1)
X_train_2d = X_train[:, :2]
X_test_2d  = X_test[:, :2]

svm_2d = SVC(kernel='linear', C=C_best, random_state=RANDOM_STATE)
svm_2d.fit(X_train_2d, y_train)
print(f'SVM 2D entrenado con C={C_best}')
print(f'Vectores de soporte: {svm_2d.n_support_.tolist()}')
print(f'Coeficientes (w): {svm_2d.coef_[0]}')
print(f'Intercepto (b):   {svm_2d.intercept_[0]:.4f}')"""
))

cells.append(nbf.v4.new_markdown_cell("### 9.2 Pintar y guardar el gráfico"))
cells.append(nbf.v4.new_code_cell(
    """import matplotlib.pyplot as plt

w = svm_2d.coef_[0]
b = svm_2d.intercept_[0]

# Crear meshgrid sobre el rango de las 2 features (en escala StandardScaler)
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500),
)
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Mapeo de clases a colores
color_map = {'GOP': '#d62728', 'DEM': '#1f77b4'}

fig, ax = plt.subplots(figsize=(10, 8))
ax.contourf(xx, yy, (Z == 'GOP').astype(int), alpha=0.15,
            levels=[-0.5, 0.5, 1.5], colors=['#1f77b4', '#d62728'])

# Puntos de entrenamiento
for cls, color in color_map.items():
    mask = y_train == cls
    ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
               c=color, label=f'Train {cls} (n={mask.sum()})',
               s=18, alpha=0.5, edgecolors='none')

# Vectores de soporte (con borde)
ax.scatter(svm_2d.support_vectors_[:, 0], svm_2d.support_vectors_[:, 1],
           s=70, facecolors='none', edgecolors='k', linewidths=1.2,
           label=f'Vectores de soporte (n={len(svm_2d.support_)})')

# Hiperplano y márgenes
# w·x + b = 0   ->   y = -(w0/w1)*x - b/w1
x_line = np.linspace(x_min, x_max, 100)
if abs(w[1]) > 1e-9:
    y_hyper  = -(w[0] / w[1]) * x_line - b / w[1]
    y_margin1 = -(w[0] / w[1]) * x_line - (b - 1) / w[1]
    y_margin2 = -(w[0] / w[1]) * x_line - (b + 1) / w[1]
    ax.plot(x_line, y_hyper,  'k-',  linewidth=2, label='Hiperplano w·x+b=0')
    ax.plot(x_line, y_margin1, 'k--', linewidth=1, label='Margen w·x+b=+1')
    ax.plot(x_line, y_margin2, 'k--', linewidth=1, label='Margen w·x+b=-1')

ax.set_xlabel('votes_gop (escalado)')
ax.set_ylabel('votes_dem (escalado)')
ax.set_title(f'SVM lineal (C={C_best}) sobre votes_gop vs votes_dem')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

fig.tight_layout()
fig_path = OUTPUT_DIR / 'svm_hiperplano_2d.png'
fig.savefig(fig_path, dpi=120)
plt.show()
print(f'Gráfico guardado en: {fig_path}')"""
))

cells.append(nbf.v4.new_markdown_cell(
    """### 9.3 Resumen de la Fase 9

- ✅ SVM reentrenado con 2 features (`votes_gop`, `votes_dem`) usando el mejor `C` de la Fase 4.
- ✅ Hiperplano, márgenes y vectores de soporte visualizados.
- ✅ Gráfico exportado a `outputs/svm_hiperplano_2d.png`.

**Con esto se completan todas las fases del proyecto (0-9).** La visualización confirma visualmente que el SVM lineal separa las clases con un hiperplano muy cercano a la diagonal `votes_gop = votes_dem` (con un pequeño sesgo hacia la clase mayoritaria GOP, consistente con el desbalance 82/18)."""
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
