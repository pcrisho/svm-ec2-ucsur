# Instrucciones para WEKA — Validación con SMO

> **Objetivo:** replicar en WEKA el experimento de clasificación SVM hecho en Python, sobre el mismo dataset (`outputs/dataset_2020_preprocesado.arff`), y comparar métricas.
>
> **Tiempo estimado:** ≤ 10 minutos (sin contar la descarga/instalación de WEKA).

---

## 0. Prerrequisitos

- **WEKA 3.8 o superior** instalado y con interfaz gráfica (Explorer).
  - Descarga: <https://www.cs.waikato.ac.nz/ml/weka/downloading.html>
  - En Windows/macOS suele venir con instalador; en Linux: `sudo apt install weka` o descomprimir el ZIP y ejecutar `./weka.sh` desde la carpeta.
- El archivo `outputs/dataset_2020_preprocesado.arff` (generado por el notebook en la Fase 6).

---

## 1. Cargar el dataset

1. Abrir **WEKA Explorer** (`weka.gui.Explorer`).
2. Pestaña **Preprocess**.
3. Click en **Open file…** → seleccionar `outputs/dataset_2020_preprocesado.arff`.
4. Verificar en la parte superior: `Relation: us_county_election_2020_svm` y `Instances: 3,152` con 4 atributos.

---

## 2. Aplicar el filtro de estandarización

**WEKA NO escala automáticamente**, así que para que la SVM vea los datos en la misma escala que Python (después de `StandardScaler`), hay que aplicar el filtro manualmente.

1. En la pestaña **Preprocess**, sección **Filter** → click en **Choose**.
2. Navegar a: `filters → unsupervised → attribute → Standardize`.
3. Aplicar al **área de trabajo**:
   - El filtro `Standardize` por defecto procesa **todos los atributos numéricos**. Como `winner` ya es nominal, no lo tocará.
   - Si WEKA preguntara por atributos a aplicar, seleccionar solo `votes_gop`, `votes_dem`, `total_votes` y dejar `winner` sin tocar.
4. Click en **Apply**.

**Verificación rápida:** click en `votes_gop` en la lista de atributos y comprobar que la media es ~0 y la std ~1. Repetir con `votes_dem` y `total_votes`.

> 💡 Alternativa: usar `Normalize` (escala a [0,1]) en lugar de `Standardize`. Los resultados deberían ser muy similares. Para máxima paridad con Python, usar `Standardize`.

---

## 3. Configurar el clasificador SMO

1. Ir a la pestaña **Classify**.
2. Click en **Choose** → `functions → SMO`.
3. Verificar los parámetros del clasificador:
   - **Kernel:** por defecto `PolyKernel` con `exponent = 1` (esto es **equivalente a kernel lineal**, ya que `(x·y + 1)^1 = x·y + 1`).
   - **C (complexity):** cambiar al valor que se quiera comparar. Por defecto es `1.0`. Para replicar el mejor modelo de Python (`C=100`), escribir `100.0` en el campo `C`.
4. **Fijar la clase nominal:**
   - Debajo de la lista de Test options, hay un selector con el texto `Class:`. Verificar que diga `(Nom) winner`.
   - Si no, hacer click y elegir `winner`.

---

## 4. Configurar la evaluación (split 80/20)

> ⚠️ **Nota importante sobre la comparación:** los algoritmos de muestreo aleatorio de WEKA y de `sklearn` (Python) **NO son idénticos** aunque se use la misma semilla. Los splits diferirán en ~0-1% de las filas, lo cual produce variaciones menores en las métricas. La comparación es **ambiental** (mismo dataset + mismo algoritmo + mismo C + misma escala), no una comparación bit-a-bit.

1. En el panel **Test options**, marcar la opción **Percentage split**.
2. En el campo `%` escribir `80`.
3. **Random seed:** WEKA tiene un campo de semilla al lado del botón *More options*. El valor por defecto es `1`, pero para mejor paridad con Python, fijarlo en `42`.

**Configuración resultante:**

| Campo | Valor |
|---|---|
| Test options | Percentage split |
| % | 80 |
| Random seed | 42 |

---

## 5. Ejecutar y leer resultados

1. Click en **Start**.
2. En el panel **Classifier output** (lado derecho) aparecerán:
   - **Correctly Classified Instances** (accuracy)
   - **Incorrectly Classified Instances**
   - **Kappa statistic**
   - **Mean absolute error**, **Root mean squared error**
   - **Confusion Matrix** (formato: `a b <- classified as`)
   - **Detailed Accuracy By Class**, con columnas: `TP Rate`, `FP Rate`, `Precision`, `Recall`, `F-Measure`, `ROC Area`, `Class`.

3. **Copiar al documento comparativo** [`outputs/comparacion_python_vs_weka.md`](comparacion_python_vs_weka.md):
   - Accuracy global.
   - Precision/Recall/F-Measure para la clase `GOP` y para la clase `DEM`.
   - Matriz de confusión (formato texto).

---

## 6. Repetir para otros valores de C (opcional, recomendado)

Si se quiere comparar el efecto de `C` en WEKA (igual que en Python):

1. Volver a la configuración del clasificador (botón *Choose* o click en el nombre del clasificador actual).
2. Cambiar el parámetro `C` a `0.01`, `0.1`, `1.0`, `10.0`, `100.0` (uno por corrida).
3. **Re-Start** cada vez.
4. Apuntar las métricas de cada corrida.

Esto produce una tabla de 5 filas comparable con la `df_results` del notebook (Fase 4.1).

---

## 7. Troubleshooting

| Problema | Solución |
|---|---|
| WEKA no carga el `.arff` | Verificar codificación UTF-8 y que la primera línea sea `@RELATION us_county_election_2020_svm` (sin caracteres raros). |
| `winner` no aparece como `(Nom)` | Revisar que en el `.arff` la línea diga `@ATTRIBUTE winner {GOP, DEM}` con la lista de valores. |
| El accuracy es muy bajo (< 0.5) | Probablemente no se aplicó el filtro `Standardize`. Repetir paso 2. |
| `Cannot handle numeric class` | La clase está mal configurada. En *Class*, seleccionar `(Nom) winner` (no `(Num) winner`). |
| Diferencia grande con Python (> 5%) | Verificar que `C` y `seed` son los mismos; revisar que el filtro `Standardize` se aplicó. Si persiste, revisar la documentación de la versión de WEKA usada (SMO defaults pueden variar entre 3.8 y 3.9). |

---

## 8. Qué se compara (resumen)

| | Python (sklearn) | WEKA (SMO) |
|---|---|---|
| Algoritmo | `SVC(kernel='linear')` | `SMO` con `PolyKernel` exponent=1 |
| C | 100 (mejor) | 100 |
| Split | 80/20 estratificado, seed=42 | 80/20 percentage, seed=42 |
| Escalado | `StandardScaler` (fit en train) | Filtro `Standardize` (en memoria) |
| Métricas | accuracy, F1 por clase, matriz conf. | accuracy, F-Measure por clase, matriz conf. |

Las pequeñas diferencias (~0.5-1%) son **esperables y no indican error**, sino diferencia de implementación interna del muestreo aleatorio y de SMO entre ambas librerías.
