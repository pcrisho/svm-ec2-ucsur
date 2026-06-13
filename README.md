# svm-ec2-ucsur

Sistema en Python que entrena un clasificador **SVM** (`sklearn.svm.SVC`, kernel lineal) sobre resultados electorales presidenciales 2020 a nivel condado (EE.UU.) y compara sus resultados con el clasificador `SMO` de **WEKA** sobre el mismo dataset exportado a `.arff`.

- **Requerimientos:** ver [`REQUERIMENTS.md`](REQUERIMENTS.md)
- **Plan detallado de implementación:** ver [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md)
- **Entregable principal:** [`notebook.ipynb`](notebook.ipynb)
- **Insumo para WEKA:** [`outputs/dataset_2020_preprocesado.arff`](outputs/dataset_2020_preprocesado.arff)

---

## 1. Requisitos

- **Python** ≥ 3.10 (probado con 3.14)
- **pip**
- **Jupyter** (Notebook o Lab) — se instala como dependencia
- **WEKA** (opcional, solo para la parte de comparación) — descargar desde [weka.io](https://www.cs.waikato.ac.nz/ml/weka/) o el paquete de la distribución

---

## 2. Setup del entorno virtual (venv)

Se usa `venv` (módulo estándar de Python) para aislar las dependencias del proyecto.

### 2.1 Crear el entorno

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
```

Esto crea la carpeta `.venv/` con un Python y `pip` aislados.

### 2.2 Activar el entorno

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```cmd
.venv\Scripts\activate.bat
```

Activado, el prompt de la terminal se ve así:

```
(.venv) usuario@maquina:~/svm-ec2-ucsur$
```

### 2.3 Instalar dependencias

Con el venv **activo**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.4 Verificar instalación

```bash
python -c "import sklearn, pandas, numpy, matplotlib, jupyter; print('OK')"
```

Debe imprimir `OK`.

### 2.5 Desactivar el entorno

Cuando termines de trabajar:

```bash
deactivate
```

### 2.6 Notas

- La carpeta `.venv/` está incluida en `.gitignore`; **no** se sube al repositorio.
- Cada vez que clones el repo en una máquina nueva, repite los pasos 2.1 → 2.3.
- Si tu IDE (VSCode, PyCharm) pregunta por el intérprete, apúntalo a `.venv/bin/python` (Linux/macOS) o `.venv\Scripts\python.exe` (Windows).

---

## 3. Ejecución

### 3.1 Abrir el notebook

Con el venv activo:

```bash
jupyter notebook
```

o, si prefieres:

```bash
jupyter lab
```

Se abrirá el navegador con la lista de archivos. Abre `notebook.ipynb` y ejecuta las celdas con *Run All* (menú *Cell*).

### 3.2 Qué produce el notebook

- `data/2020_US_County_Level_Presidential_Results.csv` — cache local del dataset (no se sube al repo).
- `outputs/dataset_2020_preprocesado.arff` — dataset preprocesado listo para WEKA.
- Gráficos y tablas en pantalla.
- Métricas de evaluación (accuracy, matriz de confusión, F1 por clase) para 5 valores de `C`.

### 3.3 Comparar con WEKA (opcional)

Sigue los pasos de [`outputs/INSTRUCCIONES_WEKA.md`](outputs/INSTRUCCIONES_WEKA.md) y pega los resultados en [`outputs/comparacion_python_vs_weka.md`](outputs/comparacion_python_vs_weka.md).

---

## 4. Estructura del repositorio

```
svm-ec2-ucsur/
├── .venv/                                       # entorno virtual (ignorado)
├── data/                                        # cache del CSV (ignorado)
├── outputs/                                     # entregables para WEKA + reportes
│   ├── dataset_2020_preprocesado.arff
│   ├── INSTRUCCIONES_WEKA.md
│   └── comparacion_python_vs_weka.md
├── IMPLEMENTATION-PLAN.md                       # masterfile del proyecto
├── REQUERIMENTS.md                              # qué se quiere construir
├── README.md                                    # este archivo
├── requirements.txt                             # dependencias
└── notebook.ipynb                               # entregable principal
```

---

## 5. Flujo de trabajo resumido

```
1. Clonar repo
2. python3 -m venv .venv
3. source .venv/bin/activate
4. pip install -r requirements.txt
5. jupyter notebook
6. Abrir notebook.ipynb → Run All
7. (Opcional) Abrir outputs/dataset_2020_preprocesado.arff en WEKA
8. (Opcional) Pegar resultados de WEKA en outputs/comparacion_python_vs_weka.md
```

---

## 6. Solución de problemas

| Problema | Solución |
|---|---|
| `python3: command not found` | Instala Python 3.10+ desde [python.org](https://www.python.org/) o tu gestor de paquetes. |
| `No module named 'sklearn'` | El venv no está activo. Ejecuta `source .venv/bin/activate` y reinstala. |
| `jupyter: command not found` | Activa el venv y ejecuta `pip install -r requirements.txt`. |
| El notebook no encuentra el CSV | Ejecuta la celda de descarga; el archivo se guarda en `data/`. |
| WEKA no carga el `.arff` | Verifica que la codificación sea UTF-8 y que la primera línea sea `@RELATION ...`. |

---

## 7. Referencias

- Dataset: [tonmcg/US_County_Level_Election_Results_08-24](https://github.com/tonmcg/US_County_Level_Election_Results_08-24)
- [scikit-learn — SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [WEKA — SMO](https://weka.sourceforge.io/doc.dev/weka/classifiers/functions/SMO.html)
