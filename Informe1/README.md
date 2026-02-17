# Detección de Gastos Hormiga en Transacciones Bancarias

Proyecto del curso **Fundamentos de Aprendizaje Automático**.

Estudiante: *Paula Inés Llanos López*

---

## 1. Resumen

Este proyecto construye un **pipeline completo de Machine Learning** para identificar **gastos hormiga** a partir del historial de transacciones bancarias de diferentes personas.

Se combinan:
- Reglas de negocio (palabras clave y umbrales de monto),
- Ingeniería de características numéricas y de texto,
- Un modelo de clasificación supervisado (Random Forest),
- Múltiples **visualizaciones automáticas** para analizar el comportamiento del gasto.

---

## 2. Descripción del Problema

Un **gasto hormiga** se define como:
- Un gasto **frecuente**,
- De **bajo monto**,
- Que típicamente se realiza en comercios como cafeterías, snacks, tiendas pequeñas, etc.

No se consideran gastos hormiga:
- Movimientos bancarios genéricos: transferencias, consignaciones, retiros de cajero, abonos de intereses, cuotas de manejo.
- Gastos grandes o poco frecuentes (ej. matrículas, grandes compras de mercado).
- Algunos conceptos de transporte y salud/farmacia, según reglas definidas.

El objetivo es:
- Etiquetar cada transacción como **gasto hormiga (1)** o **no hormiga (0)**.
- Entrenar un modelo que aprenda a distinguir estos patrones.
- Analizar visualmente el comportamiento del gasto y el desempeño del modelo.

---

## 3. Arquitectura del Proyecto

El proyecto sigue una arquitectura modular, con cada etapa en un archivo separado dentro de `src/`:

- `data_loader.py`: carga y limpieza básica del CSV original.
- `preprocessing.py`: filtrado de gastos y creación de la etiqueta `es_gasto_hormiga`.
- `feature_engineering.py`: creación de variables numéricas (fecha, frecuencia, montos).
- `train.py`: definición y entrenamiento del modelo de Machine Learning.
- `evaluate.py`: métricas cuantitativas y evaluación del modelo.
- `visualizations.py`: todas las gráficas automáticas (EDA y modelo).
- `main.py`: orquestador del pipeline de inicio a fin.

Carpetas principales:

- `data/` → contiene el archivo `Dataset_Hormiga.csv` con las transacciones.
- `src/` → código fuente del pipeline.
- `reports/` → todas las gráficas generadas automáticamente.

---

## 4. Estructura Completa de Archivos

```text
Informe1/
├── data/
│   └── Dataset_Hormiga.csv
│
├── reports/                             # (se llena con las gráficas)
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── train.py
│   └── visualizations.py
└── README.md
```

---

## 5. Instalación y Configuración

1. Activar el entorno virtual (si aplica):

   ```bash
   cd Informe1/src
   # Activar entorno en Windows (ejemplo)
   ..\ml_env\Scripts\activate
   ```

2. Instalar dependencias (si aún no están instaladas):

   ```bash
   pip install -r ..\requirements.txt
   ```


---

## 6. Pipeline Completo Paso a Paso

La ejecución completa se hace con:

```bash
cd Informe1/src
python main.py
```

Dentro de `main.py` el flujo es:

1. **Carga de datos** (`data_loader.load_data`)
   - Lee `../data/Dataset_Hormiga.csv` con separador `;` y codificación `utf-8-sig`.
   - Elimina columnas vacías.
   - Normaliza nombres de columnas a minúsculas.
   - Convierte `fecha` a tipo `datetime`.
   - Aplica `parse_money` a `valor` para tener un `float` limpio.
   - Normaliza `descripcion` a string en minúsculas sin espacios extra.

2. **Filtrado de gastos** (`preprocessing.filter_expenses`)
   - Se quedan solo las filas donde `valor < 0` (salidas de dinero).

3. **Creación de etiquetas** (`preprocessing.create_labels`)
   - Usa reglas de negocio y un umbral `DEFAULT_THRESHOLD = 25000`:
     - Si la descripción contiene palabras de **gastos NO hormiga** (arriendo, matrícula, transferencia, Nequi, cuotas de manejo, transporte, farmacias/IPS específicas, etc.), se etiqueta como `0`.
     - Si el monto absoluto (`valor_abs`) es menor o igual al umbral y la descripción contiene ciertas palabras típicas de gasto pequeño (café, tinto, snack, pan, gaseosa, ara, oxxo, etc.), se etiqueta como `1`.
     - Si el monto es pequeño (≤ threshold) aunque no tenga palabra clave clara, también se tiende a marcar como `1`.
   - El resultado es una nueva columna `es_gasto_hormiga` con valores 0/1.

4. **Ingeniería de características** (`feature_engineering.create_features`)
   - Crea `valor_abs` = `abs(valor)`.
   - Variables de fecha:
     - `dia_semana`, `mes`, `dia_mes`, `es_fin_semana`.
   - Agrega columna `fecha_dia` para agrupar por día.
   - Frecuencia y montos diarios:
     - `freq_gastos_dia`: cantidad de transacciones por día.
     - `gasto_diario_acumulado`: suma de `valor_abs` por día.
   - Variables por comercio (descripción):
     - `freq_por_descripcion`: cuántas veces aparece ese comercio en el historial.
     - `gasto_promedio_descripcion`: promedio de `valor_abs` por comercio.

5. **Análisis visual automático** (`visualizations.*`)
   - `plot_basic_eda`: histogramas, distribución de etiquetas y boxplots (incluido un "zoom" para montos pequeños).
   - `plot_top_comercios`: top de comercios más frecuentes separados en hormiga/no hormiga.
   - `plot_time_series_gasto`: serie temporal del gasto diario hormiga vs no hormiga.
   - `plot_scatter_freq_vs_monto`: dispersión frecuencia vs gasto promedio por comercio.
   - `plot_corr_heatmap`: matriz de correlación de variables numéricas clave.

6. **Entrenamiento del modelo** (`train.train_model`)
   - Define variables numéricas y de texto:
     - Numéricas: `valor_abs`, `dia_semana`, `mes`, `dia_mes`, `es_fin_semana`, `freq_gastos_dia`, `gasto_diario_acumulado`, `freq_por_descripcion`, `gasto_promedio_descripcion`.
     - Texto: `descripcion` (TF‑IDF con bigramas, hasta 3000 features).
   - Separa en:
     - 70% entrenamiento,
     - 15% validación,
     - 15% prueba (test), con `stratify`.
   - Pipeline numérico: `SimpleImputer(strategy="median")` + `StandardScaler`.
   - Pipeline de texto: `TfidfVectorizer(max_features=3000, ngram_range=(1, 2))`.
   - Modelo: `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)`.

7. **Evaluación** (`evaluate.evaluate`)
   - Para VALIDATION y TEST:
     - Imprime `classification_report` (precisión, recall, F1 por clase).
     - Calcula y muestra la matriz de confusión.
     - Si el modelo tiene `predict_proba`, calcula **ROC‑AUC**.
   - Llama a `plot_confusion_and_roc` para guardar las gráficas del modelo.


---

## 7. Resultados de Ejecución y Archivos Generados

Al ejecutar `python main.py` se generan:

- En `reports/` (nombres organizados):
  - **EDA (exploración de datos)**
    - `eda_01_hist_valor_abs.png`
    - `eda_02_label_distribution.png`
    - `eda_03_box_valor_por_label.png`
    - `eda_04_box_valor_por_label_zoom.png`
    - `eda_05_top_comercios.png`
    - `eda_06_serie_gasto_diario.png`
    - `eda_07_scatter_freq_vs_gasto_promedio.png`
    - `eda_08_corr_heatmap.png`
  - **Modelo (evaluación)**
    - `model_VALIDATION_confusion_matrix.png`
    - `model_VALIDATION_roc_curve.png`
    - `model_TEST_confusion_matrix.png`
    - `model_TEST_roc_curve.png`

En la consola se imprimen además los reportes de clasificación (precisión, recall, F1) para los conjuntos de validación y test.

---

## 8. Análisis de Resultados (Guía de lectura)

Algunos puntos a revisar en las gráficas:

- `eda_01_hist_valor_abs.png`:
  - Muestra la distribución de montos. Esperamos ver muchos gastos pequeños y pocos muy grandes.

- `eda_02_label_distribution.png`:
  - Verifica el **balance de clases** (cuántos hormiga vs no hormiga). Si está muy desbalanceado, justifica el uso de `class_weight="balanced"`.

- `eda_03_box_valor_por_label.png` y `eda_04_box_valor_por_label_zoom.png`:
  - Comparan la distribución de montos para 0 y 1.
  - El gráfico "zoom" permite ver si los hormiga tienen montos claramente menores.

- `eda_05_top_comercios.png`:
  - Identifica los comercios más frecuentes y si son mayormente hormiga o no.

- `eda_06_serie_gasto_diario.png`:
  - Muestra la evolución del gasto diario hormiga vs no hormiga.
  - Permite ver períodos donde el gasto hormiga aumenta o disminuye.

- `eda_07_scatter_freq_vs_gasto_promedio.png`:
  - Relación entre **qué tan frecuente** es un comercio y **cuánto se gasta en promedio**.
  - Los gastos hormiga deberían agruparse en la zona de **alta frecuencia y bajo monto**.

- `eda_08_corr_heatmap.png`:
  - Revisa qué variables numéricas se correlacionan más con `es_gasto_hormiga`.

- Gráficas del modelo (`model_*`):
  - Matrices de confusión → muestran aciertos y errores en cada clase.
  - Curvas ROC → miden la capacidad global del modelo para distinguir hormiga de no hormiga (mientras más arriba de la diagonal, mejor).

---

## 9. Features Creadas

Principales variables utilizadas por el modelo:

- **Monto y resumen diario**
  - `valor_abs`: monto absoluto de la transacción.
  - `freq_gastos_dia`: número de transacciones en ese día.
  - `gasto_diario_acumulado`: suma diaria de gastos.

- **Fecha y contexto temporal**
  - `dia_semana`, `mes`, `dia_mes`, `es_fin_semana`.

- **Frecuencia por comercio**
  - `freq_por_descripcion`: cuántas transacciones tiene ese comercio en todo el historial.
  - `gasto_promedio_descripcion`: gasto promedio absoluto para ese comercio.

- **Texto**
  - `descripcion`: vectorizada con TF‑IDF (unigramas y bigramas) para capturar palabras y combinaciones relevantes de los comercios.

---

## 10. Conclusiones (Guía)

- El pipeline permite **automatizar** la detección de gastos hormiga combinando reglas y aprendizaje automático.
- Las reglas de negocio filtran correctamente movimientos que **no** deben considerarse hormiga (transferencias, cuotas, intereses, transporte, etc.).
- Las features de frecuencia por comercio y por día ayudan al modelo a captar el comportamiento de gastos pequeños pero **repetitivos**.
- Las visualizaciones en `reports/` facilitan:
  - Entender en qué categorías se concentran los gastos hormiga.
  - Ver cómo cambian en el tiempo.
  - Evaluar el rendimiento del modelo de forma intuitiva.

Según se ajusten las reglas (`no_hormiga_keywords`, `hormiga_keywords`, `DEFAULT_THRESHOLD`) y el modelo, se pueden refinar aún más los resultados para adaptarlos al estilo de gasto de cada persona.

---


## 11. Trabajo Futuro

Algunas posibles extensiones:

- Afinar aún más las reglas para tipos de comercios específicos.
- Probar otros modelos (por ejemplo, `LogisticRegression`, `XGBoost`) y comparar resultados.
- Ajustar el umbral `DEFAULT_THRESHOLD` según el nivel de gasto de cada usuario.
- Incluir métricas específicas sobre el **impacto total** de los gastos hormiga (porcentaje del gasto total que representan).

