# Deteccion de Gastos Hormiga

Pipeline de Machine Learning hibrido para detectar transacciones tipo "gasto hormiga" combinando:

- Analisis no supervisado (clustering)
- Reevaluacion de etiquetas por consenso
- Modelado supervisado
- Generacion automatica de reportes y visualizaciones


## Objetivo

Mejorar la calidad de la clasificacion de gastos hormiga corrigiendo posible ruido en la etiqueta original (`ETIQUETA`) mediante consenso de multiples algoritmos de clustering, y luego comparar el desempeno de modelos supervisados con etiquetas originales vs reevaluadas.

## Estructura esperada

Esta carpeta debe contener al menos:

- `Dataset_Hormiga_Binario.csv`
- `deteccion_gastos_hormiga.py`
- `Outputs/` (se crea automaticamente si no existe)

## Dataset de entrada

El archivo CSV debe incluir estas columnas:

- `FECHA`
- `DESCRIPCION`
- `VALOR`
- `ETIQUETA` (0 = no hormiga, 1 = hormiga)

## Flujo del pipeline

1. Preprocesamiento y feature engineering
- Parsing de fechas y variables temporales (`DIA`, `MES`, `DIA_SEMANA`, etc.)
- Limpieza de texto en descripcion
- Vectorizacion TF-IDF de `DESCRIPCION`
- Escalado de `VALOR`
- Features adicionales (`LOG_VALOR`, quintiles, frecuencia de descripcion)

2. Clustering (no supervisado)
- K-Means
- Fuzzy C-Means
- Subtractive Clustering
- DBSCAN
- Agglomerative Clustering
- Reduccion de dimensionalidad con PCA y UMAP

3. Reevaluacion de etiquetas por consenso
- Voto por mayoria entre algoritmos de clustering validos
- Generacion de `ETIQUETA_REEVALUADA`

4. Modelado supervisado
- Arbol de Decision
- Regresion Logistica
- Regresion Lineal con umbral 0.5
- Comparacion en dos escenarios:
  - Etiquetas originales
  - Etiquetas reevaluadas

5. Salidas
- Graficas de clustering, matrices de confusion, importancia de variables y arbol
- Reportes CSV/TXT
- Dataset final con etiquetas reevaluadas

## Requisitos

Instala dependencias con:

```bash
pip install -r requirements.txt
```

Si falta algun paquete, instala manualmente:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-fuzzy umap-learn plotly graphviz
```

## Ejecucion

Importante: el script usa la ruta relativa:

- `Informe 2/Dataset_Hormiga_Binario.csv`

Por esto, debes ejecutarlo desde la carpeta raiz del repositorio `Paula_Llanos_Lopez`.

Ejemplo en PowerShell (desde `Github/Paula_Llanos_Lopez`):

```powershell
python "Informe 2/deteccion_gastos_hormiga.py"
```

## Archivos generados

Se guardan en `Outputs/`:

- `01_pca_original.png`
- `02_clustering_pca.png`
- `03_clustering_umap.png`
- `04_confusion_matrix_original.png`
- `04_confusion_matrix_reevaluada.png`
- `05_feature_importance.png`
- `06_arbol_decision_estructura.png`
- `07_arbol_decision_reglas.txt`
- `08_metricas_comparativas.csv`
- `09_analisis_impacto.txt`
- `10_reporte_general.txt`
- `dataset_con_reevaluadas.csv`

