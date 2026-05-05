# Fundamentos de Aprendizaje Automático

## Estructura del repositorio

- `Informe1_PaulaLlanos/`
	- `data/`: contiene el dataset `Dataset_Hormiga.csv` usado en el informe.
	- `src/`: scripts principales del proyecto (carga de datos, preprocesamiento, ingeniería de características y visualizaciones).
	- `reports/`: espacio para gráficos y resultados generados por el informe.
	- `README.md`: descripción específica del Informe 1.

- `Informe 2_PaulaLlanos/`
	- `Dataset_Hormiga_Binario.csv`: dataset binario usado en el informe.
	- `deteccion_gastos_hormiga.py`: script principal de análisis y modelado.
	- `Outputs/`: resultados y salidas generadas.
	- `README.md`: descripción específica del Informe 2.

- `Lect02_PaulaLlanos/`
	- `llanos_paula_peliculas.py`: ejercicios de la clase 2, trabajando con un dataset de películas.

- `Lect04_PaulaLlanos/`
	- `Semana4_PaulaLlanosLopez.py`: análisis y preparación de datos del dataset del Titanic (`Titanic-Dataset.csv`).

- `Lect05_PaulaLlanos/`
	- `PaulaLlanos_RL.py`: modelo de **regresión lineal** (Ridge y Lasso) para predecir la tarifa (`Fare`) en función de la edad (`Age`) usando el dataset del Titanic.
	- `PaulaLlanos_RLog.py`: modelo de **regresión logística** para predecir la supervivencia (`Survived`) de los pasajeros del Titanic.
	- `Titanic-Dataset.csv`: dataset del Titanic usado en las clases.
	- `Graficos/`: carpeta donde se guardan automáticamente las gráficas generadas.

- `Lect06_PaulaLlanos/`
	- `src/PaulaLl_Modelos.py`: modelos de clasificación (**Árbol de Decisión** y **Random Forest**) aplicados al dataset de la Hormiga.
	- `src/Dataset_Hormiga.csv`: dataset original de la Hormiga.
	- `src/Dataset_Hormiga_Limpio.csv`: dataset preprocesado y limpio.
	- `Reportes/`: carpeta donde se guardan automáticamente los reportes y gráficas generadas.

- `Lect08_PaulaLlanos/`
	- `PaulaLl_UdeA.py`: pipeline completo **FIRE-UdeA** — EDA, preprocesamiento y modelos de clasificación (**Árbol de Decisión**, **Random Forest** y **Gradient Boosting**) para estimar riesgo de tensión financiera universitaria.
	- `dataset_sintetico_FIRE_UdeA_realista.csv`: dataset sintético realista con variables financieras de unidades académicas.
	- `graficas/`: carpeta con las 14 gráficas generadas automáticamente (EDA, métricas, matrices de confusión, curvas ROC, importancia de variables y árbol de decisión).

- `Lect09_PaulaLlanos/`
	- `Sintetico.py`: análisis y modelado con dataset sintético FIRE-UdeA.
	- `Realista.py`: análisis y modelado con dataset sintético realista FIRE-UdeA.
	- `dataset_sintetico_FIRE_UdeA.csv`: dataset sintético base.
	- `dataset_sintetico_FIRE_UdeA_realista.csv`: dataset sintético realista.
	- `Graficas/`: carpeta con las gráficas generadas.

- `Lect10_PaulaLlanos/`
	- `analisis_completo.py`: análisis completo y modelado del dataset FIRE-UdeA.
	- `eliminar_labels.py`: script de limpieza para remover etiquetas.
	- `dataset_sintetico_FIRE_UdeA.csv`: dataset sintético base.
	- `dataset_sintetico_FIRE_UdeA_realista.csv`: dataset sintético realista.
	- `dataset_sintetico_FIRE_UdeA_sin_label.csv`: dataset sintético sin etiqueta.
	- `dataset_sintetico_FIRE_UdeA_realista_sin_label.csv`: dataset sintético realista sin etiqueta.
	- `Graficas/`: carpeta con las gráficas generadas.

## Requisitos

Se recomienda usar un entorno virtual de Python (por ejemplo `ml_env`) con las librerías indicadas por el curso, incluyendo al menos:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

Si cuentas con un archivo `requirements.txt` en el proyecto del curso, puedes instalar las dependencias con:

```bash
pip install -r requirements.txt
```
