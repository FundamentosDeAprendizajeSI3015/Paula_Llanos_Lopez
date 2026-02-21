# Fundamentos de Aprendizaje Automático

## Estructura del repositorio

- `Informe1_PaulaLlanos/`
	- `data/`: contiene el dataset `Dataset_Hormiga.csv` usado en el informe.
	- `src/`: scripts principales del proyecto (carga de datos, preprocesamiento, ingeniería de características y visualizaciones).
	- `reports/`: espacio para gráficos y resultados generados por el informe.
	- `README.md`: descripción específica del Informe 1.

- `Lect02_PaulaLlanos/`
	- `llanos_paula_peliculas.py`: ejercicios de la clase 2, trabajando con un dataset de películas.

- `Lect04_PaulaLlanos/`
	- `Semana4_PaulaLlanosLopez.py`: análisis y preparación de datos del dataset del Titanic (`Titanic-Dataset.csv`).

- `Lect05_PaulaLlanos/`
	- `PaulaLlanos_RL.py`: modelo de **regresión lineal** (Ridge y Lasso) para predecir la tarifa (`Fare`) en función de la edad (`Age`) usando el dataset del Titanic.
	- `PaulaLlanos_RLog.py`: modelo de **regresión logística** para predecir la supervivencia (`Survived`) de los pasajeros del Titanic.
	- `Titanic-Dataset.csv`: dataset del Titanic usado en las clases.
	- `Graficos/`: carpeta donde se guardan automáticamente las gráficas generadas.

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
