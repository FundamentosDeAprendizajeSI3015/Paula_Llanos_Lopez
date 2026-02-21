import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Carpeta donde se guardarán las gráficas
output_dir = "Graficos"
os.makedirs(output_dir, exist_ok=True)

# Cargar dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Limpieza de datos
df = df[['Age', 'Fare']]

# Eliminar nulos
df = df.dropna()

# 1. Definimos las variables independientes (X) y la variable dependiente (y)
X = df[['Age']]
y = df['Fare']

# 2. Dividir el dataset en conjunto de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Entrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)

# 3. Graficar entrenamiento y prueba
plt.figure(figsize=(8,6))

# Entrenamiento
plt.scatter(X_train, y_train, color='blue', label='Entrenamiento')

# Prueba
plt.scatter(X_test, y_test, color='red', label='Prueba')

plt.xlabel("Edad (Age)")
plt.ylabel("Tarifa (Fare)")
plt.title("Conjunto de Entrenamiento vs Prueba")
plt.legend()
plt.savefig(os.path.join(output_dir, "RL_entrenamiento_vs_prueba.png"))

# 4. Crear pipeline para Ridge y Lasso
pipeline_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

pipeline_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso())
])

# 5. Definir distribuciones de parámetros
param_dist_ridge = {
    'model__alpha': uniform(0.01, 10)
}

param_dist_lasso = {
    'model__alpha': uniform(0.01, 10)
}

# 6. Aplicar Búsqueda Aleatoria + Cross Validation
random_ridge = RandomizedSearchCV(
    pipeline_ridge,
    param_distributions=param_dist_ridge,
    n_iter=50,
    cv=5,
    scoring='r2',
    random_state=42
)

random_lasso = RandomizedSearchCV(
    pipeline_lasso,
    param_distributions=param_dist_lasso,
    n_iter=50,
    cv=5,
    scoring='r2',
    random_state=42
)

# 7. Entrenar los modelos
random_ridge.fit(X_train, y_train)
random_lasso.fit(X_train, y_train)

# 8. Obtener mejores parámetros
print("===== ENTRENAMIENTO COMPLETADO =====")
print("Mejor alpha Ridge:", random_ridge.best_params_)
print("Mejor alpha Lasso:", random_lasso.best_params_)

# 9. Evaluar en conjunto de prueba

# Predicciones
y_train_pred_ridge = random_ridge.predict(X_train)
y_train_pred_lasso = random_lasso.predict(X_train)

y_pred_ridge = random_ridge.predict(X_test)
y_pred_lasso = random_lasso.predict(X_test)

print("===== EVALUACIÓN DEL MODELO =====")

# Calcular R² (train y test)
r2_ridge_train = r2_score(y_train, y_train_pred_ridge)
r2_lasso_train = r2_score(y_train, y_train_pred_lasso)

r2_ridge = r2_score(y_test, y_pred_ridge)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\nR2 Ridge:")
print(f"  Train: {r2_ridge_train:.4f}")
print(f"  Test : {r2_ridge:.4f}")

print("\nR2 Lasso:")
print(f"  Train: {r2_lasso_train:.4f}")
print(f"  Test : {r2_lasso:.4f}")

# Calcular MAE (train y test)
mae_ridge_train = mean_absolute_error(y_train, y_train_pred_ridge)
mae_lasso_train = mean_absolute_error(y_train, y_train_pred_lasso)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print("\nMAE Ridge:")
print(f"  Train: {mae_ridge_train:.4f}")
print(f"  Test : {mae_ridge:.4f}")

print("\nMAE Lasso:")
print(f"  Train: {mae_lasso_train:.4f}")
print(f"  Test : {mae_lasso:.4f}")

# Calcular RMSE (solo test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

print("\nRMSE (Test):")
print(f"  Ridge: {rmse_ridge:.4f}")
print(f"  Lasso: {rmse_lasso:.4f}")

# 10. Graficar modelo predicho (Ridge y Lasso)

# Ordenar para graficar línea continua
sorted_idx = X_test['Age'].argsort()

X_test_sorted = X_test.iloc[sorted_idx]
y_test_sorted = y_test.iloc[sorted_idx]

y_pred_ridge_sorted = y_pred_ridge[sorted_idx]
y_pred_lasso_sorted = y_pred_lasso[sorted_idx]

# Grafica Ridge
plt.figure(figsize=(8,6))

plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test_sorted, y_pred_ridge_sorted, color='red', linewidth=2, label='Ridge')

plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Modelo Ridge")
plt.legend()
plt.savefig(os.path.join(output_dir, "RL_modelo_ridge.png"))

# Grafica Lasso
plt.figure(figsize=(8,6))

plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test_sorted, y_pred_lasso_sorted, color='green', linewidth=2, label='Lasso')

plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Modelo Lasso")
plt.legend()
plt.savefig(os.path.join(output_dir, "RL_modelo_lasso.png"))
