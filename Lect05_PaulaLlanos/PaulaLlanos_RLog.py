import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,f1_score,precision_score
from sklearn.metrics import recall_score,confusion_matrix,ConfusionMatrixDisplay


# Carpeta donde se guardarán las gráficas
output_dir = "Graficos"
os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv("Titanic-Dataset.csv")

# Usamos más variables (numéricas y categóricas)
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

df.head()

# Definimos las variables independientes (X) y la variable dependiente (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# 2. Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Definimos preprocesamiento y pipeline
numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Sex', 'Embarked']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline_log = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# 4. Definir distribución de parámetros (mejor búsqueda)
param_dist_log = {
    'model__C': loguniform(1e-3, 1e2)
}

# 5. Aplicar Búsqueda Aleatoria + Cross Validation
random_log = RandomizedSearchCV(
    pipeline_log,
    param_distributions=param_dist_log,
    n_iter=100,
    cv=5,
    scoring='f1',
    random_state=42
)


# 6. Entrenar el modelo
random_log.fit(X_train, y_train)

# 7. Obtener mejores parámetros
print("===== ENTRENAMIENTO COMPLETADO =====")
print("Mejores parámetros:", random_log.best_params_)

# 8. Evaluar el modelo

# Predicciones
y_train_pred = random_log.predict(X_train)
y_test_pred = random_log.predict(X_test)

# Métricas en entrenamiento
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Métricas en prueba
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("===== EVALUACIÓN DEL MODELO =====")

print("\nMÉTRICAS DE ENTRENAMIENTO:")
print(f"  Accuracy:  {train_accuracy:.4f}")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1 Score:  {train_f1:.4f}")

print("\nMÉTRICAS DE PRUEBA:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

# 9. Graficar modelo predicho
y_prob = random_log.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8,6))
plt.scatter(X_test['Fare'], y_prob, c=y_test, cmap='bwr', alpha=0.7)
plt.xlabel("Fare")
plt.ylabel("Probabilidad de Sobrevivir")
plt.title("Probabilidad Predicha (Regresión Logística)")
plt.colorbar(label="Clase Real")
plt.savefig(os.path.join(output_dir, "RLog_probabilidad_sobrevivir.png"))

# 10. Matriz de Confusión
print("===== MATRIZ DE CONFUSIÓN =====")

cm = confusion_matrix(y_test, y_test_pred)
print("\nMatriz de confusión (datos de prueba):")
print(cm)

tn, fp, fn, tp = cm.ravel()
print("\nInterpretación:")
print(f"  Verdaderos Negativos (TN): {tn} - Predijo no sobrevivió correctamente")
print(f"  Falsos Positivos (FP):     {fp} - Predijo sobrevivió, pero no fue así")
print(f"  Falsos Negativos (FN):     {fn} - Predijo no sobrevivió, pero sí fue así")
print(f"  Verdaderos Positivos (TP): {tp} - Predijo sobrevivió correctamente")

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión")
plt.savefig(os.path.join(output_dir, "RLog_matriz_confusion.png"))

# 11. Importancia de características
print("===== IMPORTANCIA DE CARACTERÍSTICAS =====")

best_model = random_log.best_estimator_
log_reg = best_model.named_steps['model']
preprocess = best_model.named_steps['preprocess']

feature_names = preprocess.get_feature_names_out()
coefficients = log_reg.coef_[0]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nCoeficientes del modelo (ordenados por importancia):")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in importance_df['Coefficient']]
plt.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors)
plt.xlabel('Coeficiente')
plt.title('Importancia de Características - Regresión Logística')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'RLog_importancia_caracteristicas.png'))
