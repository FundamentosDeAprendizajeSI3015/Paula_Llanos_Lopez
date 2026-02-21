"""
Análisis del dataset de películas OMDB.
Incluye: obtención de datos, exploración, visualización, ingeniería de características,
preprocesamiento, partición, entrenamiento de múltiples modelos, evaluación,
validación cruzada, ajuste de hiperparámetros y guardado del mejor modelo.
"""
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from joblib import dump

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

API_KEY = "ACÁ VA LA API KEY, no la puse por seguridad..."
URL = "http://www.omdbapi.com/"

# ==========================================
# 1) Obtención de datos desde OMDB API
# ==========================================
print("=== Obteniendo datos de películas ===")

movies = [
    "Inception",
    "Interstellar",
    "Titanic",
    "Avatar",
    "The Matrix",
    "Gladiator",
    "The Godfather",
    "Joker",
    "Forrest Gump",
    "The Dark Knight",
    "Pulp Fiction",
    "Fight Club",
    "The Shawshank Redemption",
    "The Lord of the Rings: The Return of the King",
    "Star Wars",
    "Avengers: Endgame",
    "Jurassic Park",
    "The Lion King",
    "Toy Story",
    "Finding Nemo",
    "The Avengers",
    "Iron Man",
    "Spider-Man",
    "Batman Begins",
    "The Departed",
    "Catch Me If You Can",
    "The Prestige",
    "Memento",
    "Shutter Island",
    "Django Unchained"
]

data = []
for movie in movies:
    params = {"apikey": API_KEY, "t": movie}
    try:
        response = requests.get(URL, params=params).json()
        if response["Response"] == "True":
            data.append({
                "title": response["Title"],
                "rating": float(response["imdbRating"]),
                "votes": int(response["imdbVotes"].replace(",", "")),
                "runtime": int(response["Runtime"].split()[0]) if response["Runtime"] != "N/A" else 0,
                "year": int(response["Year"].split("–")[0]) if "–" not in response["Year"] else int(response["Year"])
            })
    except Exception as e:
        print(f"Error con {movie}: {e}")
        continue

df = pd.DataFrame(data)
print(f"\nDatos obtenidos: {len(df)} películas")
print(df.head())

# Crear carpeta de salidas
os.makedirs("outputs_movies", exist_ok=True)

# ==========================================
# 2) Exploración inicial
# ==========================================
print("\n=== Dimensiones ===")
print(df.shape)

print("\n=== Descripción estadística ===")
print(df.describe())

print("\n=== Información del dataset ===")
print(df.info())

# Definir variable objetivo: películas exitosas (rating >= 8.0)
# Ajustamos el umbral para tener mejor balance de clases
df["success"] = (df["rating"] >= 8.0).astype(int)
print("\n=== Distribución de clases ===")
print(df["success"].value_counts())
print(f"Proporción de éxito: {df['success'].mean():.2%}")

# Verificar que hay suficientes muestras de cada clase
min_class_count = df["success"].value_counts().min()
if min_class_count < 2:
    print(f"\nADVERTENCIA: Clase minoritaria tiene solo {min_class_count} muestra(s).")
    print("Ajustando umbral de éxito para mejor balance...")
    # Probar diferentes umbrales
    for threshold in [7.8, 7.5, 7.0, 6.5]:
        df["success"] = (df["rating"] >= threshold).astype(int)
        min_count = df["success"].value_counts().min()
        if min_count >= 2:
            print(f"Usando umbral de rating >= {threshold}")
            print(df["success"].value_counts())
            break

# ==========================================
# 3) Visualizaciones exploratorias
# ==========================================
# Histogramas
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
df["rating"].hist(bins=15, ax=axes[0,0], color='skyblue', edgecolor='black')
axes[0,0].set_title("Distribución de Rating")
axes[0,0].set_xlabel("Rating")

df["votes"].hist(bins=15, ax=axes[0,1], color='salmon', edgecolor='black')
axes[0,1].set_title("Distribución de Votos")
axes[0,1].set_xlabel("Votos")

df["runtime"].hist(bins=15, ax=axes[1,0], color='lightgreen', edgecolor='black')
axes[1,0].set_title("Distribución de Duración")
axes[1,0].set_xlabel("Minutos")

df["year"].hist(bins=15, ax=axes[1,1], color='plum', edgecolor='black')
axes[1,1].set_title("Distribución de Año")
axes[1,1].set_xlabel("Año")

plt.tight_layout()
plt.savefig("outputs_movies/histograms.png", dpi=150)
plt.close()

# Boxplots por clase
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(['votes', 'runtime', 'year']):
    sns.boxplot(data=df, x='success', y=col, ax=axes[i], palette='Set2')
    axes[i].set_xticklabels(['No exitosa', 'Exitosa'])
    axes[i].set_title(f'{col.capitalize()} por clase')
plt.tight_layout()
plt.savefig("outputs_movies/boxplots_by_class.png", dpi=150)
plt.close()

# Matriz de correlación
X_features = df[["votes", "runtime", "year"]]
corr = X_features.corr()
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Matriz de correlación")
plt.tight_layout()
plt.savefig("outputs_movies/correlation_heatmap.png", dpi=150)
plt.close()

# Scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
scatter_pairs = [('votes', 'rating'), ('runtime', 'rating'), ('year', 'rating')]
for i, (x_col, y_col) in enumerate(scatter_pairs):
    for success_val in [0, 1]:
        mask = df['success'] == success_val
        axes[i].scatter(df[mask][x_col], df[mask][y_col], 
                       label=['No exitosa', 'Exitosa'][success_val],
                       alpha=0.6)
    axes[i].set_xlabel(x_col.capitalize())
    axes[i].set_ylabel(y_col.capitalize())
    axes[i].set_title(f'{y_col} vs {x_col}')
    axes[i].legend()
plt.tight_layout()
plt.savefig("outputs_movies/scatter_plots.png", dpi=150)
plt.close()

# ==========================================
# 4) Ingeniería de características
# ==========================================
X_feat = df[["votes", "runtime", "year"]].copy()

# Características adicionales
X_feat["votes_per_min"] = X_feat["votes"] / (X_feat["runtime"] + 1)
X_feat["votes_log"] = np.log1p(X_feat["votes"])
X_feat["age"] = 2026 - X_feat["year"]
X_feat["runtime_squared"] = X_feat["runtime"] ** 2

print("\n=== Características creadas ===")
print(X_feat.columns.tolist())

# PCA para visualización
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X_feat))
plt.figure(figsize=(8, 6))
for success_val in [0, 1]:
    mask = df['success'] == success_val
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               label=['No exitosa', 'Exitosa'][success_val],
               alpha=0.7, s=80)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.title("PCA - Películas")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs_movies/pca_2d.png", dpi=150)
plt.close()

# ==========================================
# 5) Preparación de datos
# ==========================================
y = df["success"]

# Verificar si podemos usar estratificación
min_class_samples = y.value_counts().min()
use_stratify = y if min_class_samples >= 2 else None

if use_stratify is None:
    print("\nNo se puede usar estratificación (muy pocas muestras por clase)")

X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.25, stratify=use_stratify, random_state=RANDOM_STATE
)

print(f"\n=== Partición de datos ===")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Distribución train: {y_train.value_counts().to_dict()}")
print(f"Distribución test: {y_test.value_counts().to_dict()}")

# Validación cruzada ajustada según el tamaño del dataset
n_splits = min(5, min_class_samples) if min_class_samples >= 2 else 3
if n_splits < 5:
    print(f"\nUsando {n_splits}-fold CV debido a tamaño limitado del dataset")

# ==========================================
# 6) Modelos y pipelines
# ==========================================
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),
    "DecisionTree": Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ]),
    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE))
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ])
}

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

print("\n=== Validación cruzada (accuracy media ± std) ===")
cv_results = {}
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = scores
    print(f"{name:>20}: {scores.mean():.3f} ± {scores.std():.3f}")

# Visualizar resultados CV
plt.figure(figsize=(10, 6))
plt.boxplot([cv_results[name] for name in models.keys()], 
           labels=list(models.keys()), patch_artist=True)
plt.ylabel("Accuracy")
plt.title("Comparación de modelos - Validación Cruzada")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs_movies/cv_comparison.png", dpi=150)
plt.close()

# ==========================================
# 7) Ajuste de hiperparámetros
# ==========================================
print("\n=== Ajuste de hiperparámetros ===")

# Grid Search para RandomForest
param_grid_rf = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 3, 5, 7, 10],
    "clf__min_samples_split": [2, 5, 10]
}
rf_grid = GridSearchCV(models["RandomForest"], param_grid_rf, cv=cv, 
                       scoring="accuracy", n_jobs=-1, verbose=0)
rf_grid.fit(X_train, y_train)
print(f"Mejor RF: {rf_grid.best_params_}, accuracy={rf_grid.best_score_:.3f}")

# Grid Search para SVM
param_grid_svm = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.1, 0.01]
}
svm_grid = GridSearchCV(models["SVM_RBF"], param_grid_svm, cv=cv, 
                        scoring="accuracy", n_jobs=-1, verbose=0)
svm_grid.fit(X_train, y_train)
print(f"Mejor SVM: {svm_grid.best_params_}, accuracy={svm_grid.best_score_:.3f}")

# Grid Search para GradientBoosting
param_grid_gb = {
    "clf__n_estimators": [50, 100, 200],
    "clf__learning_rate": [0.01, 0.1, 0.3],
    "clf__max_depth": [3, 5, 7]
}
gb_grid = GridSearchCV(models["GradientBoosting"], param_grid_gb, cv=cv, 
                       scoring="accuracy", n_jobs=-1, verbose=0)
gb_grid.fit(X_train, y_train)
print(f"Mejor GB: {gb_grid.best_params_}, accuracy={gb_grid.best_score_:.3f}")

# Seleccionar el mejor modelo
best_grids = [rf_grid, svm_grid, gb_grid]
best_names = ["RandomForest", "SVM_RBF", "GradientBoosting"]
best_idx = np.argmax([g.best_score_ for g in best_grids])
best_estimator = best_grids[best_idx]
best_model = best_estimator.best_estimator_
best_name = best_names[best_idx]

print(f"\n=== Modelo seleccionado: {best_name} ===")
print(f"Mejores parámetros: {best_estimator.best_params_}")

# ==========================================
# 8) Evaluación en test
# ==========================================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
cm = confusion_matrix(y_test, y_pred)

print("\n=== Resultados en test ===")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-Score: {f1:.3f}")

print("\n=== Reporte de clasificación ===")
print(classification_report(y_test, y_pred, target_names=['No exitosa', 'Exitosa']))

# Matriz de confusión
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['No exitosa', 'Exitosa'],
           yticklabels=['No exitosa', 'Exitosa'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title(f'Matriz de Confusión - {best_name}')
plt.tight_layout()
plt.savefig("outputs_movies/confusion_matrix.png", dpi=150)
plt.close()

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Curva ROC - {best_name}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs_movies/roc_curve.png", dpi=150)
plt.close()

print(f"\nROC-AUC Score: {auc_score:.3f}")

# ==========================================
# 9) Importancia de características
# ==========================================
feature_names = X_feat.columns
feature_importances = None

try:
    clf = best_model.named_steps.get('clf')
    if hasattr(clf, 'feature_importances_'):
        feature_importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        feature_importances = np.abs(clf.coef_[0])
except Exception as e:
    print(f"No se pudo extraer importancias: {e}")

if feature_importances is not None:
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    print("\n=== Importancia de características ===")
    print(imp_df)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x='importance', y='feature', palette='viridis')
    plt.title(f'Importancia de Características - {best_name}')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.savefig("outputs_movies/feature_importances.png", dpi=150)
    plt.close()

# ==========================================
# 10) Guardar modelo y resultados
# ==========================================
dump(best_model, "outputs_movies/best_movie_model.joblib")
print("\n=== Modelo guardado en outputs_movies/best_movie_model.joblib ===")

# Guardar resumen
with open("outputs_movies/summary.txt", "w", encoding="utf-8") as f:
    f.write("=== RESUMEN DEL ANÁLISIS DE PELÍCULAS ===\n\n")
    f.write(f"Modelo seleccionado: {best_name}\n")
    f.write(f"Parámetros: {best_estimator.best_params_}\n\n")
    f.write("=== MÉTRICAS EN TEST ===\n")
    f.write(f"Accuracy: {acc:.3f}\n")
    f.write(f"Precision: {prec:.3f}\n")
    f.write(f"Recall: {rec:.3f}\n")
    f.write(f"F1-Score: {f1:.3f}\n")
    f.write(f"ROC-AUC: {auc_score:.3f}\n\n")
    f.write("=== MATRIZ DE CONFUSIÓN ===\n")
    f.write(str(cm) + "\n\n")
    if feature_importances is not None:
        f.write("=== IMPORTANCIA DE CARACTERÍSTICAS ===\n")
        f.write(str(imp_df.to_string(index=False)))

print("\nAnálisis completo. Todos los gráficos y resultados guardados en 'outputs_movies/'")

# ==========================================
# 11) Función de predicción
# ==========================================
def predict_movie(votes, runtime, year):
    """
    Predice si una película será exitosa basándose en sus características.
    
    Args:
        votes: Número de votos
        runtime: Duración en minutos
        year: Año de estreno
    
    Returns:
        str: "Exitosa" o "No exitosa"
    """
    # Crear características
    votes_per_min = votes / (runtime + 1)
    votes_log = np.log1p(votes)
    age = 2026 - year
    runtime_squared = runtime ** 2
    
    data = np.array([[votes, runtime, year, votes_per_min, votes_log, age, runtime_squared]])
    prediction = best_model.predict(data)[0]
    proba = best_model.predict_proba(data)[0]
    
    result = "Exitosa" if prediction == 1 else "No exitosa"
    confidence = proba[prediction] * 100
    
    return f"{result} (confianza: {confidence:.1f}%)"

# Ejemplo:
print("\n=== Ejemplo de predicción ===")
ejemplo = predict_movie(votes=1000000, runtime=148, year=2020)
print(f"Película con 1M votos, 148 min, año 2020: {ejemplo}")