# ============================================================
# 1) Medidas de tendencia central
# 2) Medidas de dispersión
# 3) Medidas de posición y eliminación de outliers (si es necesario)
# 4) Histogramas (distribución)
# 5) Dispersión entre dos columnas (relación)
# 6) Transformaciones:
#    - One Hot Encoding, Label Encoding, Binary Encoding
#    - Correlación (decidir si eliminar columnas)
#    - Escalamiento (MinMax o Standard)
#    - Transformación logarítmica (si es necesario)
# 7) Conclusiones
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import category_encoders as ce

# ---------------------------
# 0) Cargar datos
# ---------------------------
data = pd.read_csv("Titanic-Dataset.csv")
print("Shape original:", data.shape)
print("\nPrimeras filas:\n", data.head())
print("\nInfo:\n")
print(data.info())

# Columnas numéricas base para EDA (puedes ampliar si quieres)
num_cols = ["Age", "Fare", "SibSp", "Parch"]
cat_cols = ["Pclass", "Sex", "Embarked", "Cabin"]

# ---------------------------
# 1) Medidas de Tendencia Central
# ---------------------------
print("\n==============================")
print("1) Medidas de Tendencia Central")
print("==============================")

print("\nMedia:")
print(data[num_cols].mean(numeric_only=True))

print("\nMediana:")
print(data[num_cols].median(numeric_only=True))

print("\nModa (primera moda por columna):")
# mode() puede devolver varias modas; tomamos la primera fila
print(data[num_cols].mode(numeric_only=True).iloc[0])

# Comentario corto (útil para la entrega)
print("\nComentario:")
print("- Si la media y la mediana difieren mucho, suele indicar asimetría (sesgo) y/o outliers.\n")

# ---------------------------
# 2) Medidas de Dispersión
# ---------------------------
print("\n==============================")
print("2) Medidas de Dispersión")
print("==============================")

print("\nDesviación estándar:")
print(data[num_cols].std(numeric_only=True))

print("\nVarianza:")
print(data[num_cols].var(numeric_only=True))

print("\nRango (max - min):")
print(data[num_cols].max(numeric_only=True) - data[num_cols].min(numeric_only=True))

# ---------------------------
# 3) Medidas de Posición + Outliers (IQR) en Age y Fare
#    Nota: Se crea data_sin_outliers y se usa para el análisis posterior.
# ---------------------------
print("\n==============================")
print("3) Medidas de Posición y Outliers (IQR)")
print("==============================")

# Para outliers, tiene sentido trabajar con valores no nulos
pos_cols = ["Age", "Fare"]
df_pos = data[pos_cols].dropna().copy()

Q1 = df_pos.quantile(0.25)
Q3 = df_pos.quantile(0.75)
IQR = Q3 - Q1

print("\nQ1:\n", Q1)
print("\nQ3:\n", Q3)
print("\nIQR:\n", IQR)

# Regla 1.5*IQR
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Máscara de outliers solo para Age y Fare (en filas con Age y Fare no nulos)
mask_inliers = (df_pos["Age"].between(lower["Age"], upper["Age"])) & (df_pos["Fare"].between(lower["Fare"], upper["Fare"]))

df_pos_inliers = df_pos[mask_inliers].copy()
print(f"\nFilas evaluadas (Age y Fare no nulos): {len(df_pos)}")
print(f"Filas sin outliers (Age y Fare): {len(df_pos_inliers)}")
print(f"Outliers removidos (en esa submuestra): {len(df_pos) - len(df_pos_inliers)}")

# Construimos un dataset limpio para EDA/modelado: removemos filas outlier SOLO si tienen Age y Fare,
# y conservamos filas con nulos en Age o Fare (no se evaluaron en IQR).
# Para evitar “perder” filas con nulos, hacemos esto sobre el dataset original:
data_sin_outliers = data.copy()
# Solo aplicamos filtro donde ambas columnas existen:
mask_eval = data_sin_outliers["Age"].notna() & data_sin_outliers["Fare"].notna()
mask_keep = (~mask_eval) | (
    data_sin_outliers.loc[mask_eval, "Age"].between(lower["Age"], upper["Age"]) &
    data_sin_outliers.loc[mask_eval, "Fare"].between(lower["Fare"], upper["Fare"])
)
data_sin_outliers = data_sin_outliers[mask_keep].copy()

print("\nShape después de remover outliers (solo donde se pudo evaluar):", data_sin_outliers.shape)
print("Comentario: Se recomienda entrenar modelos con y sin outliers y comparar desempeño.\n")

# Usaremos este para el resto del análisis:
df = data_sin_outliers

# ---------------------------
# 4) Histogramas (numéricos + categóricos)
# ---------------------------
print("\n==============================")
print("4) Histogramas (Distribución)")
print("==============================")

# Numéricos en una sola figura
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

for i, col in enumerate(num_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
    axes[i].set_title(f"Histograma de {col}")
plt.tight_layout()
plt.show()

# Categóricos en una sola figura
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.countplot(x="Pclass", data=df, ax=axes[0])
axes[0].set_title("Conteo de Pclass (Clase)")

sns.countplot(x="Sex", data=df, ax=axes[1])
axes[1].set_title("Conteo de Sex (Sexo)")

sns.countplot(x="Embarked", data=df, ax=axes[2])
axes[2].set_title("Conteo de Embarked (Puerto)")

plt.tight_layout()
plt.show()

# ---------------------------
# 5) Gráficos de dispersión (relación entre columnas)
# ---------------------------
print("\n==============================")
print("5) Gráficos de Dispersión")
print("==============================")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(x="SibSp", y="Parch", data=df, ax=axes[0])
axes[0].set_title("Dispersión: SibSp vs Parch")
axes[0].grid(True)

sns.scatterplot(x="Age", y="Fare", data=df, ax=axes[1])
axes[1].set_title("Dispersión: Age vs Fare")
axes[1].grid(True)

plt.tight_layout()
plt.show()

print("Comentario:")
print("- SibSp y Parch pueden mostrar patrones de grupos familiares.")
print("- Age vs Fare suele ser disperso; si Fare es muy sesgado, conviene log.\n")

# ---------------------------
# 6) Transformaciones de columnas
#    - One Hot: Sex, Embarked
#    - Label: Pclass (ordinal: 1 < 2 < 3)
#    - Binary: Cabin (alta cardinalidad + muchos nulos)
# ---------------------------
print("\n==============================")
print("6) Transformaciones (Encoding)")
print("==============================")

# Copia base para transformaciones
df_enc = df.copy()

# One Hot Encoding (Sex, Embarked)
df_onehot = pd.get_dummies(df_enc, columns=["Sex", "Embarked"], drop_first=False)
print("One Hot aplicado a: Sex, Embarked")

# Label Encoding para Pclass (es ordinal). Nota: también podrías dejarla tal cual (ya es numérica),
# pero lo mostramos para cumplir el requisito explícito.
le = LabelEncoder()
df_onehot["Pclass_encoded"] = le.fit_transform(df_onehot["Pclass"])
print("Label Encoding aplicado a: Pclass -> Pclass_encoded (Pclass es ordinal)")

# Binary Encoding para Cabin (rellenamos nulos primero)
df_onehot["Cabin"] = df_onehot["Cabin"].fillna("Unknown")
bin_encoder = ce.BinaryEncoder(cols=["Cabin"])
df_binary = bin_encoder.fit_transform(df_onehot)
print("Binary Encoding aplicado a: Cabin (nulos -> 'Unknown')")

print("\nShape tras encoding:", df_binary.shape)

# ---------------------------
# 7) Correlación y decisión de eliminación de columnas
# ---------------------------
print("\n==============================")
print("7) Correlación y decisión de features")
print("==============================")

# Para correlación, tomamos numéricas disponibles (incluye Pclass_encoded si está)
# Nota: 'Fare_log' se crea más abajo, luego puedes recalcular si quieres.
corr_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass_encoded"]
corr_df = df_binary[corr_cols].copy()

# Manejo mínimo de nulos para correlación: la correlación requiere pares válidos
corr = corr_df.corr(numeric_only=True)
print("\nMatriz de correlación:\n", corr)

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Mapa de calor (correlación) - variables numéricas")
plt.tight_layout()
plt.show()

# Regla simple para sugerir eliminación: correlaciones muy altas (>|0.90|)
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        val = corr.iloc[i, j]
        if pd.notna(val) and abs(val) >= 0.90:
            high_corr_pairs.append((corr.columns[i], corr.columns[j], float(val)))

if len(high_corr_pairs) == 0:
    print("\nDecisión:")
    print("- No se observan correlaciones extremadamente altas (>|0.90|) entre estas variables.")
    print("- No es necesario eliminar columnas por multicolinealidad fuerte en este subconjunto.\n")
else:
    print("\nDecisión (posibles columnas redundantes):")
    for a, b, v in high_corr_pairs:
        print(f"- {a} vs {b}: corr={v:.3f}")
    print("Sugerencia: considerar eliminar una de cada par altamente correlacionado.\n")

# ---------------------------
# 8) Escalamiento (MinMax y Standard)
#    Importante: escaladores NO aceptan NaN -> imputamos mínimo.
# ---------------------------
print("\n==============================")
print("8) Escalamiento (MinMax / Standard)")
print("==============================")

df_scaled = df_binary.copy()

# Imputación simple SOLO para escalar (no necesariamente final de proyecto):
# - Age: mediana (robusto a outliers)
# - Fare: mediana
for col in ["Age", "Fare"]:
    if col in df_scaled.columns:
        df_scaled[col] = df_scaled[col].fillna(df_scaled[col].median())

scaler_minmax = MinMaxScaler()
scaler_std = StandardScaler()

df_scaled["Fare_minmax"] = scaler_minmax.fit_transform(df_scaled[["Fare"]])
df_scaled["Fare_std"] = scaler_std.fit_transform(df_scaled[["Fare"]])

print("Escalamiento aplicado a Fare -> Fare_minmax y Fare_std\n")

# ---------------------------
# 9) Transformación logarítmica (Fare)
# ---------------------------
print("\n==============================")
print("9) Transformación logarítmica")
print("==============================")

# log1p es estándar: log(1 + x) (evita problema con 0)
df_scaled["Fare_log"] = np.log1p(df_scaled["Fare"])

plt.figure(figsize=(7, 4))
sns.histplot(df_scaled["Fare_log"], kde=True)
plt.title("Histograma de Fare_log = log(1 + Fare)")
plt.tight_layout()
plt.show()

print("Comentario:")
print("- Fare suele estar sesgado a la derecha; log(1+Fare) reduce ese sesgo.\n")

# ---------------------------
# 10) Conclusiones (más específicas)
# ---------------------------
print("\n==============================")
print("10) Conclusiones")
print("==============================")

age_mean = df["Age"].mean()
age_median = df["Age"].median()
fare_mean = df["Fare"].mean()
fare_median = df["Fare"].median()

conclusiones = f"""
Conclusiones (basadas en el EDA):
1) Tendencia central:
   - Age: media ≈ {age_mean:.2f} y mediana ≈ {age_median:.2f}. Si difieren, sugiere asimetría y/o valores extremos.
   - Fare: media ≈ {fare_mean:.2f} y mediana ≈ {fare_median:.2f}. La diferencia suele indicar sesgo por tarifas altas.

2) Dispersión:
   - Fare presenta alta dispersión (rango y desviación). Esto sugiere presencia de valores extremos e inestabilidad para algunos modelos.

3) Outliers:
   - Se detectaron outliers en Age y Fare usando IQR (1.5*IQR).
   - Se generó un dataset sin outliers (cuando se pudo evaluar Age y Fare) para analizar el efecto en distribuciones y modelos.

4) Distribuciones (histogramas):
   - La mayoría de pasajeros paga tarifas relativamente bajas (Fare sesgado).
   - Age suele concentrarse en adultos jóvenes, pero puede contener nulos y algunos extremos.

5) Relaciones (dispersión):
   - SibSp vs Parch puede reflejar grupos familiares (valores discretos).
   - Age vs Fare no siempre muestra relación lineal clara; Fare tiende a tener “colas” por valores altos.

6) Transformaciones:
   - One Hot Encoding: útil para variables nominales (Sex, Embarked).
   - Label Encoding: se mostró para Pclass (ordinal), aunque Pclass ya codifica orden.
   - Binary Encoding: útil para Cabin por alta cardinalidad y muchos valores faltantes.
   - Escalamiento (MinMax/Standard): importante para modelos sensibles a escala.
   - Log (Fare): reduce sesgo y puede mejorar modelos lineales o basados en distancia.

7) Correlación:
   - No se observan correlaciones extremadamente altas (>|0.90|) entre las variables numéricas usadas,
     por lo que no es necesario eliminar columnas por multicolinealidad fuerte en este subconjunto.

"""

print(conclusiones)

