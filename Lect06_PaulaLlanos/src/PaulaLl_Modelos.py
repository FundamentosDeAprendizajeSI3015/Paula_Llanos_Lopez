"""
Lect06 - Árboles de Decisión / Ensemble Methods
Clasificación binaria: ¿Es un gasto hormiga o no?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

import os

# ─────────────────────────────────────────────────────────────
# 1. CARGA Y LIMPIEZA DE DATOS
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.join(BASE_DIR, "Dataset_Hormiga.csv")

df = pd.read_csv(ruta_csv, sep=";", encoding="utf-8")
print("═" * 55)
print("  DATASET ORIGINAL")
print("═" * 55)
print(df.shape)
print(df.head(5))

# Excluir abonos de intereses, ingresos y movimientos que no
# representan un gasto real del bolsillo.
EXCLUIR = [
    "ABONO INTERESES AHORROS",
    "AJUSTE INTERES AHORROS DB",
    "AJUSTE COMPRA INTL APPLE.COM",
    "CONSIGNACION CORRESPONSAL CB",
    "DEV CUOTA MANEJO TARJ DEB",
    "Recarga de Saldo",
    "TRANSFERENCIA CTA SUC VIRTUAL",
    "TRANSFERENCIA DESDE NEQUI",
]

# Quedarse solo con egresos (VALOR negativo) que no estén en EXCLUIR
df_gastos = df[~df["DESCRIPCION"].isin(EXCLUIR)].copy()
df_gastos = df_gastos[df_gastos["VALOR"] < 0].copy()

# Convertir el valor a positivo
df_gastos["VALOR"] = df_gastos["VALOR"].abs()

print(f"\nDespués de limpiar (solo gastos): {df_gastos.shape[0]} filas × {df_gastos.shape[1]} columnas")

# ─────────────────────────────────────────────────────────────
# 2. ETIQUETAR – variable objetivo binaria (y)
# ─────────────────────────────────────────────────────────────
# HORMIGA  → gasto pequeño (< 15 000 COP) que NO sea almuerzo,
#            salud, transporte, servicio ni transferencia.
# NO_HORMIGA → todo lo demás.

UMBRAL_HORMIGA = 15_000   # pesos colombianos

# ── Categorías SIEMPRE NO_HORMIGA (sin importar el monto) ────

# Almuerzos / restaurantes
ALMUERZOS = [
    "HOME FOOD", "BIGOS", "FRISBY", "ALDEA NIKK", "TACO FACTO",
    "CREPES Y W", "MCDONALD", "PRESTO", "MMO SANTA", "FIRE HOUSE",
    "EC EAFIT", "DELI OVIED", "RESTAURANTE", "TATIANA ANDREA",
]

# Salud
SALUD = [
    "FARMACIA", "FARMATODO", "BIOSALUD", "COMSOCIAL",
    "VETERINARIO", "VET ",
]

# Servicios / suscripciones
SERVICIOS = [
    "CUOTA MANEJO", "CLARO", "APPLE", "COMISION",
    "UNIVERSIDAD EAF", "PAGO DE PROV",
]

# Transporte
TRANSPORTE = ["CIVICA", "RETIRO", "CAJERO"]

# Transferencias a otras personas / billeteras
TRANSFERENCIAS = ["TRANSF QR", "TRANSFERENCIA A", "TRANSFERENCIAS A", "NEQUI"]

# Supermercados grandes
SUPERMERCADO_GRANDE = ["EXITO", "JUMBO", "METRO FATE"]

# Bodeguitas / tiendas de barrio → SIEMPRE HORMIGA (sin límite de monto)
BODEGUITAS = ["INVERSIONE", "HYM"]

# ─────────────────────────────────────────────────────────────
def etiquetar(row) -> str:
    d   = str(row["DESCRIPCION"]).upper().strip()
    val = row["VALOR"]   # ya es positivo

    # Bodeguitas: siempre hormiga independiente del monto
    if any(p in d for p in BODEGUITAS):     return "HORMIGA"

    if any(p in d for p in ALMUERZOS):      return "NO_HORMIGA"
    if any(p in d for p in SALUD):          return "NO_HORMIGA"
    if any(p in d for p in SERVICIOS):      return "NO_HORMIGA"
    if any(p in d for p in TRANSPORTE):     return "NO_HORMIGA"
    if any(p in d for p in TRANSFERENCIAS): return "NO_HORMIGA"
    if any(p in d for p in SUPERMERCADO_GRANDE): return "NO_HORMIGA"

    # Regla general: pequeño y cotidiano → HORMIGA
    return "HORMIGA" if val < UMBRAL_HORMIGA else "NO_HORMIGA"

df_gastos["ETIQUETA"] = df_gastos.apply(etiquetar, axis=1)

print("\nDistribución de etiquetas (y):")
print(df_gastos["ETIQUETA"].value_counts())
print(df_gastos[["DESCRIPCION", "VALOR", "ETIQUETA"]].head(15).to_string(index=False))

# Exportar dataset limpio y etiquetado
ruta_limpio = os.path.join(BASE_DIR, "Dataset_Hormiga_Limpio.csv")
df_gastos[["FECHA", "DESCRIPCION", "VALOR", "ETIQUETA"]].to_csv(
    ruta_limpio, sep=";", index=False, encoding="utf-8"
)
print(f"\nDataset limpio guardado en: {ruta_limpio}")

# ─────────────────────────────────────────────────────────────
# 3. INGENIERÍA DE CARACTERÍSTICAS – columnas de entrada (X)
# ─────────────────────────────────────────────────────────────
df_gastos["FECHA"] = pd.to_datetime(df_gastos["FECHA"], format="mixed", dayfirst=True)

df_gastos["DIA"]           = df_gastos["FECHA"].dt.day         # 1–31
df_gastos["MES"]           = df_gastos["FECHA"].dt.month       # 1–12
df_gastos["DIA_SEMANA"]    = df_gastos["FECHA"].dt.dayofweek   # 0=Lun … 6=Dom
df_gastos["ES_FIN_SEMANA"] = (df_gastos["DIA_SEMANA"] >= 5).astype(int)

# FREQ_DESC: cuántas veces aparece ese comercio en todo el historial.
# Un gasto hormiga tiende a ocurrir en lugares que visitas frecuentemente.
df_gastos["FREQ_DESC"] = df_gastos.groupby("DESCRIPCION")["DESCRIPCION"].transform("count")

# X: características numéricas disponibles por cada transacción
# y: etiqueta binaria  →  HORMIGA | NO_HORMIGA
FEATURES = ["VALOR", "DIA", "MES", "DIA_SEMANA", "ES_FIN_SEMANA", "FREQ_DESC"]
TARGET   = "ETIQUETA"

X = df_gastos[FEATURES]
y = df_gastos[TARGET]

print("\n─── Dimensiones del problema ───")
print(f"  Características (X) : {FEATURES}")
print(f"  Objetivo        (y) : {TARGET}  →  {sorted(y.unique())}")
print(f"  Total muestras      : {X.shape[0]}")

# ─────────────────────────────────────────────────────────────
# 4. DIVISIÓN ENTRENAMIENTO / VALIDACIÓN / PRUEBA (60/20/20)
# ─────────────────────────────────────────────────────────────
# Primer split: 60 % entrenamiento, 40 % temp (validación + prueba)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Segundo split: el 40 % temp se divide en 50/50 → 20 % / 20 %
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f"\nEntrenamiento : {X_train.shape[0]} muestras  (60 %)")
print(f"Validación    : {X_val.shape[0]}  muestras  (20 %)")
print(f"Prueba        : {X_test.shape[0]}  muestras  (20 %)")

# ─────────────────────────────────────────────────────────────
# 5. PIPELINES
# ─────────────────────────────────────────────────────────────
pipeline_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline_gb = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

# ─────────────────────────────────────────────────────────────
# 6. ENTRENAMIENTO  (ajuste con X_train, evaluación intermedia con X_val)
# ─────────────────────────────────────────────────────────────
print("\nEntrenando Random Forest …")
pipeline_rf.fit(X_train, y_train)
print(f"  Accuracy en validación (RF) : {pipeline_rf.score(X_val, y_val):.4f}")

print("Entrenando Gradient Boosting …")
pipeline_gb.fit(X_train, y_train)
print(f"  Accuracy en validación (GB) : {pipeline_gb.score(X_val, y_val):.4f}")

print("Entrenamiento completado.")

# ─────────────────────────────────────────────────────────────
# 7. MÉTRICAS DE DESEMPEÑO
# ─────────────────────────────────────────────────────────────
def evaluar_modelo(nombre: str, pipeline, X_ev, y_ev):
    y_pred = pipeline.predict(X_ev)

    acc  = accuracy_score(y_ev, y_pred)
    prec = precision_score(y_ev, y_pred, pos_label="HORMIGA", zero_division=0)
    rec  = recall_score(y_ev, y_pred, pos_label="HORMIGA", zero_division=0)
    f1   = f1_score(y_ev, y_pred, pos_label="HORMIGA", zero_division=0)

    print(f"\n{'═'*55}")
    print(f"  {nombre}")
    print(f"{'═'*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (¿de los que predijo HORMIGA, cuántos lo eran?)")
    print(f"  Recall    : {rec:.4f}  (¿de los HORMIGA reales, cuántos detectó?)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Reporte de clasificación:")
    print(classification_report(y_ev, y_pred, zero_division=0))

    return y_pred, {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}

# ── Evaluación en entrenamiento (¿aprendió bien?)
print("\n══════ CONJUNTO DE ENTRENAMIENTO ══════")
y_pred_rf_train, met_rf_train = evaluar_modelo("RANDOM FOREST   — Entrenamiento", pipeline_rf, X_train, y_train)
y_pred_gb_train, met_gb_train = evaluar_modelo("GRADIENT BOOST  — Entrenamiento", pipeline_gb, X_train, y_train)

# ── Evaluación en validación (¿generaliza?)
print("\n══════ CONJUNTO DE VALIDACIÓN ══════")
y_pred_rf_val, met_rf_val = evaluar_modelo("RANDOM FOREST   — Validación", pipeline_rf, X_val, y_val)
y_pred_gb_val, met_gb_val = evaluar_modelo("GRADIENT BOOST  — Validación", pipeline_gb, X_val, y_val)

# ── Evaluación en prueba (resultado final)
print("\n══════ CONJUNTO DE PRUEBA ══════")
y_pred_rf, met_rf_test = evaluar_modelo("RANDOM FOREST   — Prueba", pipeline_rf, X_test, y_test)
y_pred_gb, met_gb_test = evaluar_modelo("GRADIENT BOOST  — Prueba", pipeline_gb, X_test, y_test)

# ─────────────────────────────────────────────────────────────
# 8. MATRICES DE CONFUSIÓN  (Entrenamiento y Validación — 4 matrices)
# ─────────────────────────────────────────────────────────────
REPORTES_DIR = r"C:\Users\llano\Desktop\EAFIT\FundamentosApren\MiRepo\Paula_Llanos_Lopez\Lect06_PaulaLlanos\Reportes"
os.makedirs(REPORTES_DIR, exist_ok=True)

clases = sorted(y.unique())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

configuraciones = [
    (axes[0, 0], y_train,  y_pred_rf_train, "Random Forest — Entrenamiento"),
    (axes[0, 1], y_train,  y_pred_gb_train, "Gradient Boosting — Entrenamiento"),
    (axes[1, 0], y_val,    y_pred_rf_val,   "Random Forest — Validación"),
    (axes[1, 1], y_val,    y_pred_gb_val,   "Gradient Boosting — Validación"),
]

for ax, y_real, y_pred, titulo in configuraciones:
    cm = confusion_matrix(y_real, y_pred, labels=clases)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(titulo, fontsize=11, fontweight="bold")

plt.suptitle(
    "Matrices de Confusión — ¿Es Gasto Hormiga?\nEntrenamiento vs Validación",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()

salida_png = os.path.join(REPORTES_DIR, "matrices_confusion_train_val.png")
plt.savefig(salida_png, dpi=150, bbox_inches="tight")
print(f"\nMatrices guardadas en: {salida_png}")

# ─────────────────────────────────────────────────────────────
# 9. GRÁFICA COMPARATIVA DE MÉTRICAS (Train / Val / Test)
# ─────────────────────────────────────────────────────────────
metricas_nombres = ["Accuracy", "Precision", "Recall", "F1 Score"]
conjuntos = ["Entrenamiento", "Validación", "Prueba"]

# Organizar valores por modelo
valores_rf = [
    [met_rf_train[m] for m in metricas_nombres],
    [met_rf_val[m]   for m in metricas_nombres],
    [met_rf_test[m]  for m in metricas_nombres],
]
valores_gb = [
    [met_gb_train[m] for m in metricas_nombres],
    [met_gb_val[m]   for m in metricas_nombres],
    [met_gb_test[m]  for m in metricas_nombres],
]

x      = np.arange(len(metricas_nombres))
width  = 0.13
colores_rf = ["#1f77b4", "#4a9fd4", "#7ec8e3"]   # azules
colores_gb = ["#d62728", "#e87070", "#f5b7b7"]   # rojos

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, valores, colores, titulo in [
    (axes2[0], valores_rf, colores_rf, "Random Forest (Bagging)"),
    (axes2[1], valores_gb, colores_gb, "Gradient Boosting"),
]:
    for i, (conj, vals, color) in enumerate(zip(conjuntos, valores, colores)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=conj, color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8
            )
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_nombres)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Valor")
    ax.legend(title="Conjunto")
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle(
    "Comparación de Métricas — ¿Es Gasto Hormiga?",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()

salida_metricas = os.path.join(REPORTES_DIR, "metricas_comparativas.png")
plt.savefig(salida_metricas, dpi=150, bbox_inches="tight")
print(f"Gráfica de métricas guardada en: {salida_metricas}")