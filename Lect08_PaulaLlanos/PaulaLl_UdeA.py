"""
==============================================================================
FIRE-UdeA  |  Modelo de Riesgo Financiero Universitario
==============================================================================
Descripción:
  Modelo de clasificación basado en Gradient Boosting para estimar la
  probabilidad de tensión de caja en el período t+1.

Features:
  Ratios financieros, composición de ingresos, participación por fuente,
  liquidez, CFO, volatilidad, tendencias, HHI, endeudamiento, gp_ratio.

Variable Objetivo (label):
  Tensión financiera = 1  si:
    - CFO negativo dos años consecutivos, O
    - Liquidez < 1, O
    - Días de efectivo < 30
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('graficas', exist_ok=True)

# Limpiar graficas
for _f in os.listdir('graficas'):
    if _f.endswith('.png'):
        os.remove(os.path.join('graficas', _f))

SEED = 42
np.random.seed(SEED)

TARGET   = 'label'
FEATURES = [
    'liquidez', 'dias_efectivo', 'cfo',
    'participacion_ley30', 'participacion_regalias',
    'participacion_servicios', 'participacion_matriculas',
    'hhi_fuentes', 'endeudamiento', 'tendencia_ingresos', 'gp_ratio',
]

# =============================================================================
# 1. CARGA DEL DATASET
# =============================================================================
df = pd.read_csv('dataset_sintetico_FIRE_UdeA_realista.csv')

print("=" * 65)
print("  FIRE-UdeA — MODELO DE RIESGO FINANCIERO UNIVERSITARIO")
print("=" * 65)
print(f"Dimensiones     : {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Columnas        : {list(df.columns)}")
print(f"Valores nulos   : {df.isnull().sum().sum()}")
print(f"\nBalance label:")
vc = df[TARGET].value_counts().sort_index()
for clase, cnt in vc.items():
    print(f"  Clase {clase}: {cnt} registros ({cnt/len(df)*100:.1f}%)")

# =============================================================================
# 2. EDA — CRITERIOS DE TENSION FINANCIERA (DEFINICION DEL LABEL)
# =============================================================================
print("\n" + "=" * 65)
print("  CRITERIOS DE TENSION FINANCIERA (definicion del label)")
print("=" * 65)
print("  label = 1  SI:")
print("    • CFO negativo dos anos consecutivos, O")
print("    • Liquidez < 1, O")
print("    • Dias de efectivo < 30")
print("  label = 0  en caso contrario")

# Verificar cuantos registros de clase-1 cumplen cada criterio
for var, umbral, direccion, desc in [
    ('cfo',           0,  'menor', 'CFO < 0'),
    ('liquidez',      1,  'menor', 'Liquidez < 1'),
    ('dias_efectivo', 30, 'menor', 'Dias efectivo < 30'),
]:
    sub = df[[var, TARGET]].dropna()
    con_riesgo = sub[sub[TARGET] == 1]
    if direccion == 'menor':
        cumplen = (con_riesgo[var] < umbral).mean() * 100
    print(f"  Clase 1 con {desc:<28}: {cumplen:>5.1f}%")

# Grafico criterios de tension
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
criterios_plot = [
    ('cfo',           0,  'CFO (Flujo de Caja Operativo)', 'CFO = 0'),
    ('liquidez',      1,  'Liquidez',                       'Umbral = 1'),
    ('dias_efectivo', 30, 'Dias de Efectivo',               'Umbral = 30 dias'),
]
for ax, (var, umbral, xlabel, etiq) in zip(axes, criterios_plot):
    for clase, color, lbl in [(0, '#2ecc71', 'Sin Riesgo'), (1, '#e74c3c', 'Con Riesgo')]:
        datos = df[df[TARGET] == clase][var].dropna()
        ax.hist(datos, bins=14, alpha=0.65, color=color, label=lbl,
                edgecolor='white', linewidth=0.5)
    ax.axvline(umbral, color='black', linestyle='--', linewidth=2, label=etiq)
    ax.set_title(f'{xlabel}', fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.4)
plt.suptitle('Criterios de Tension Financiera — Definicion de la Variable Objetivo',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas/01_criterios_tension.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 3. EDA — VALORES NULOS
# =============================================================================
nulos     = df.isnull().sum()
pct_nulos = (nulos / len(df) * 100).round(2)
tabla_nulos = pd.DataFrame({'Valores Nulos': nulos, '% del Total': pct_nulos})
tabla_nulos = tabla_nulos[tabla_nulos['Valores Nulos'] > 0].sort_values('% del Total', ascending=False)
print(f"\nVariables con valores nulos ({len(tabla_nulos)} de {df.shape[1]}):")
print(tabla_nulos)

fig, ax = plt.subplots(figsize=(13, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
ax.set_title('Mapa de Valores Nulos por Variable', fontsize=14, fontweight='bold')
ax.set_xlabel('Variables')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('graficas/02_valores_nulos.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 4. EDA — BALANCE DE CLASES
# =============================================================================
conteo = df[TARGET].value_counts().sort_index()
labels_desc = {0: 'Sin Riesgo (0)', 1: 'Riesgo Financiero (1)'}
idx_labels = [labels_desc[i] for i in conteo.index]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(idx_labels, conteo.values, color=['#2ecc71', '#e74c3c'],
            edgecolor='black', linewidth=0.8)
axes[0].set_title('Distribucion de la Variable Objetivo', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Cantidad de Registros')
for i, v in enumerate(conteo.values):
    axes[0].text(i, v + 0.3, str(v), ha='center', fontsize=12, fontweight='bold')
pct = conteo / conteo.sum() * 100
axes[1].pie(pct.values, labels=idx_labels, autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1].set_title('Proporcion de Clases', fontsize=13, fontweight='bold')
plt.suptitle('Variable Objetivo: label (Tension Financiera FIRE-UdeA)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('graficas/03_balance_clases.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 5. EDA — ESTADISTICA DESCRIPTIVA POR CLASE
# =============================================================================
print("\n" + "=" * 65)
print("  ESTADISTICA DESCRIPTIVA POR CLASE")
print("=" * 65)
for clase in [0, 1]:
    print(f"\n  Clase {clase} — {'Sin Riesgo' if clase == 0 else 'Con Riesgo'}:")
    sub = df[df[TARGET] == clase][FEATURES].describe().loc[['mean', 'std', 'min', '50%', 'max']]
    print(sub.round(3).to_string())

# Heatmap medias normalizadas por clase
medias = df.groupby(TARGET)[FEATURES].mean()
medias_norm = (medias - medias.min()) / (medias.max() - medias.min() + 1e-9)
fig, ax = plt.subplots(figsize=(14, 3.5))
sns.heatmap(medias_norm, annot=medias.round(3).values, fmt='.3f', cmap='RdYlGn_r',
            ax=ax, linewidths=0.5, linecolor='white',
            xticklabels=FEATURES,
            yticklabels=['Sin Riesgo (0)', 'Con Riesgo (1)'],
            annot_kws={'size': 8})
ax.set_title('Medias por Variable segun Clase (rojo = mayor valor, verde = menor)',
             fontsize=11, fontweight='bold')
plt.xticks(rotation=40, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('graficas/04_medias_por_clase.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. EDA — DISTRIBUCIONES POR CLASE
# =============================================================================
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
colores   = {0: '#2ecc71', 1: '#e74c3c'}
etiquetas = {0: 'Sin Riesgo', 1: 'Con Riesgo'}

skewness = df[FEATURES].skew().round(3)
for i, var in enumerate(FEATURES):
    ax = axes[i]
    for clase in [0, 1]:
        datos = df[df[TARGET] == clase][var].dropna()
        ax.hist(datos, bins=13, alpha=0.62, color=colores[clase],
                label=etiquetas[clase], edgecolor='white', linewidth=0.4)
    ax.set_title(f'{var}  | skew={skewness[var]:.2f}', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.4)

axes[-1].set_visible(False)
plt.suptitle('Distribucion de Variables por Clase de Riesgo', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('graficas/05_distribuciones.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 7. EDA — BOXPLOTS POR VARIABLE Y CLASE
# =============================================================================
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for i, var in enumerate(FEATURES):
    ax = axes[i]
    data_plot = df[['label', var]].dropna().copy()
    data_plot['Clase'] = data_plot['label'].map({0: 'Sin Riesgo (0)', 1: 'Con Riesgo (1)'})
    sns.boxplot(data=data_plot, x='Clase', y=var, palette=['#2ecc71', '#e74c3c'],
                ax=ax, width=0.5, flierprops=dict(marker='o', markersize=4, alpha=0.5))
    ax.set_title(var, fontsize=10, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.4)
axes[-1].set_visible(False)
plt.suptitle('Boxplots por Variable y Clase de Riesgo', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('graficas/06_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 8. EDA — MATRIZ DE CORRELACION
# =============================================================================
cols_corr   = FEATURES + [TARGET]
corr_matrix = df[cols_corr].corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, linecolor='white', annot_kws={'size': 9})
ax.set_title('Matriz de Correlacion — Variables Numericas + Label', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('graficas/07_correlacion.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nCorrelacion de cada variable con 'label':")
corr_target = corr_matrix[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
for var, val in corr_target.items():
    barra = '█' * int(abs(val) * 20)
    signo = '+' if val > 0 else '-'
    print(f"  {var:<30} {signo}{abs(val):.3f}  {barra}")

# =============================================================================
# 9. EDA — RIESGO POR UNIDAD ACADEMICA Y ANO
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
riesgo_unidad = df.groupby('unidad')[TARGET].mean().sort_values(ascending=False)
bars = axes[0].barh(riesgo_unidad.index, riesgo_unidad.values * 100,
                    color=['#e74c3c' if v > 0.5 else '#3498db' for v in riesgo_unidad.values],
                    edgecolor='black', linewidth=0.6)
axes[0].set_xlabel('% Registros con Tension Financiera')
axes[0].set_title('Tasa de Riesgo por Unidad Academica', fontsize=11, fontweight='bold')
axes[0].axvline(50, color='black', linestyle='--', alpha=0.5, label='50%')
for bar, val in zip(bars, riesgo_unidad.values):
    axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{val*100:.0f}%', va='center', fontsize=9)
axes[0].legend()

riesgo_anio = df.groupby('anio')[TARGET].mean()
axes[1].plot(riesgo_anio.index, riesgo_anio.values * 100, 'o-', color='#8e44ad',
             linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
axes[1].fill_between(riesgo_anio.index, riesgo_anio.values * 100, alpha=0.15, color='#8e44ad')
axes[1].axhline(50, color='red', linestyle='--', alpha=0.6, label='50%')
axes[1].set_xlabel('Ano')
axes[1].set_ylabel('% Unidades con Tension')
axes[1].set_title('Evolucion del Riesgo Financiero por Ano', fontsize=11, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('graficas/08_riesgo_unidad_ano.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 10. PREPROCESAMIENTO
# =============================================================================
print("\n" + "=" * 65)
print("  PREPROCESAMIENTO")
print("=" * 65)

X = df[FEATURES].copy()
y = df[TARGET].copy()

imputer  = SimpleImputer(strategy='median')
X_imp    = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)
mask_val = y.notna()
X_clean  = X_imp[mask_val].reset_index(drop=True)
y_clean  = y[mask_val].reset_index(drop=True).astype(int)

print(f"Registros limpios : {X_clean.shape[0]} x {X_clean.shape[1]}")
print(f"  Sin Riesgo (0)  : {(y_clean == 0).sum()}")
print(f"  Con Riesgo (1)  : {(y_clean == 1).sum()}")
print(f"Imputacion        : mediana por variable")
print(f"Desbalance        : {(y_clean==0).sum() / (y_clean==1).sum():.2f}:1  -> class_weight='balanced'")

# =============================================================================
# 11. DIVISION TRAIN / TEST  (75% / 25%, estratificado)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.25, random_state=SEED, stratify=y_clean
)
print(f"\nEntrenamiento : {X_train.shape[0]} registros")
print(f"Prueba        : {X_test.shape[0]} registros")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# =============================================================================
# 12. MODELO 1 — ARBOL DE DECISION (BASELINE)
# =============================================================================
print("\n" + "=" * 65)
print("  MODELO 1 — ARBOL DE DECISION (baseline)")
print("=" * 65)
dt = DecisionTreeClassifier(criterion='gini', class_weight='balanced', random_state=SEED)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]
cv_dt     = cross_val_score(dt, X_train, y_train, cv=cv, scoring='f1').mean()
print(f"  Profundidad : {dt.get_depth()}   |   Nodos hoja : {dt.get_n_leaves()}")
print(f"  CV F1 (5-fold) : {cv_dt:.4f}")

# =============================================================================
# 13. MODELO 2 — RANDOM FOREST
# =============================================================================
print("\n" + "=" * 65)
print("  MODELO 2 — RANDOM FOREST")
print("=" * 65)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None,
    min_samples_split=4, min_samples_leaf=2,
    max_features='sqrt', class_weight='balanced',
    random_state=SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf  = rf.predict_proba(X_test)[:, 1]
cv_rf      = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1').mean()
print(f"  Estimadores : {rf.n_estimators}   |   max_features : sqrt")
print(f"  CV F1 (5-fold) : {cv_rf:.4f}")

# =============================================================================
# 14. MODELO 3 — GRADIENT BOOSTING  (modelo principal)
# =============================================================================
print("\n" + "=" * 65)
print("  MODELO 3 — GRADIENT BOOSTING  [MODELO PRINCIPAL]")
print("=" * 65)
print("  Objetivo: estimar P(tension financiera en t+1)")
gbm = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.08,
    max_depth=4, min_samples_split=5,
    min_samples_leaf=3, subsample=0.8,
    max_features='sqrt', random_state=SEED
)
gbm.fit(X_train, y_train)
y_pred_gbm = gbm.predict(X_test)
y_prob_gbm  = gbm.predict_proba(X_test)[:, 1]
cv_gbm      = cross_val_score(gbm, X_train, y_train, cv=cv, scoring='f1').mean()
print(f"  Estimadores : {gbm.n_estimators}   |   lr : {gbm.learning_rate}   |   max_depth : {gbm.max_depth}")
print(f"  Subsample   : {gbm.subsample}   |   max_features : sqrt")
print(f"  CV F1 (5-fold) : {cv_gbm:.4f}")

# =============================================================================
# 15. COMPARACION DE METRICAS
# =============================================================================
print("\n" + "=" * 65)
print("  COMPARACION DE METRICAS — TODOS LOS MODELOS")
print("=" * 65)

modelos_info = {
    'Arbol Decision'   : (y_pred_dt,  y_prob_dt,  cv_dt,  'steelblue'),
    'Random Forest'    : (y_pred_rf,  y_prob_rf,  cv_rf,  '#27ae60'),
    'Gradient Boosting': (y_pred_gbm, y_prob_gbm, cv_gbm, '#e67e22'),
}

resultados = {}
print(f"\n{'Modelo':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ROC-AUC':>8} {'CV-F1':>7}")
print("-" * 65)
for nombre, (y_pred, y_prob, cv_s, _) in modelos_info.items():
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_prob)
    resultados[nombre] = dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, ROC_AUC=roc, CV_F1=cv_s)
    marker = '>>>' if nombre == 'Gradient Boosting' else '   '
    print(f"{marker} {nombre:<19} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {roc:>8.4f} {cv_s:>7.4f}")

mejor = max(resultados, key=lambda k: resultados[k]['F1'])
print(f"\n>>> Mejor modelo por F1-Score: {mejor}")

# Grafico comparacion de metricas
df_res  = pd.DataFrame(resultados).T
met_plt = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
colores_mod = ['steelblue', '#27ae60', '#e67e22']

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(met_plt))
ancho = 0.25
for i, (nombre, color) in enumerate(zip(df_res.index, colores_mod)):
    vals = [df_res.loc[nombre, m] for m in met_plt]
    bars = ax.bar(x + i * ancho, vals, ancho, label=nombre, color=color,
                  edgecolor='black', linewidth=0.6, alpha=0.88)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
ax.set_xticks(x + ancho)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparacion de Metricas — DT  vs  Random Forest  vs  Gradient Boosting',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('graficas/09_comparacion_metricas.png', dpi=150, bbox_inches='tight')
plt.close()

# CV F1
fig, ax = plt.subplots(figsize=(7, 4))
cv_vals = [resultados[n]['CV_F1'] for n in resultados]
bars = ax.bar(list(resultados.keys()), cv_vals, color=colores_mod,
              edgecolor='black', linewidth=0.7, alpha=0.9)
for bar, val in zip(bars, cv_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_ylabel('CV F1-Score (5-fold)', fontsize=12)
ax.set_title('Validacion Cruzada F1 — 5-Fold Estratificado', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('graficas/10_cv_f1.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 16. MATRICES DE CONFUSION — LOS TRES MODELOS
# =============================================================================
class_names = ['Sin Riesgo (0)', 'Con Riesgo (1)']
fig, axes   = plt.subplots(1, 3, figsize=(17, 5))
for ax, (nombre, (y_pred, _, _, color)) in zip(axes, modelos_info.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=2, linecolor='white', annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Prediccion', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    f1_val  = resultados[nombre]['F1']
    roc_val = resultados[nombre]['ROC_AUC']
    ax.set_title(f'{nombre}\nF1={f1_val:.3f}  |  ROC={roc_val:.3f}', fontsize=10, fontweight='bold')
plt.suptitle('Matrices de Confusion — Comparacion de Modelos', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas/11_matrices_confusion.png', dpi=150, bbox_inches='tight')
plt.close()

# Interpretacion del mejor modelo
print(f"\nInterpretacion de la matriz — {mejor}:")
cm_m = confusion_matrix(y_test, modelos_info[mejor][0])
print(f"  VP — detecto tension correctamente     : {cm_m[1,1]}")
print(f"  VN — detecto sin tension correctamente : {cm_m[0,0]}")
print(f"  FP — alerto tension innecesariamente   : {cm_m[0,1]}")
print(f"  FN — no detecto tension real (critico) : {cm_m[1,0]}")
print(f"\nReporte completo — {mejor}:")
print(classification_report(y_test, modelos_info[mejor][0], target_names=class_names))

# =============================================================================
# 17. CURVAS ROC
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 6))
for nombre, (_, y_prob, _, color) in modelos_info.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    lw  = 2.5 if nombre == 'Gradient Boosting' else 1.8
    ax.plot(fpr, tpr, linewidth=lw, color=color, label=f'{nombre}  (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Clasificador aleatorio')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
ax.set_title('Curvas ROC — Deteccion de Tension Financiera FIRE-UdeA', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('graficas/12_curvas_roc.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 18. IMPORTANCIA DE VARIABLES — RF y GBM
# =============================================================================
imp_rf  = pd.Series(rf.feature_importances_,  index=FEATURES).sort_values(ascending=False)
imp_gbm = pd.Series(gbm.feature_importances_, index=FEATURES).sort_values(ascending=False)

print("\n" + "=" * 65)
print("  IMPORTANCIA DE VARIABLES")
print("=" * 65)
print("\nRandom Forest:")
for v, val in imp_rf.items():
    barra = '▓' * int(val * 50)
    print(f"  {v:<30} {val*100:>6.2f}%  {barra}")
print("\nGradient Boosting:")
for v, val in imp_gbm.items():
    barra = '▓' * int(val * 50)
    print(f"  {v:<30} {val*100:>6.2f}%  {barra}")

# Grafico comparativo
df_imp = pd.DataFrame({'Random Forest': imp_rf * 100, 'Gradient Boosting': imp_gbm * 100})
df_imp = df_imp.sort_values('Random Forest', ascending=True)

fig, ax = plt.subplots(figsize=(11, 7))
y_pos = np.arange(len(df_imp))
ax.barh(y_pos - 0.2, df_imp['Random Forest'],     0.38, color='#27ae60',
        label='Random Forest',     edgecolor='black', linewidth=0.5)
ax.barh(y_pos + 0.2, df_imp['Gradient Boosting'],  0.38, color='#e67e22',
        label='Gradient Boosting', edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(df_imp.index, fontsize=10)
ax.set_xlabel('Importancia (%)', fontsize=12)
ax.set_title('Importancia de Variables — Random Forest vs Gradient Boosting',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig('graficas/13_importancia_rf_gbm.png', dpi=150, bbox_inches='tight')
plt.close()

# Top 5 del modelo principal (GBM)
print(f"\nTop 5 variables criticas — Gradient Boosting (modelo principal):")
for rank, (var, val) in enumerate(imp_gbm.head(5).items(), 1):
    print(f"  #{rank}  {var:<30} {val*100:.2f}%")

# =============================================================================
# 19. VISUALIZACION DEL ARBOL BASELINE (max_depth=4)
# =============================================================================
dt_vis = DecisionTreeClassifier(criterion='gini', max_depth=4,
                                 class_weight='balanced', random_state=SEED)
dt_vis.fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(dt_vis, feature_names=FEATURES,
          class_names=['Sin Riesgo', 'Con Riesgo'],
          filled=True, rounded=True, fontsize=8, ax=ax,
          impurity=True, proportion=False, precision=3)
ax.set_title('Arbol de Decision — FIRE UdeA (max_depth=4, visualizacion del baseline)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas/14_arbol_decision.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 20. RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 65)
print("  RESUMEN FINAL  —  FIRE-UdeA")
print("=" * 65)
print(f"  Dataset       : {df.shape[0]} registros x {df.shape[1]} columnas")
print(f"  Features      : {len(FEATURES)} variables")
print(f"  Imputacion    : mediana")
print(f"  Split         : 75% train / 25% test  (estratificado)")
print(f"  Validacion    : 5-Fold Estratificado CV")
print()
for nombre in resultados:
    r = resultados[nombre]
    tag = '  [PRINCIPAL]' if nombre == 'Gradient Boosting' else ''
    print(f"  {'>>>' if nombre == mejor else '   '} {nombre:<22}{tag}")
    print(f"      Acc={r['Accuracy']:.4f}  F1={r['F1']:.4f}  ROC={r['ROC_AUC']:.4f}  CV-F1={r['CV_F1']:.4f}")
print()
print(f"  Graficas guardadas en graficas/ (14 archivos)")
print("=" * 65)