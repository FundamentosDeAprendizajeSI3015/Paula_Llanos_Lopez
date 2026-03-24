import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# Ruta de datasets
proyecto_dir = r"c:\Users\llano\Desktop\EAFIT\FundamentosApren\Github\Paula_Llanos_Lopez\Lect10_PaulaLlanos"

# Crear carpeta Graficas si no existe
graficas_dir = os.path.join(proyecto_dir, "Graficas")
if not os.path.exists(graficas_dir):
    os.makedirs(graficas_dir)
    print(f"[OK] Carpeta 'Graficas' creada")

# ════════════════════════════════════════════════════════════════════════════════
# CARGAR DATOS
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("ANÁLISIS COMPLETO: KMeans + DBSCAN (Matriz de Confusión + Visualización)")
print("="*80)

df_realista_completo = pd.read_csv(os.path.join(proyecto_dir, "dataset_sintetico_FIRE_UdeA_realista.csv"))
df_sintetico_completo = pd.read_csv(os.path.join(proyecto_dir, "dataset_sintetico_FIRE_UdeA.csv"))

labels_realista = df_realista_completo['label'].values
labels_sintetico = df_sintetico_completo['label'].values

df_realista = pd.read_csv(os.path.join(proyecto_dir, "dataset_sintetico_FIRE_UdeA_realista_sin_label.csv"))
df_sintetico = pd.read_csv(os.path.join(proyecto_dir, "dataset_sintetico_FIRE_UdeA_sin_label.csv"))

# ════════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════════

def preparar_datos(df, labels_originales):
    """Prepara datos para clustering"""
    df_numerico = df.select_dtypes(include=[np.number])
    indices_validos = ~df_numerico.isna().any(axis=1)
    df_limpio = df_numerico[indices_validos]
    labels_limpio = labels_originales[indices_validos]
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_limpio)
    
    return df_scaled, labels_limpio

def encontrar_eps_optimo(df_scaled):
    """Encuentra eps óptimo para DBSCAN"""
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(df_scaled)
    distances, indices = neighbors_fit.kneighbors(df_scaled)
    distances = np.sort(distances[:, -1], axis=0)
    eps_optimo = distances[int(len(distances) * 0.9)]
    return eps_optimo

# ════════════════════════════════════════════════════════════════════════════════
# 1. VALIDACIÓN CON K-MEANS
# ════════════════════════════════════════════════════════════════════════════════

def validar_kmeans(df, labels_originales, nombre_dataset):
    """Aplica KMeans con 2 clusters y valida"""
    
    print(f"\n{'='*80}")
    print(f"1. VALIDACIÓN CON K-MEANS - {nombre_dataset}")
    print(f"{'='*80}")
    
    df_numerico = df.select_dtypes(include=[np.number])
    df_limpio = df_numerico.dropna()
    labels_limpio = labels_originales[:len(df_limpio)]
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_limpio)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    
    print(f"\nDatos analizados: {len(labels_limpio)}")
    print(f"Label 0: {(labels_limpio == 0).sum()}")
    print(f"Label 1: {(labels_limpio == 1).sum()}")
    
    # Cálculos
    mask_label_1 = labels_limpio == 1
    mask_label_0 = labels_limpio == 0
    
    count_label_1_cluster_0 = (clusters[mask_label_1] == 0).sum()
    count_label_1_cluster_1 = (clusters[mask_label_1] == 1).sum()
    count_label_0_cluster_0 = (clusters[mask_label_0] == 0).sum()
    count_label_0_cluster_1 = (clusters[mask_label_0] == 1).sum()
    
    if count_label_1_cluster_1 >= count_label_1_cluster_0:
        porcentaje_1 = (count_label_1_cluster_1 / mask_label_1.sum()) * 100
    else:
        porcentaje_1 = (count_label_1_cluster_0 / mask_label_1.sum()) * 100
    
    if count_label_0_cluster_0 >= count_label_0_cluster_1:
        porcentaje_0 = (count_label_0_cluster_0 / mask_label_0.sum()) * 100
    else:
        porcentaje_0 = (count_label_0_cluster_1 / mask_label_0.sum()) * 100
    
    precision_global = (porcentaje_0 + porcentaje_1) / 2
    
    print(f"\nLabel 1: {porcentaje_1:.2f}% correcto")
    print(f"Label 0: {porcentaje_0:.2f}% correcto")
    print(f"PRECISIÓN GLOBAL: {precision_global:.2f}%")
    
    return df_scaled, labels_limpio, clusters, precision_global

# ════════════════════════════════════════════════════════════════════════════════
# 2. MATRIZ DE CONFUSIÓN CON DBSCAN
# ════════════════════════════════════════════════════════════════════════════════

def crear_matriz_confusion_dbscan(df_scaled, labels_originales, eps, min_samples, nombre_dataset):
    """Crea matriz de confusión para DBSCAN"""
    
    print(f"\n{'='*80}")
    print(f"2. MATRIZ DE CONFUSIÓN - DBSCAN - {nombre_dataset}")
    print(f"{'='*80}")
    print(f"Parámetros: eps={eps:.4f}, min_samples={min_samples}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df_scaled)
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_outliers = list(clusters).count(-1)
    
    print(f"\nClusters encontrados: {n_clusters}")
    print(f"Outliers: {n_outliers} ({n_outliers/len(labels_originales)*100:.1f}%)")
    
    # Crear matriz
    matriz = np.zeros((2, n_clusters + 1))
    for i in range(len(labels_originales)):
        label = labels_originales[i]
        cluster = clusters[i]
        if cluster == -1:
            matriz[label, n_clusters] += 1
        else:
            matriz[label, cluster] += 1
    
    cluster_labels = [f'Cluster {i}' for i in range(n_clusters)]
    cluster_labels.append('Outliers')
    row_labels = ['Label 0', 'Label 1']
    
    print(f"\nMatriz de Confusión:")
    print(f"{'─'*80}")
    for i, row in enumerate(row_labels):
        print(f"{row}: {[int(x) for x in matriz[i]]}")
    
    return matriz, cluster_labels, row_labels, n_clusters, clusters

def visualizar_matriz_confusion(matriz, cluster_labels, row_labels, nombre_dataset):
    """Visualiza la matriz de confusión"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Matriz absoluta
    ax1 = axes[0]
    sns.heatmap(matriz, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1, 
                cbar_kws={'label': 'Cantidad'}, 
                xticklabels=cluster_labels, yticklabels=row_labels,
                linewidths=1, linecolor='gray', annot_kws={'size': 12})
    ax1.set_title(f'Matriz de Confusión DBSCAN (Absoluta)\n{nombre_dataset}', 
                 fontweight='bold', fontsize=14)
    ax1.set_xlabel('Clusters Predichos', fontweight='bold')
    ax1.set_ylabel('Etiquetas Originales', fontweight='bold')
    
    # Matriz normalizada
    ax2 = axes[1]
    matriz_norm = matriz.copy()
    for i in range(matriz_norm.shape[0]):
        total = matriz_norm[i].sum()
        if total > 0:
            matriz_norm[i] = (matriz_norm[i] / total) * 100
    
    sns.heatmap(matriz_norm, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2, 
                cbar_kws={'label': 'Porcentaje (%)'}, 
                xticklabels=cluster_labels, yticklabels=row_labels,
                linewidths=1, linecolor='gray', annot_kws={'size': 12}, vmin=0, vmax=100)
    ax2.set_title(f'Matriz de Confusión DBSCAN (Normalizada %)\n{nombre_dataset}', 
                 fontweight='bold', fontsize=14)
    ax2.set_xlabel('Clusters Predichos', fontweight='bold')
    ax2.set_ylabel('Etiquetas Originales', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(graficas_dir, f'dbscan_matriz_confusion_{nombre_dataset.lower().replace(" ", "_")}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"\n[OK] Gráfica guardada: Graficas/dbscan_matriz_confusion_{nombre_dataset.lower().replace(' ', '_')}.png")
    plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# 3. VISUALIZACIONES 3D Y ANÁLISIS CON DBSCAN
# ════════════════════════════════════════════════════════════════════════════════

def crear_visualizaciones_dbscan(df_scaled, labels_originales, clusters, nombre_dataset, n_clusters):
    """Crea visualizaciones 2D/3D para DBSCAN"""
    
    print(f"\n{'='*80}")
    print(f"3. VISUALIZACIONES 3D - DBSCAN - {nombre_dataset}")
    print(f"{'='*80}")
    
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(df_scaled)
    
    print(f"Varianza explicada PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"Varianza explicada PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"Varianza explicada PC3: {pca.explained_variance_ratio_[2]:.2%}")
    print(f"Varianza total: {pca.explained_variance_ratio_.sum():.2%}")
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 2D PCA - Etiquetas Originales
    ax1 = plt.subplot(2, 3, 1)
    scatter1 = ax1.scatter(pca_data[:, 0], pca_data[:, 1], c=labels_originales, 
                          cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('2D PCA - Etiquetas Originales', fontweight='bold', fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label='Label')
    
    # 2. 2D PCA - Clusters DBSCAN
    ax2 = plt.subplot(2, 3, 2)
    colores = np.where(clusters == -1, -1, clusters)
    scatter2 = ax2.scatter(pca_data[:, 0], pca_data[:, 1], c=colores, 
                          cmap='tab10', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    outlier_mask = clusters == -1
    ax2.scatter(pca_data[outlier_mask, 0], pca_data[outlier_mask, 1], 
               marker='x', s=200, c='red', linewidths=3, label='Outliers')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_title('2D PCA - Clusters DBSCAN', fontweight='bold', fontsize=12)
    ax2.legend()
    
    # 3. 2D PCA - Precisión
    ax3 = plt.subplot(2, 3, 3)
    aciertos = np.zeros(len(labels_originales))
    for i in range(len(labels_originales)):
        if clusters[i] != -1:
            aciertos[i] = 1 if (labels_originales[i] == 1 and clusters[i] % 2 == 1 or 
                               labels_originales[i] == 0 and clusters[i] % 2 == 0) else 0
        else:
            aciertos[i] = 0.5
    
    scatter3 = ax3.scatter(pca_data[:, 0], pca_data[:, 1], c=aciertos, 
                          cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black', linewidth=0.5, vmin=0, vmax=1)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax3.set_title('2D PCA - Análisis de Calidad', fontweight='bold', fontsize=12)
    plt.colorbar(scatter3, ax=ax3, label='Calidad')
    
    # 4. 3D PCA - Etiquetas
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    scatter4 = ax4.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], 
                          c=labels_originales, cmap='RdYlGn', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax4.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax4.set_title('3D PCA - Etiquetas Originales', fontweight='bold', fontsize=12)
    
    # 5. 3D PCA - Clusters DBSCAN
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    colores_3d = np.where(clusters == -1, -1, clusters)
    scatter5 = ax5.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], 
                          c=colores_3d, cmap='tab10', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    outlier_mask = clusters == -1
    ax5.scatter(pca_data[outlier_mask, 0], pca_data[outlier_mask, 1], pca_data[outlier_mask, 2],
               marker='x', s=200, c='red', linewidths=3)
    ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax5.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax5.set_title('3D PCA - Clusters DBSCAN', fontweight='bold', fontsize=12)
    
    # 6. Distribución de clusters
    ax6 = plt.subplot(2, 3, 6)
    cluster_counts = []
    cluster_labels_plot = []
    for i in range(n_clusters):
        count = (clusters == i).sum()
        cluster_counts.append(count)
        cluster_labels_plot.append(f'C{i}')
    
    outlier_count = (clusters == -1).sum()
    cluster_counts.append(outlier_count)
    cluster_labels_plot.append('Outliers')
    
    colors_bar = plt.cm.tab10(np.linspace(0, 1, len(cluster_counts)))
    colors_bar[-1] = [1, 0, 0, 1]
    
    bars = ax6.bar(cluster_labels_plot, cluster_counts, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax6.set_ylabel('Número de Puntos')
    ax6.set_title('Distribución de Clusters', fontweight='bold', fontsize=12)
    ax6.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(graficas_dir, f'dbscan_visualizacion_{nombre_dataset.lower().replace(" ", "_")}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"[OK] Gráfica guardada: Graficas/dbscan_visualizacion_{nombre_dataset.lower().replace(' ', '_')}.png")
    plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZACIONES HERMOSAS KMEANS (2D/3D + Reporte Interpretativo)
# ════════════════════════════════════════════════════════════════════════════════

def crear_visualizaciones_kmeans(df_scaled, labels_originales, clusters, nombre_dataset):
    """Crea visualizaciones hermosas en varias dimensiones para KMeans"""
    
    print(f"\n{'='*80}")
    print(f"4. VISUALIZACIONES HERMOSAS KMEANS - {nombre_dataset}")
    print(f"{'='*80}")
    
    # Aplicar PCA
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(df_scaled)
    
    print(f"Varianza explicada PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"Varianza explicada PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"Varianza explicada PC3: {pca.explained_variance_ratio_[2]:.2%}")
    print(f"Varianza total (3 componentes): {pca.explained_variance_ratio_.sum():.2%}")
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. VISUALIZACIÓN 2D CON PCA - Etiquetas Originales
    ax1 = plt.subplot(2, 3, 1)
    scatter1 = ax1.scatter(pca_data[:, 0], pca_data[:, 1], c=labels_originales, 
                          cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('2D PCA - Etiquetas Originales\n(Lo que los analistas dijeron)', fontweight='bold', fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label='Label')
    
    # 2. VISUALIZACIÓN 2D CON PCA - Clusters KMeans
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, 
                          cmap='coolwarm', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_title('2D PCA - Clusters KMeans\n(Lo que encontró el algoritmo)', fontweight='bold', fontsize=12)
    plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    # 3. VISUALIZACIÓN 2D - Acuracidad Visual
    ax3 = plt.subplot(2, 3, 3)
    # Determinar mapeo correcto de clusters a labels
    mask_label_1 = labels_originales == 1
    if (clusters[mask_label_1] == 1).sum() >= (clusters[mask_label_1] == 0).sum():
        cluster_correcto = 1
    else:
        cluster_correcto = 0
    
    # Crear array de aciertos
    aciertos = np.zeros(len(labels_originales))
    for i in range(len(labels_originales)):
        if labels_originales[i] == 1:
            aciertos[i] = 1 if clusters[i] == cluster_correcto else 0
        else:
            aciertos[i] = 1 if clusters[i] != cluster_correcto else 0
    
    scatter3 = ax3.scatter(pca_data[:, 0], pca_data[:, 1], c=aciertos, 
                          cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax3.set_title('2D PCA - Precisión\n(Verde=Bien, Rojo=Mal)', fontweight='bold', fontsize=12)
    plt.colorbar(scatter3, ax=ax3, label='Acierto')
    
    # 4. VISUALIZACIÓN 3D CON PCA - Etiquetas
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    scatter4 = ax4.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], 
                          c=labels_originales, cmap='RdYlGn', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax4.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax4.set_title('3D PCA - Etiquetas Originales', fontweight='bold', fontsize=12)
    
    # 5. VISUALIZACIÓN 3D CON CLUSTERS
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    scatter5 = ax5.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], 
                          c=clusters, cmap='coolwarm', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax5.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax5.set_title('3D PCA - Clusters KMeans', fontweight='bold', fontsize=12)
    
    # 6. MATRIZ DE CONFUSIÓN
    ax6 = plt.subplot(2, 3, 6)
    
    # Crear matriz de confusión
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(labels_originales)):
        confusion_matrix[labels_originales[i]][clusters[i]] += 1
    
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='YlOrRd', ax=ax6, 
                cbar_kws={'label': 'Cantidad'}, xticklabels=['Cluster 0', 'Cluster 1'],
                yticklabels=['Label 0', 'Label 1'])
    ax6.set_title('Matriz de Confusión\n(Fila=Etiquetas, Columna=Clusters)', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Etiqueta Original')
    ax6.set_xlabel('Cluster Predicho')
    
    plt.tight_layout()
    plt.savefig(os.path.join(graficas_dir, f'visualizacion_clusters_kmeans_{nombre_dataset.lower().replace(" ", "_")}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"[OK] Gráfica guardada: Graficas/visualizacion_clusters_kmeans_{nombre_dataset.lower().replace(' ', '_')}.png")
    plt.close()

def generar_reporte_kmeans(labels_originales, clusters, nombre_dataset, precision_global):
    """Genera un reporte interpretativo con veredicto sobre los analistas"""
    
    print(f"\n{'='*80}")
    print(f"REPORTE INTERPRETATIVO - {nombre_dataset}")
    print(f"{'='*80}")
    
    # Calcular métricas
    mask_label_1 = labels_originales == 1
    mask_label_0 = labels_originales == 0
    
    count_label_1_cluster_0 = (clusters[mask_label_1] == 0).sum()
    count_label_1_cluster_1 = (clusters[mask_label_1] == 1).sum()
    count_label_0_cluster_0 = (clusters[mask_label_0] == 0).sum()
    count_label_0_cluster_1 = (clusters[mask_label_0] == 1).sum()
    
    # Determinar cluster correcto
    if count_label_1_cluster_1 >= count_label_1_cluster_0:
        cluster_correcto_1 = 1
        porcentaje_1 = (count_label_1_cluster_1 / mask_label_1.sum()) * 100
    else:
        cluster_correcto_1 = 0
        porcentaje_1 = (count_label_1_cluster_0 / mask_label_1.sum()) * 100
    
    if count_label_0_cluster_0 >= count_label_0_cluster_1:
        cluster_correcto_0 = 0
        porcentaje_0 = (count_label_0_cluster_0 / mask_label_0.sum()) * 100
    else:
        cluster_correcto_0 = 1
        porcentaje_0 = (count_label_0_cluster_1 / mask_label_0.sum()) * 100
    
    print(f"\n[DESEMPEÑO DE LA ETIQUETACION]\n")
    print(f"  - Label 1: {porcentaje_1:.1f}% etiquetado correctamente")
    print(f"  - Label 0: {porcentaje_0:.1f}% etiquetado correctamente")
    print(f"  - PROMEDIO: {precision_global:.1f}% de precisión")
    
    print(f"\n{'-'*80}")
    print(f"[VEREDICTO SOBRE LOS ANALISTAS FINANCIEROS]\n")
    
    if precision_global >= 85:
        print(f"  [EXCELENTE] ASCENDER INMEDIATAMENTE (Precisión: {precision_global:.1f}%)")
        print(f"  - Los analistas son EXCELENTES distinguiendo las categorías.")
        print(f"  - Muestran una comprensión profunda de los datos financieros.")
        print(f"  - Recomendación: Promover y darles bonificación por desempeño.")
    elif precision_global >= 75:
        print(f"  [BUENO] MANTENER Y CAPACITAR (Precisión: {precision_global:.1f}%)")
        print(f"  - Los analistas son BUENOS, pero hay margen de mejora.")
        print(f"  - Algunos datos fronterizos fueron mal clasificados.")
        print(f"  - Recomendación: Entrenarlos en casos edge y aumentar salario.")
    elif precision_global >= 60:
        print(f"  [ALERTA] REVISAR Y MEJORAR (Precisión: {precision_global:.1f}%)")
        print(f"  - El trabajo de los analistas es MEDIOCRE.")
        print(f"  - Hay confusión consistente entre categorías.")
        print(f"  - Recomendación: Auditoría, capacitación intensiva, o reasignación.")
    else:
        print(f"  [CRITICO] ECHAR DEL TRABAJO (Precisión: {precision_global:.1f}%)")
        print(f"  - Los analistas NO entienden los patrones de datos.")
        print(f"  - Mejor usar KMeans automático que etiquetas manuales.")
        print(f"  - Recomendación: Terminar contrato y buscar nuevos analistas.")
    
    print(f"\n{'-'*80}")
    print(f"[DETALLES TECNICOS]\n")
    print(f"  N total de casos: {len(labels_originales)}")
    print(f"  Casos Label 1: {mask_label_1.sum()} ({mask_label_1.sum()/len(labels_originales)*100:.1f}%)")
    print(f"  Casos Label 0: {mask_label_0.sum()} ({mask_label_0.sum()/len(labels_originales)*100:.1f}%)")
    
    # Análisis de outliers
    mal_clasificados = []
    for i in range(len(labels_originales)):
        if labels_originales[i] == 1:
            if clusters[i] != cluster_correcto_1:
                mal_clasificados.append(i)
        else:
            if clusters[i] != cluster_correcto_0:
                mal_clasificados.append(i)
    
    if len(mal_clasificados) > 0:
        print(f"\n  Puntos mal etiquetados: {len(mal_clasificados)} ({len(mal_clasificados)/len(labels_originales)*100:.1f}%)")

# ════════════════════════════════════════════════════════════════════════════════
# ANÁLISIS COMPLETO
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PROCESANDO DATASET REALISTA")
print("="*80)

df_scaled_r, labels_r = preparar_datos(df_realista, labels_realista)
df_scaled_r_km, labels_r_km, clusters_r_km, prec_r_km = validar_kmeans(df_realista, labels_realista, "Dataset REALISTA")

# Visualizaciones y reporte KMeans
crear_visualizaciones_kmeans(df_scaled_r_km, labels_r_km, clusters_r_km, "Dataset REALISTA")
generar_reporte_kmeans(labels_r_km, clusters_r_km, "Dataset REALISTA", prec_r_km)

# Análisis DBSCAN
eps_r = encontrar_eps_optimo(df_scaled_r)
matriz_r, labels_cluster_r, labels_row_r, n_clusters_r, clusters_r_db = crear_matriz_confusion_dbscan(
    df_scaled_r, labels_r, eps_r, min_samples=5, nombre_dataset="Dataset REALISTA"
)
# visualizar_matriz_confusion(matriz_r, labels_cluster_r, labels_row_r, "Dataset REALISTA")
crear_visualizaciones_dbscan(df_scaled_r, labels_r, clusters_r_db, "Dataset REALISTA", n_clusters_r)

print("\n" + "="*80)
print("PROCESANDO DATASET SINTÉTICO")
print("="*80)

df_scaled_s, labels_s = preparar_datos(df_sintetico, labels_sintetico)
df_scaled_s_km, labels_s_km, clusters_s_km, prec_s_km = validar_kmeans(df_sintetico, labels_sintetico, "Dataset SINTÉTICO")

# Visualizaciones y reporte KMeans
crear_visualizaciones_kmeans(df_scaled_s_km, labels_s_km, clusters_s_km, "Dataset SINTÉTICO")
generar_reporte_kmeans(labels_s_km, clusters_s_km, "Dataset SINTÉTICO", prec_s_km)

# Análisis DBSCAN
eps_s = encontrar_eps_optimo(df_scaled_s)
matriz_s, labels_cluster_s, labels_row_s, n_clusters_s, clusters_s_db = crear_matriz_confusion_dbscan(
    df_scaled_s, labels_s, eps_s, min_samples=5, nombre_dataset="Dataset SINTÉTICO"
)
# visualizar_matriz_confusion(matriz_s, labels_cluster_s, labels_row_s, "Dataset SINTÉTICO")
crear_visualizaciones_dbscan(df_scaled_s, labels_s, clusters_s_db, "Dataset SINTÉTICO", n_clusters_s)

# ════════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("RESUMEN FINAL DE ANÁLISIS")
print("="*80)

print("\n[DATASET REALISTA]")
print(f"  KMeans Precisión Global: {prec_r_km:.2f}%")
print(f"  DBSCAN Clusters: {n_clusters_r} | Outliers: {(clusters_r_db == -1).sum()}")

print("\n[DATASET SINTETICO]")
print(f"  KMeans Precisión Global: {prec_s_km:.2f}%")
print(f"  DBSCAN Clusters: {n_clusters_s} | Outliers: {(clusters_s_db == -1).sum()}")

print("\n[COMPLETO] ANÁLISIS COMPLETADO")
print("="*80)
print("\nArchivos generados en Graficas/:")
print("  KMeans:")
print("    ├─ visualizacion_clusters_kmeans_dataset_realista.png")
print("    └─ visualizacion_clusters_kmeans_dataset_sintético.png")
print("  DBSCAN:")
print("    ├─ dbscan_visualizacion_dataset_realista.png")
print("    └─ dbscan_visualizacion_dataset_sintético.png")
print("="*80 + "\n")
