"""
=============================================================================
DETECCIÓN DE GASTOS HORMIGA - ENFOQUE HÍBRIDO NO SUPERVISADO + SUPERVISADO
=============================================================================
Script completo de Data Science para identificar transacciones clasificadas 
como "Gastos Hormiga" (pequeños gastos frecuentes) utilizando:
  - Análisis No Supervisado (Clustering)
  - Reevaluación de Etiquetas mediante Consenso
  - Modelado Supervisado (Comparación de escenarios)

Autor: Data Scientist Senior
Fecha: 2026
=============================================================================
"""

# ============================================================================
# 1. IMPORTACIONES Y CONFIGURACIÓN INICIAL
# ============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Clustering avanzado
import skfuzzy as fuzz
import umap

# Visualización
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import graphviz

# Suprimir warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# 2. FUNCIONES DE PREPROCESAMIENTO Y FEATURE ENGINEERING
# ============================================================================

class PreprocesadorDatos:
    """Clase para preprocesar y extraer características del dataset."""
    
    def __init__(self, df):
        self.df = df.copy()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.scaler_valor = StandardScaler()
        
    def extraer_caracteristicas_fecha(self):
        """Extrae características de la columna FECHA."""
        print("[PREP] Extrayendo características de FECHA...")
        
        # Intentar múltiples formatos de fecha
        try:
            self.df['FECHA'] = pd.to_datetime(self.df['FECHA'], format='%d/%m/%Y')
        except:
            try:
                self.df['FECHA'] = pd.to_datetime(self.df['FECHA'], format='%m/%d/%Y')
            except:
                # Usar inferencia automática
                self.df['FECHA'] = pd.to_datetime(self.df['FECHA'], format='mixed', dayfirst=True)
        
        self.df['DIA'] = self.df['FECHA'].dt.day
        self.df['MES'] = self.df['FECHA'].dt.month
        self.df['DIA_SEMANA'] = self.df['FECHA'].dt.dayofweek  # 0=lunes, 6=domingo
        self.df['TRIMESTRE'] = self.df['FECHA'].dt.quarter
        self.df['SEMANA_AÑO'] = self.df['FECHA'].dt.isocalendar().week
        
        # Crear variable binaria para fin de semana
        self.df['ES_FINDE'] = (self.df['DIA_SEMANA'] >= 5).astype(int)
        
        return self.df
    
    def vectorizar_descripcion(self, max_features=50):
        """Vectoriza la columna DESCRIPCION usando TF-IDF."""
        print(f"[PREP] Vectorizando DESCRIPCION con TF-IDF (max_features={max_features})...")
        
        # Limpieza básica del texto
        self.df['DESCRIPCION_LIMPIA'] = (
            self.df['DESCRIPCION']
            .str.upper()
            .str.strip()
            .str.replace(r'[^\w\s]', '', regex=True)
        )
        
        # Aplicar TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.df['DESCRIPCION_LIMPIA']
        ).toarray()
        
        # Crear DataFrame con características TF-IDF
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(
            self.tfidf_matrix,
            columns=[f'TFIDF_{name}' for name in feature_names]
        )
        
        self.df = pd.concat([self.df.reset_index(drop=True), tfidf_df], axis=1)
        print(f"[PREP]   → Se extrajeron {len(feature_names)} características TF-IDF")
        
        return self.df
    
    def escalar_valor(self):
        """Escala la columna VALOR."""
        print("[PREP] Escalando VALOR...")
        valor_original = self.df[['VALOR']].copy()
        self.df['VALOR_ESCALADO'] = self.scaler_valor.fit_transform(valor_original)
        return self.df
    
    def crear_features_adicionales(self):
        """Crea características adicionales."""
        print("[PREP] Creando características adicionales...")
        
        # Logaritmo del valor (para capturar magnitud)
        self.df['LOG_VALOR'] = np.log1p(self.df['VALOR'])
        
        # Quintiles del valor
        self.df['QUINTIL_VALOR'] = pd.qcut(self.df['VALOR'], 5, labels=False, duplicates='drop')
        
        # Frecuencia de la descripción
        desc_counts = self.df['DESCRIPCION'].value_counts()
        self.df['FRECUENCIA_DESC'] = self.df['DESCRIPCION'].map(desc_counts)
        self.df['FRECUENCIA_DESC'] = self.df['FRECUENCIA_DESC'].fillna(1)
        
        return self.df
    
    def obtener_features_numericas(self):
        """Retorna matriz de características numéricas para clustering."""
        numeric_cols = [col for col in self.df.columns 
                       if col.startswith('TFIDF_') or col in [
                           'VALOR_ESCALADO', 'LOG_VALOR', 'DIA', 'MES', 
                           'DIA_SEMANA', 'TRIMESTRE', 'SEMANA_AÑO', 
                           'ES_FINDE', 'QUINTIL_VALOR', 'FRECUENCIA_DESC'
                       ]]
        return self.df[numeric_cols].values, numeric_cols
    
    def ejecutar_preprocesamiento(self):
        """Ejecuta todo el pipeline de preprocesamiento."""
        print("\n" + "="*70)
        print("FASE 1: PREPROCESAMIENTO Y FEATURE ENGINEERING")
        print("="*70)
        
        self.extraer_caracteristicas_fecha()
        self.vectorizar_descripcion(max_features=50)
        self.escalar_valor()
        self.crear_features_adicionales()
        
        print("[PREP] ✓ Preprocesamiento completado")
        print(f"[PREP] Dimensiones finales: {self.df.shape}")
        
        return self.df


# ============================================================================
# 3. ANÁLISIS NO SUPERVISADO - CLUSTERING
# ============================================================================

class AnalisadorClustering:
    """Clase para ejecutar múltiples algoritmos de clustering."""
    
    def __init__(self, X, nombres_features):
        self.X = X
        self.nombres_features = nombres_features
        self.n_clusters = 3  # Número de clusters a usar
        self.resultados_clustering = {}
        
        # Inicializar arrays para guardar predicciones
        self.kmeans_pred = None
        self.fuzzy_pred = None
        self.subtractive_pred = None
        self.dbscan_pred = None
        self.hierarchical_pred = None
        
    def ejecutar_kmeans(self):
        """Ejecuta K-Means clustering."""
        print("[CLUSTER] Ejecutando K-Means (k=3)...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans_pred = kmeans.fit_predict(self.X)
        self.resultados_clustering['KMeans'] = self.kmeans_pred
        print(f"[CLUSTER]   → Clusters asignados: {np.unique(self.kmeans_pred)}")
        
    def ejecutar_fuzzy_cmeans(self):
        """Ejecuta Fuzzy C-Means clustering."""
        print("[CLUSTER] Ejecutando Fuzzy C-Means (c=3)...")
        try:
            # Fuzzy C-Means - scikit-fuzzy necesita X como (features, samples)
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data=self.X.T,  # Transponer para que sea (features, samples)
                c=self.n_clusters,
                m=2,           # Grado de fuzificación
                error=0.005,
                maxiter=1000,
                init=None,
                seed=42
            )
            # Obtener etiquetas hard (el cluster con mayor pertenencia)
            self.fuzzy_pred = np.argmax(u, axis=0)
            self.resultados_clustering['Fuzzy_CMeans'] = self.fuzzy_pred
            print(f"[CLUSTER]   → Clusters asignados: {np.unique(self.fuzzy_pred)}")
        except Exception as e:
            print(f"[CLUSTER]   ⚠ Error en Fuzzy C-Means: {e}")
            self.fuzzy_pred = np.zeros(self.X.shape[0], dtype=int)
    
    def ejecutar_clustering_sustractivo(self):
        """
        Implementa Subtractive Clustering.
        Basado en el algoritmo descrito por Chiu (1994).
        """
        print("[CLUSTER] Ejecutando Subtractive Clustering...")
        try:
            # Normalizar datos
            X_norm = (self.X - self.X.min(axis=0)) / (self.X.max(axis=0) - self.X.min(axis=0) + 1e-10)
            X_norm = X_norm.astype(np.float64)  # Asegurar que es float64
            
            n_samples = X_norm.shape[0]
            radio_influencia = float(np.percentile(
                np.sqrt(np.sum(X_norm**2, axis=1)), 75
            ) / 2)
            
            # Calcular densidad de cada punto
            densidades = np.zeros(n_samples, dtype=np.float64)
            for i in range(n_samples):
                distancias = np.sqrt(np.sum((X_norm - X_norm[i])**2, axis=1))
                densidades[i] = float(np.sum(np.exp(-(distancias**2 / (radio_influencia**2)))))
            
            # Seleccionar centros de clusters
            indices_ordenados = np.argsort(-densidades)
            centros_indices = []
            min_densidad_centro = float(np.max(densidades) * 0.15)
            
            for idx in indices_ordenados:
                if densidades[idx] > min_densidad_centro:
                    # Verificar que no esté muy cerca de otros centros
                    es_nuevo_centro = True
                    for centro_idx in centros_indices:
                        dist = float(np.sqrt(np.sum((X_norm[idx] - X_norm[centro_idx])**2)))
                        if dist < radio_influencia * 0.5:
                            es_nuevo_centro = False
                            break
                    
                    if es_nuevo_centro:
                        centros_indices.append(idx)
                
                if len(centros_indices) >= self.n_clusters:
                    break
            
            # Asignar cada punto al centro más cercano
            centros = X_norm[centros_indices]
            distancias_a_centros = np.sqrt(
                ((X_norm[:, np.newaxis, :] - centros[np.newaxis, :, :])**2).sum(axis=2)
            )
            self.subtractive_pred = np.argmin(distancias_a_centros, axis=1)
            
            self.resultados_clustering['Subtractive'] = self.subtractive_pred
            print(f"[CLUSTER]   → Clusters encontrados: {len(centros_indices)}")
            print(f"[CLUSTER]   → Clusters asignados: {np.unique(self.subtractive_pred)}")
        
        except Exception as e:
            print(f"[CLUSTER]   ⚠ Error en Subtractive Clustering: {e}")
            self.subtractive_pred = np.zeros(self.X.shape[0], dtype=int)
    
    def ejecutar_dbscan(self):
        """Ejecuta DBSCAN clustering."""
        print("[CLUSTER] Ejecutando DBSCAN...")
        try:
            # Encontrar eps óptimo via k-distancias
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=5)
            neighbors_fit = neighbors.fit(self.X)
            distances, indices = neighbors_fit.kneighbors(self.X)
            distances = np.sort(distances[:, -1], axis=0)
            eps = np.percentile(distances, 90)
            
            dbscan = DBSCAN(eps=eps, min_samples=5)
            self.dbscan_pred = dbscan.fit_predict(self.X)
            
            # Si hay clusters de ruido (-1), reasignarlos al cluster más cercano
            if -1 in self.dbscan_pred:
                ruido_indices = np.where(self.dbscan_pred == -1)[0]
                for idx in ruido_indices:
                    distancias_a_clusters = []
                    for cluster_id in np.unique(self.dbscan_pred):
                        if cluster_id != -1:
                            centroide = self.X[self.dbscan_pred == cluster_id].mean(axis=0)
                            dist = np.sqrt(np.sum((self.X[idx] - centroide)**2))
                            distancias_a_clusters.append((cluster_id, dist))
                    
                    if distancias_a_clusters:
                        cluster_asignado = min(distancias_a_clusters, key=lambda x: x[1])[0]
                        self.dbscan_pred[idx] = cluster_asignado
            
            # Remapear clusters si es necesario para obtener números consecutivos
            cluster_ids = np.unique(self.dbscan_pred)
            cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_ids)}
            self.dbscan_pred = np.array([cluster_mapping[cid] for cid in self.dbscan_pred])
            
            self.resultados_clustering['DBSCAN'] = self.dbscan_pred
            print(f"[CLUSTER]   → Clusters encontrados: {len(np.unique(self.dbscan_pred))}")
            print(f"[CLUSTER]   → Clusters asignados: {np.unique(self.dbscan_pred)}")
        
        except Exception as e:
            print(f"[CLUSTER]   ⚠ Error en DBSCAN: {e}")
            self.dbscan_pred = np.zeros(self.X.shape[0], dtype=int)
    
    def ejecutar_clustering_jerarquico(self):
        """Ejecuta Agglomerative Hierarchical Clustering."""
        print("[CLUSTER] Ejecutando Agglomerative Hierarchical Clustering...")
        try:
            hierarchical = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
            self.hierarchical_pred = hierarchical.fit_predict(self.X)
            self.resultados_clustering['Hierarchical'] = self.hierarchical_pred
            print(f"[CLUSTER]   → Clusters asignados: {np.unique(self.hierarchical_pred)}")
        
        except Exception as e:
            print(f"[CLUSTER]   ⚠ Error en Hierarchical Clustering: {e}")
            self.hierarchical_pred = np.zeros(self.X.shape[0], dtype=int)
    
    def ejecutar_reduccion_dimensionalidad(self):
        """Ejecuta PCA y UMAP para visualización."""
        print("[CLUSTER] Ejecutando reducción de dimensionalidad...")
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        self.X_pca = pca.fit_transform(self.X)
        print(f"[CLUSTER]   → PCA: Varianza explicada = {pca.explained_variance_ratio_.sum():.2%}")
        
        # UMAP
        umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        self.X_umap = umap_model.fit_transform(self.X)
        print(f"[CLUSTER]   → UMAP calculado")
        
        return self.X_pca, self.X_umap, pca
    
    def ejecutar_clustering_completo(self):
        """Ejecuta todo el pipeline de clustering."""
        print("\n" + "="*70)
        print("FASE 2: ANÁLISIS NO SUPERVISADO - CLUSTERING")
        print("="*70)
        
        self.ejecutar_kmeans()
        self.ejecutar_fuzzy_cmeans()
        self.ejecutar_clustering_sustractivo()
        self.ejecutar_dbscan()
        self.ejecutar_clustering_jerarquico()
        
        X_pca, X_umap, pca = self.ejecutar_reduccion_dimensionalidad()
        
        print("[CLUSTER] ✓ Clustering completado")
        
        return {
            'X_pca': X_pca,
            'X_umap': X_umap,
            'pca': pca,
            'resultados': self.resultados_clustering
        }


# ============================================================================
# 4. REEVALUACIÓN DE ETIQUETAS MEDIANTE CONSENSO
# ============================================================================

class RevaluadorEtiquetas:
    """Clase para reevaluar etiquetas basándose en consenso de clustering."""
    
    def __init__(self, df, resultados_clustering):
        self.df = df.copy()
        self.resultados_clustering = resultados_clustering
        self.etiqueta_reevaluada = None
        
    def calcular_consenso(self):
        """
        Calcula consenso de etiquetas basándose en los clusters.
        Lógica: para cada cluster, si la mayoría de transacciones tienen ETIQUETA=1,
        se reasignan todas las de ese cluster a 1 (y viceversa para 0).
        Se usan solo los algoritmos que funcionaron correctamente.
        """
        print("\n" + "="*70)
        print("FASE 3: REEVALUACIÓN DE ETIQUETAS MEDIANTE CONSENSO")
        print("="*70)
        
        # Filtrar algoritmos que tienen predicciones válidas (no todas ceros)
        algoritmos_validos = {}
        for nombre_algo, predicciones in self.resultados_clustering.items():
            n_clusters_unicos = len(np.unique(predicciones))
            # Considerar válido si tiene al menos 2 clusters diferentes
            if n_clusters_unicos >= 2:
                algoritmos_validos[nombre_algo] = predicciones
                print(f"\n[REEVALUACION] {nombre_algo} utilizado (clusters: {n_clusters_unicos})")
            else:
                print(f"\n[REEVALUACION] {nombre_algo} descartado (insuficientes clusters: {n_clusters_unicos})")
        
        if not algoritmos_validos:
            # Si no hay algoritmos válidos, usar todos
            print("[REEVALUACION] ⚠ Ningún algoritmo válido, usando todos")
            algoritmos_validos = self.resultados_clustering
        
        n_samples = len(self.df)
        votos_por_muestra = np.zeros((n_samples, len(algoritmos_validos)))
        
        # Recolectar votos de cada algoritmo de clustering
        for col_idx, (nombre_algo, predicciones) in enumerate(algoritmos_validos.items()):
            print(f"\n[REEVALUACION] Procesando consenso de {nombre_algo}...")
            
            # Para cada cluster, determinar la etiqueta mayoritaria
            etiquetas_por_cluster = {}
            for cluster_id in np.unique(predicciones):
                mask = predicciones == cluster_id
                etiquetas_cluster = self.df[mask]['ETIQUETA'].values
                
                # Determinar etiqueta mayoritaria
                conteo = np.bincount(etiquetas_cluster)
                if len(conteo) == 1:
                    # Solo hay una clase en este cluster
                    etiqueta_mayoritaria = conteo[0] if 0 in np.unique(etiquetas_cluster) else 1
                else:
                    etiqueta_mayoritaria = np.argmax(conteo)
                
                etiquetas_por_cluster[cluster_id] = etiqueta_mayoritaria
                
                n_en_cluster = np.sum(mask)
                n_etiqueta_1 = np.sum(etiquetas_cluster == 1) if len(conteo) > 1 else (0 if etiqueta_mayoritaria == 0 else n_en_cluster)
                print(f"  → Cluster {cluster_id}: {n_en_cluster} muestras, "
                      f"{n_etiqueta_1} etiqueta=1 → Veredicto: {etiqueta_mayoritaria}")
            
            # Asignar votos basados en la etiqueta mayoritaria del cluster
            votos_cluster = np.array([etiquetas_por_cluster[pred] for pred in predicciones])
            votos_por_muestra[:, col_idx] = votos_cluster
        
        # Calcular consenso: mayoría simple entre algoritmos
        # Para cada muestra, contar votos
        votos_promedio = votos_por_muestra.mean(axis=1)
        self.etiqueta_reevaluada = (votos_promedio >= 0.5).astype(int)
        
        # Agregar al dataframe
        self.df['ETIQUETA_REEVALUADA'] = self.etiqueta_reevaluada
        
        # Estadísticas de cambios
        cambios = np.sum(self.df['ETIQUETA'] != self.df['ETIQUETA_REEVALUADA'])
        pct_cambios = 100 * cambios / len(self.df) if len(self.df) > 0 else 0
        
        print(f"\n[REEVALUACION] ✓ Reevaluación completada")
        print(f"[REEVALUACION] Cambios de etiqueta: {cambios} ({pct_cambios:.1f}%)")
        print(f"[REEVALUACION] Distribución original: 0={np.sum(self.df['ETIQUETA']==0)}, "
              f"1={np.sum(self.df['ETIQUETA']==1)}")
        print(f"[REEVALUACION] Distribución reevaluada: 0={np.sum(self.df['ETIQUETA_REEVALUADA']==0)}, "
              f"1={np.sum(self.df['ETIQUETA_REEVALUADA']==1)}")
        
        return self.df


# ============================================================================
# 5. MODELADO SUPERVISADO
# ============================================================================

class ModeladorSupervisado:
    """Clase para entrenar y evaluar modelos supervisados."""
    
    def __init__(self, X, y_original, y_reevaluado):
        self.X = X
        self.y_original = y_original
        self.y_reevaluado = y_reevaluado
        self.resultados = []
        
        # División train/test
        (self.X_train, self.X_test,
         self.y_train_orig, self.y_test_orig,
         self.y_train_reevaluado, self.y_test_reevaluado) = train_test_split(
            self.X,
            self.y_original,
            self.y_reevaluado,
            test_size=0.2,
            random_state=42,
            stratify=self.y_reevaluado  # Estratificar por las reevaluadas para balance
        )
        
        print(f"\n[SUPERVISADO] División train/test realizada:")
        print(f"  → Train: {self.X_train.shape[0]} muestras, "
              f"Test: {self.X_test.shape[0]} muestras")
    
    def entrenar_arbol_decision(self, etiquetas_tipo):
        """Entrena un Árbol de Decisión."""
        print(f"\n[SUPERVISADO] Entrenando Árbol de Decisión ({etiquetas_tipo})...")
        
        y_train = self.y_train_orig if etiquetas_tipo == "Original" else self.y_train_reevaluado
        y_test = self.y_test_orig if etiquetas_tipo == "Original" else self.y_test_reevaluado
        
        dt = DecisionTreeClassifier(max_depth=10, random_state=42, min_samples_split=10)
        dt.fit(self.X_train, y_train)
        
        y_pred = dt.predict(self.X_test)
        metricas = self.calcular_metricas(y_test, y_pred, f"DecisionTree-{etiquetas_tipo}")
        
        return dt, metricas, y_test, y_pred
    
    def entrenar_regresion_logistica(self, etiquetas_tipo):
        """Entrena una Regresión Logística."""
        print(f"[SUPERVISADO] Entrenando Regresión Logística ({etiquetas_tipo})...")
        
        y_train = self.y_train_orig if etiquetas_tipo == "Original" else self.y_train_reevaluado
        y_test = self.y_test_orig if etiquetas_tipo == "Original" else self.y_test_reevaluado
        
        # Validar que hay al menos 2 clases
        if len(np.unique(y_train)) < 2:
            print(f"  ⚠ Skipping: Solo una clase en training ({np.unique(y_train)})")
            # Retornar dummy predictions
            y_pred = np.zeros_like(y_test)
            metricas = self.calcular_metricas(y_test, y_pred, f"LogisticRegression-{etiquetas_tipo}")
            return None, metricas, y_test, y_pred
        
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, y_train)
        
        y_pred = lr.predict(self.X_test)
        metricas = self.calcular_metricas(y_test, y_pred, f"LogisticRegression-{etiquetas_tipo}")
        
        return lr, metricas, y_test, y_pred
    
    def entrenar_regresion_lineal(self, etiquetas_tipo):
        """Entrena una Regresión Lineal con threshold."""
        print(f"[SUPERVISADO] Entrenando Regresión Lineal ({etiquetas_tipo})...")
        
        y_train = self.y_train_orig if etiquetas_tipo == "Original" else self.y_train_reevaluado
        y_test = self.y_test_orig if etiquetas_tipo == "Original" else self.y_test_reevaluado
        
        linreg = LinearRegression()
        linreg.fit(self.X_train, y_train)
        
        # Predicción con threshold en 0.5
        y_pred_continu = linreg.predict(self.X_test)
        y_pred = (y_pred_continu >= 0.5).astype(int)
        
        metricas = self.calcular_metricas(y_test, y_pred, f"LinearRegression-{etiquetas_tipo}")
        
        return linreg, metricas, y_test, y_pred
    
    def calcular_metricas(self, y_true, y_pred, nombre_modelo):
        """Calcula métricas de desempeño."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
        except:
            roc_auc = 0.0
        
        metricas_dict = {
            'Modelo': nombre_modelo,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        
        self.resultados.append(metricas_dict)
        
        print(f"  → Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
              f"Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return metricas_dict
    
    def entrenar_modelos_completo(self):
        """Entrena todos los modelos en ambos escenarios."""
        print("\n" + "="*70)
        print("FASE 4: MODELADO SUPERVISADO")
        print("="*70)
        
        modelos_dict = {}
        
        # Escenario A: Etiquetas Originales
        print("\n[SUPERVISADO] ═══ ESCENARIO A: Etiquetas Originales ═══")
        dt_orig, metricas_dt_orig, y_test_orig, y_pred_dt_orig = self.entrenar_arbol_decision("Original")
        lr_orig, metricas_lr_orig, y_test_orig, y_pred_lr_orig = self.entrenar_regresion_logistica("Original")
        linreg_orig, metricas_linreg_orig, y_test_orig, y_pred_linreg_orig = self.entrenar_regresion_lineal("Original")
        
        modelos_dict['Original'] = {
            'DecisionTree': (dt_orig, y_test_orig, y_pred_dt_orig),
            'LogisticRegression': (lr_orig, y_test_orig, y_pred_lr_orig),
            'LinearRegression': (linreg_orig, y_test_orig, y_pred_linreg_orig)
        }
        
        # Escenario B: Etiquetas Reevaluadas
        print("\n[SUPERVISADO] ═══ ESCENARIO B: Etiquetas Reevaluadas ═══")
        dt_reevaluado, metricas_dt_reevaluado, y_test_reevaluado, y_pred_dt_reevaluado = self.entrenar_arbol_decision("Reevaluada")
        lr_reevaluado, metricas_lr_reevaluado, y_test_reevaluado, y_pred_lr_reevaluado = self.entrenar_regresion_logistica("Reevaluada")
        linreg_reevaluado, metricas_linreg_reevaluado, y_test_reevaluado, y_pred_linreg_reevaluado = self.entrenar_regresion_lineal("Reevaluada")
        
        modelos_dict['Reevaluada'] = {
            'DecisionTree': (dt_reevaluado, y_test_reevaluado, y_pred_dt_reevaluado),
            'LogisticRegression': (lr_reevaluado, y_test_reevaluado, y_pred_lr_reevaluado),
            'LinearRegression': (linreg_reevaluado, y_test_reevaluado, y_pred_linreg_reevaluado)
        }
        
        print("\n[SUPERVISADO] ✓ Entrenamiento completado")
        
        return modelos_dict
    
    def obtener_dataframe_resultados(self):
        """Retorna un DataFrame con los resultados."""
        return pd.DataFrame(self.resultados)


# ============================================================================
# 6. GENERACIÓN DE GRÁFICAS Y VISUALIZACIONES
# ============================================================================

class GeneradorVisualizaciones:
    """Clase para generar todas las visualizaciones y reportes."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def graficar_clustering_pca(self, X_pca, etiqueta_original):
        """Grafica PCA sin clustering."""
        print("[VIZ] Generando gráfica PCA sin clusters...")
        
        fig = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=etiqueta_original, 
                             cmap='RdYlBu', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.xlabel(f'PC1', fontsize=12)
        plt.ylabel(f'PC2', fontsize=12)
        plt.title('PCA - Etiquetas Originales', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Etiqueta (0=No hormiga, 1=Hormiga)')
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_pca_original.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def graficar_clustering_multiples(self, X_pca, X_umap, resultados_clustering):
        """Grafica PCA y UMAP con clusters de múltiples algoritmos."""
        print("[VIZ] Generando gráficas multi-algoritmo...")
        
        algoritmos = list(resultados_clustering.keys())
        n_algo = len(algoritmos)
        
        # Crear subplots con PCA
        fig_pca = plt.figure(figsize=(20, 4 * (n_algo // 2 + n_algo % 2)))
        for idx, (nombre_algo, predicciones) in enumerate(resultados_clustering.items(), 1):
            ax = plt.subplot(n_algo // 2 + n_algo % 2, 2, idx)
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=predicciones, 
                               cmap='Set3', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('PC1', fontsize=10)
            ax.set_ylabel('PC2', fontsize=10)
            ax.set_title(f'PCA - {nombre_algo}', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_clustering_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Crear subplots con UMAP
        fig_umap = plt.figure(figsize=(20, 4 * (n_algo // 2 + n_algo % 2)))
        for idx, (nombre_algo, predicciones) in enumerate(resultados_clustering.items(), 1):
            ax = plt.subplot(n_algo // 2 + n_algo % 2, 2, idx)
            scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=predicciones, 
                               cmap='Set3', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('UMAP 1', fontsize=10)
            ax.set_ylabel('UMAP 2', fontsize=10)
            ax.set_title(f'UMAP - {nombre_algo}', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_clustering_umap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def graficar_matrices_confusion(self, modelos_dict):
        """Grafica matrices de confusión para todos los modelos."""
        print("[VIZ] Generando matrices de confusión...")
        
        for escenario, modelos in modelos_dict.items():
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            for idx, (nombre_modelo, (modelo, y_test, y_pred)) in enumerate(modelos.items()):
                cm = confusion_matrix(y_test, y_pred)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                           cbar_kws={'label': 'Cantidad'})
                axes[idx].set_ylabel('Verdadero', fontsize=11)
                axes[idx].set_xlabel('Predicho', fontsize=11)
                axes[idx].set_title(f'{nombre_modelo}\n{escenario}', fontsize=12, fontweight='bold')
                axes[idx].set_xticklabels(['No Hormiga', 'Hormiga'])
                axes[idx].set_yticklabels(['No Hormiga', 'Hormiga'])
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'04_confusion_matrix_{escenario.lower()}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
    def graficar_importancia_variables(self, dt_modelo, nombres_features):
        """Grafica importancia de variables del Árbol de Decisión."""
        print("[VIZ] Generando gráfica de importancia de variables...")
        
        importancias = dt_modelo.feature_importances_
        indices_ordenados = np.argsort(importancias)[-15:]  # Top 15
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(np.array(nombres_features)[indices_ordenados], importancias[indices_ordenados],
               color='steelblue', edgecolor='black')
        ax.set_xlabel('Importancia', fontsize=12)
        ax.set_title('Feature Importance - Árbol de Decisión\n(Etiquetas Reevaluadas)', 
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def graficar_arbol_decision(self, dt_modelo, nombres_features):
        """Grafica la estructura del Árbol de Decisión."""
        print("[VIZ] Generando visualización del Árbol de Decisión...")
        
        fig, ax = plt.subplots(figsize=(25, 15))
        plot_tree(dt_modelo, feature_names=nombres_features, class_names=['No Hormiga', 'Hormiga'],
                 filled=True, ax=ax, fontsize=10)
        plt.tight_layout()
        plt.savefig(self.output_dir / '06_arbol_decision_estructura.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # Exportar como texto
        tree_rules = export_text(dt_modelo, feature_names=nombres_features)
        with open(self.output_dir / '07_arbol_decision_reglas.txt', 'w', encoding='utf-8') as f:
            f.write("ESTRUCTURA DEL ÁRBOL DE DECISIÓN (Etiquetas Reevaluadas)\n")
            f.write("="*80 + "\n\n")
            f.write(tree_rules)


# ============================================================================
# 7. GENERADOR DE REPORTES
# ============================================================================

class GeneradorReportes:
    """Clase para generar reportes en formato CSV y texto."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        
    def generar_reporte_metricas(self, df_resultados):
        """Genera reporte comparativo de métricas."""
        print("[REPORTE] Generando reporte de métricas...")
        
        df_resultados.to_csv(self.output_dir / '08_metricas_comparativas.csv', index=False)
        
        # Crear resumen por tipo de etiqueta
        print(df_resultados.to_string(index=False))
        
        # Análisis de impacto del reetiquetado
        resumen_texto = """
================================================================================
ANÁLISIS COMPARATIVO: ETIQUETAS ORIGINALES vs REEVALUADAS
================================================================================

Se comparan los modelos entrenados en dos escenarios:
  - Escenario A: ETIQUETA original (potencialmente con ruido)
  - Escenario B: ETIQUETA_REEVALUADA (consenso de clustering)

Metodología:
  1. Los algoritmos de clustering agruparon automáticamente las transacciones
  2. Para cada cluster, se determinó la etiqueta mayoritaria
  3. Las nuevas etiquetas se asignaron por consenso de múltiples algoritmos
  
Impacto esperado:
  - Mejora en consistencia de clasificaciones
  - Reducción de ruido en las etiquetas
  - Mejor generalización de los modelos supervisados
  
================================================================================
"""
        
        with open(self.output_dir / '09_analisis_impacto.txt', 'w', encoding='utf-8') as f:
            f.write(resumen_texto)
            f.write("\n\nRESULTADOS DETALLADOS:\n")
            f.write("="*80 + "\n")
            f.write(df_resultados.to_string(index=False))
    
    def generar_reporte_general(self, df_original, df_con_reevaluadas):
        """Genera reporte general del análisis."""
        print("[REPORTE] Generando reporte general...")
        
        reporte = f"""
================================================================================
REPORTE GENERAL: DETECCIÓN DE GASTOS HORMIGA
================================================================================

Fecha de Ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DATASET
───────────
   • Total de registros: {len(df_original):,}
   • Características: FECHA, DESCRIPCION, VALOR, ETIQUETA
   • Distribución original: 
     - No Hormiga (0): {(df_original['ETIQUETA']==0).sum()} transacciones ({100*(df_original['ETIQUETA']==0).sum()/len(df_original):.1f}%)
     - Hormiga (1): {(df_original['ETIQUETA']==1).sum()} transacciones ({100*(df_original['ETIQUETA']==1).sum()/len(df_original):.1f}%)
   
   • Valor promedio: ${df_original['VALOR'].mean():,.0f}
   • Valor mínimo: ${df_original['VALOR'].min():,.0f}
   • Valor máximo: ${df_original['VALOR'].max():,.0f}

2. PREPROCESSING
────────────────
   • Extracción de características temporales: 7 variables
   • Vectorización de descripciones: 50 características TF-IDF
   • Escalado de valores: StandardScaler
   • Características adicionales: Log(VALOR), Quintiles, Frecuencias
   • Total de features: {len([c for c in df_con_reevaluadas.columns if c.startswith('TFIDF_') or c in ['VALOR_ESCALADO', 'LOG_VALOR', 'DIA', 'MES', 'DIA_SEMANA', 'TRIMESTRE', 'SEMANA_AÑO', 'ES_FINDE', 'QUINTIL_VALOR', 'FRECUENCIA_DESC']])}

3. ANÁLISIS NO SUPERVISADO (CLUSTERING)
────────────────────────────────────────
   Algoritmos aplicados:
   • K-Means (k=3)
   • Fuzzy C-Means (c=3)
   • Subtractive Clustering
   • DBSCAN (eps adaptativo)
   • Agglomerative Clustering (linkage=ward, k=3)
   
   Técnicas de visualización:
   • PCA (2 componentes)
   • UMAP (2 componentes)

4. REEVALUACIÓN DE ETIQUETAS
──────────────────────────────
   Metodología: Consenso de clustering
   
   Cambios realizados: {(df_original['ETIQUETA'] != df_con_reevaluadas['ETIQUETA_REEVALUADA']).sum()} transacciones reasignadas
   
   Distribución reevaluada:
   - No Hormiga (0): {(df_con_reevaluadas['ETIQUETA_REEVALUADA']==0).sum()} transacciones ({100*(df_con_reevaluadas['ETIQUETA_REEVALUADA']==0).sum()/len(df_con_reevaluadas):.1f}%)
   - Hormiga (1): {(df_con_reevaluadas['ETIQUETA_REEVALUADA']==1).sum()} transacciones ({100*(df_con_reevaluadas['ETIQUETA_REEVALUADA']==1).sum()/len(df_con_reevaluadas):.1f}%)

5. MODELADO SUPERVISADO
────────────────────────
   Escenario A: Etiquetas Originales
   Escenario B: Etiquetas Reevaluadas
   
   Modelos entrenados:
   • Árbol de Decisión (max_depth=10)
   • Regresión Logística (max_iter=1000)
   • Regresión Lineal (threshold=0.5)
   
   División: 80% entrenamiento, 20% prueba (estratificado)

6. ARTEFACTOS GENERADOS
────────────────────────
   Carpeta: Outputs/
   
   Gráficas:
   • 01_pca_original.png - Visualización PCA con etiquetas originales
   • 02_clustering_pca.png - Clustering en espacio PCA
   • 03_clustering_umap.png - Clustering en espacio UMAP
   • 04_confusion_matrix_original.png - Matrices de confusión (escenario A)
   • 04_confusion_matrix_reevaluada.png - Matrices de confusión (escenario B)
   • 05_feature_importance.png - Importancia de variables
   • 06_arbol_decision_estructura.png - Visualización del árbol
   
   Reportes:
   • 08_metricas_comparativas.csv - Métricas de desempeño
   • 09_analisis_impacto.txt - Análisis del impacto del reetiquetado
   • 07_arbol_decision_reglas.txt - Reglas del árbol en formato texto
   • 10_reporte_general.txt - Este archivo

================================================================================
FIN DEL REPORTE
================================================================================
"""
        
        with open(self.output_dir / '10_reporte_general.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print(reporte)


# ============================================================================
# 8. PIPELINE PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta el pipeline completo."""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "DETECCIÓN DE GASTOS HORMIGA - ENFOQUE HÍBRIDO".center(78) + "║")
    print("║" + "Aprendizaje No Supervisado + Supervisado".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # Definir ruta del dataset
        ruta_dataset = Path("Informe 2/Dataset_Hormiga_Binario.csv")
        
        if not ruta_dataset.exists():
            print(f"❌ Error: No se encontró el archivo {ruta_dataset}")
            return
        
        # Crear directorio de salida
        output_dir = Path("Outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ════════════════════════════════════════════════════════════════════
        # FASE 1: PREPROCESAMIENTO
        # ════════════════════════════════════════════════════════════════════
        df = pd.read_csv(ruta_dataset)
        
        preparador = PreprocesadorDatos(df)
        df_procesado = preparador.ejecutar_preprocesamiento()
        
        # Obtener features numéricas
        X, nombres_features = preparador.obtener_features_numericas()
        
        # ════════════════════════════════════════════════════════════════════
        # FASE 2: CLUSTERING
        # ════════════════════════════════════════════════════════════════════
        analizador = AnalisadorClustering(X, nombres_features)
        clustering_resultados = analizador.ejecutar_clustering_completo()
        
        # ════════════════════════════════════════════════════════════════════
        # FASE 3: REEVALUACIÓN DE ETIQUETAS
        # ════════════════════════════════════════════════════════════════════
        reevaluador = RevaluadorEtiquetas(df_procesado, clustering_resultados['resultados'])
        df_reevaluado = reevaluador.calcular_consenso()
        
        # ════════════════════════════════════════════════════════════════════
        # FASE 4: MODELADO SUPERVISADO
        # ════════════════════════════════════════════════════════════════════
        modelador = ModeladorSupervisado(
            X,
            df_reevaluado['ETIQUETA'].values,
            df_reevaluado['ETIQUETA_REEVALUADA'].values
        )
        modelos_dict = modelador.entrenar_modelos_completo()
        
        # ════════════════════════════════════════════════════════════════════
        # FASE 5: GENERACIÓN DE VISUALIZACIONES
        # ════════════════════════════════════════════════════════════════════
        print("\n" + "="*70)
        print("FASE 5: GENERACIÓN DE VISUALIZACIONES Y REPORTES")
        print("="*70)
        
        visualizador = GeneradorVisualizaciones(output_dir)
        
        # Obtener predicciones del árbol entrenado
        dt_reevaluado = modelos_dict['Reevaluada']['DecisionTree'][0]
        
        visualizador.graficar_clustering_pca(
            clustering_resultados['X_pca'],
            df_reevaluado['ETIQUETA'].values
        )
        
        visualizador.graficar_clustering_multiples(
            clustering_resultados['X_pca'],
            clustering_resultados['X_umap'],
            clustering_resultados['resultados']
        )
        
        visualizador.graficar_matrices_confusion(modelos_dict)
        visualizador.graficar_importancia_variables(dt_reevaluado, nombres_features)
        visualizador.graficar_arbol_decision(dt_reevaluado, nombres_features)
        
        # ════════════════════════════════════════════════════════════════════
        # FASE 6: GENERACIÓN DE REPORTES
        # ════════════════════════════════════════════════════════════════════
        reportero = GeneradorReportes(output_dir)
        
        df_metricas = modelador.obtener_dataframe_resultados()
        reportero.generar_reporte_metricas(df_metricas)
        reportero.generar_reporte_general(df_procesado, df_reevaluado)
        
        # Guardar dataset procesado
        df_reevaluado.to_csv(output_dir / 'dataset_con_reevaluadas.csv', index=False)
        print("\n[GUARDADO] Dataset con etiquetas reevaluadas: "
              f"Outputs/dataset_con_reevaluadas.csv")
        
        # ════════════════════════════════════════════════════════════════════
        # FINALIZACIÓN
        # ════════════════════════════════════════════════════════════════════
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"\nTodos los artefactos se encuentran en la carpeta: {output_dir.absolute()}")
        print("\nArchivos generados:")
        for i, archivo in enumerate(sorted(output_dir.glob('*')), 1):
            if archivo.is_file():
                tamaño = archivo.stat().st_size
                if tamaño > 1024*1024:
                    tamaño_str = f"{tamaño/(1024*1024):.1f} MB"
                elif tamaño > 1024:
                    tamaño_str = f"{tamaño/1024:.1f} KB"
                else:
                    tamaño_str = f"{tamaño} B"
                print(f"  {i:2d}. {archivo.name:<50s} ({tamaño_str})")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
