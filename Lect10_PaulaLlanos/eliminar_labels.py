import pandas as pd
import os

# Ruta de los datasets
proyecto_dir = r"c:\Users\llano\Desktop\EAFIT\FundamentosApren\Github\Paula_Llanos_Lopez\Lect10_PaulaLlanos"

# Cargar los datasets
df_realista = pd.read_csv(os.path.join(proyecto_dir, "dataset_sintetico_FIRE_UdeA_realista.csv"))
df_sintetico = pd.read_csv(os.path.join(proyecto_dir, "dataset_sintetico_FIRE_UdeA.csv"))

print("Dataset Realista - Columnas originales:")
print(df_realista.columns.tolist())
print(f"Shape: {df_realista.shape}\n")

print("Dataset Sintético - Columnas originales:")
print(df_sintetico.columns.tolist())
print(f"Shape: {df_sintetico.shape}\n")

# Eliminar la columna 'label' de ambos datasets
df_realista_sin_label = df_realista.drop(columns=['label'])
df_sintetico_sin_label = df_sintetico.drop(columns=['label'])

print("Dataset Realista - Columnas después de eliminar label:")
print(df_realista_sin_label.columns.tolist())
print(f"Shape: {df_realista_sin_label.shape}\n")

print("Dataset Sintético - Columnas después de eliminar label:")
print(df_sintetico_sin_label.columns.tolist())
print(f"Shape: {df_sintetico_sin_label.shape}\n")

# Guardar los nuevos datasets sin la columna label
df_realista_sin_label.to_csv(
    os.path.join(proyecto_dir, "dataset_sintetico_FIRE_UdeA_realista_sin_label.csv"),
    index=False
)
df_sintetico_sin_label.to_csv(
    os.path.join(proyecto_dir, "dataset_sintetico_FIRE_UdeA_sin_label.csv"),
    index=False
)

print("✓ Datasets sin etiquetas guardados correctamente:")
print(f"  - dataset_sintetico_FIRE_UdeA_realista_sin_label.csv")
print(f"  - dataset_sintetico_FIRE_UdeA_sin_label.csv")
