import os
import joblib

from data_loader import load_data
from preprocessing import filter_expenses, create_labels, DEFAULT_THRESHOLD
from feature_engineering import create_features
from train import train_model
from evaluate import evaluate
from visualizations import (
    plot_basic_eda,
    plot_top_comercios,
    plot_time_series_gasto,
    plot_scatter_freq_vs_monto,
    plot_corr_heatmap,
)


def main():
    """Ejecuta de principio a fin el pipeline de gastos hormiga.

    Carga datos, filtra gastos, crea etiquetas y features,
    genera análisis visual, entrena el modelo, evalúa y guarda
    el modelo entrenado y los reportes.
    """

    print(" Cargando datos...")
    df = load_data("../data/Dataset_Hormiga.csv")

    print(" Filtrando gastos...")
    df = filter_expenses(df)

    print(" Creando etiquetas...")
    df = create_labels(df, threshold=DEFAULT_THRESHOLD)

    print(" Generando features...")
    df = create_features(df)

    print(" Generando análisis visual básico del dataset...")
    plot_basic_eda(df, output_dir="../reports")
    plot_top_comercios(df, output_dir="../reports")
    plot_time_series_gasto(df, output_dir="../reports")
    plot_scatter_freq_vs_monto(df, output_dir="../reports")
    plot_corr_heatmap(df, output_dir="../reports")

    print(" Entrenando modelo...")
    model, X_val, y_val, X_test, y_test = train_model(df)

    print(" Evaluando en validación...")
    evaluate(model, X_val, y_val, name="VALIDATION", output_dir="../reports")

    print(" Evaluando en test...")
    evaluate(model, X_test, y_test, name="TEST", output_dir="../reports")
    
    print(" Pipeline completado correctamente.")


if __name__ == "__main__":
    main()
