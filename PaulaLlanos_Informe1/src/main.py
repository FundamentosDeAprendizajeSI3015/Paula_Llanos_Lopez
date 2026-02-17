import os

from data_loader import load_data
from preprocessing import filter_expenses, create_labels, DEFAULT_THRESHOLD
from feature_engineering import create_features
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
    y genera un análisis visual descriptivo del dataset.
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
    
    print(" Pipeline completado correctamente.")


if __name__ == "__main__":
    main()
