import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def _ensure_dir(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_basic_eda(df, output_dir="../reports"):
    """Gráficas básicas del dataset (distribución de valores y etiquetas)."""

    output_dir = _ensure_dir(output_dir)

    # Histograma de montos absolutos
    plt.figure(figsize=(8, 5))
    sns.histplot(df["valor_abs"], bins=40, kde=False)
    plt.title("Distribución de montos (valor_abs)")
    plt.xlabel("Monto absoluto")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_01_hist_valor_abs.png"))
    plt.close()

    if "es_gasto_hormiga" in df.columns:
        # Conteo de clases
        plt.figure(figsize=(5, 4))
        sns.countplot(x="es_gasto_hormiga", data=df)
        plt.title("Distribución de la etiqueta es_gasto_hormiga")
        plt.xlabel("Es gasto hormiga (0 = No, 1 = Sí)")
        plt.ylabel("Conteo")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_02_label_distribution.png"))
        plt.close()

        # Boxplot de montos por etiqueta
        plt.figure(figsize=(7, 5))
        sns.boxplot(x="es_gasto_hormiga", y="valor_abs", data=df)
        plt.title("Distribución de montos por etiqueta")
        plt.xlabel("Es gasto hormiga (0 = No, 1 = Sí)")
        plt.ylabel("Monto absoluto")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_03_box_valor_por_label.png"))
        plt.close()

        # Versión "zoom" del boxplot, centrada en montos pequeños (ej. <= 100000)
        df_small = df[df["valor_abs"] <= 100000]
        if not df_small.empty:
            plt.figure(figsize=(7, 5))
            sns.boxplot(x="es_gasto_hormiga", y="valor_abs", data=df_small)
            plt.title("Distribución de montos por etiqueta (zoom montos pequeños)")
            plt.xlabel("Es gasto hormiga (0 = No, 1 = Sí)")
            plt.ylabel("Monto absoluto (<= 100000)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "eda_04_box_valor_por_label_zoom.png"))
            plt.close()


def plot_top_comercios(df, top_n=10, output_dir="../reports"):
    """Top comercios más frecuentes, diferenciando hormiga vs no hormiga."""

    output_dir = _ensure_dir(output_dir)

    if "es_gasto_hormiga" not in df.columns:
        return

    # Agrupar por descripción y etiqueta
    counts = (
        df.groupby(["descripcion", "es_gasto_hormiga"])["valor"]
        .size()
        .reset_index(name="conteo")
    )

    # Obtener top N descripciones por conteo total
    top_desc = (
        counts.groupby("descripcion")["conteo"].sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    counts_top = counts[counts["descripcion"].isin(top_desc)]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=counts_top,
        x="descripcion",
        y="conteo",
        hue="es_gasto_hormiga",
    )
    plt.title("Top comercios por número de transacciones")
    plt.xlabel("Descripción del comercio")
    plt.ylabel("Conteo de transacciones")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Es gasto hormiga")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_05_top_comercios.png"))
    plt.close()


def plot_time_series_gasto(df, output_dir="../reports"):
    """Serie temporal del gasto diario hormiga vs no hormiga."""

    output_dir = _ensure_dir(output_dir)

    if "fecha_dia" not in df.columns or "es_gasto_hormiga" not in df.columns:
        return

    daily = (
        df.groupby(["fecha_dia", "es_gasto_hormiga"])["valor_abs"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(10, 5))
    for label_value, label_name in [(0, "No hormiga"), (1, "Hormiga")]:
        subset = daily[daily["es_gasto_hormiga"] == label_value]
        if not subset.empty:
            plt.plot(subset["fecha_dia"], subset["valor_abs"], label=label_name)

    plt.title("Evolución del gasto diario (hormiga vs no hormiga)")
    plt.xlabel("Fecha")
    plt.ylabel("Gasto diario absoluto")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_06_serie_gasto_diario.png"))
    plt.close()


def plot_scatter_freq_vs_monto(df, output_dir="../reports"):
    """Relación entre frecuencia por comercio y gasto promedio, coloreado por etiqueta."""

    output_dir = _ensure_dir(output_dir)

    cols_needed = {"freq_por_descripcion", "gasto_promedio_descripcion", "es_gasto_hormiga"}
    if not cols_needed.issubset(df.columns):
        return

    # Muestreamos un poco si hay muchos puntos
    plot_df = df[["freq_por_descripcion", "gasto_promedio_descripcion", "es_gasto_hormiga"]].dropna()
    if len(plot_df) > 2000:
        plot_df = plot_df.sample(2000, random_state=42)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=plot_df,
        x="freq_por_descripcion",
        y="gasto_promedio_descripcion",
        hue="es_gasto_hormiga",
        alpha=0.6,
    )
    plt.title("Frecuencia vs gasto promedio por comercio")
    plt.xlabel("Frecuencia por descripción")
    plt.ylabel("Gasto promedio absoluto")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_07_scatter_freq_vs_gasto_promedio.png"))
    plt.close()


def plot_corr_heatmap(df, output_dir="../reports"):
    """Matriz de correlación de las principales variables numéricas."""

    output_dir = _ensure_dir(output_dir)

    num_cols = [
        c
        for c in [
            "valor_abs",
            "dia_semana",
            "mes",
            "dia_mes",
            "es_fin_semana",
            "freq_gastos_dia",
            "gasto_diario_acumulado",
            "freq_por_descripcion",
            "gasto_promedio_descripcion",
            "es_gasto_hormiga",
        ]
        if c in df.columns
    ]

    if len(num_cols) < 2:
        return

    corr = df[num_cols].corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Matriz de correlación de variables numéricas")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_08_corr_heatmap.png"))
    plt.close()


def plot_confusion_and_roc(y_true, y_pred, y_proba=None, name="MODEL", output_dir="../reports"):
    """Guarda matriz de confusión y, si hay probabilidades, curva ROC."""

    output_dir = _ensure_dir(output_dir)

    # Matriz de confusión
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax)
    ax.set_title(f"Matriz de confusión - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"model_{name}_confusion_matrix.png"))
    plt.close(fig)

    # Curva ROC (si hay probabilidades)
    if y_proba is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
        ax.set_title(f"Curva ROC - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"model_{name}_roc_curve.png"))
        plt.close(fig)
