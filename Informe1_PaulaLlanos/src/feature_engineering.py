import pandas as pd


def create_features(df):
    """Genera variables numéricas derivadas de fecha, frecuencia y comercio.

    Añade columnas como valor_abs, día de la semana, frecuencia diaria y
    frecuencia/promedio por descripción.
    """

    df["valor_abs"] = df["valor"].abs()

    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.month
    df["dia_mes"] = df["fecha"].dt.day
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)

    df["fecha_dia"] = df["fecha"].dt.date

    # Frecuencia por día
    freq = df.groupby("fecha_dia")["valor"].size().rename("freq_gastos_dia")
    df = df.merge(freq, on="fecha_dia", how="left")

    # Gasto diario acumulado
    suma = df.groupby("fecha_dia")["valor_abs"].sum().rename("gasto_diario_acumulado")
    df = df.merge(suma, on="fecha_dia", how="left")

    # === Features de frecuencia por comercio (descripcion) ===
    # Cuántas veces se repite exactamente la misma descripción
    df["freq_por_descripcion"] = df.groupby("descripcion")["valor"].transform("size")

    # Gasto absoluto promedio por descripción (opcionalmente útil)
    df["gasto_promedio_descripcion"] = (
        df.groupby("descripcion")["valor_abs"].transform("mean")
    )

    return df
