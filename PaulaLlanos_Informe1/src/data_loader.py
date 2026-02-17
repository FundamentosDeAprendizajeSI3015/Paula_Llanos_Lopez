import pandas as pd
import numpy as np


def parse_money(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("$", "").replace(" ", "")
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_data(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    # Eliminar columnas vacías
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    df.columns = [c.strip().lower() for c in df.columns]

    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    df["valor"] = df["valor"].apply(parse_money)
    df["descripcion"] = df["descripcion"].astype(str).str.lower().str.strip()

    return df
