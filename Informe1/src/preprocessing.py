DEFAULT_THRESHOLD = 25000


def filter_expenses(df):
    """Devuelve solo las transacciones que son salidas de dinero (valor < 0)."""
    return df[df["valor"] < 0].copy()


def create_labels(df, threshold=DEFAULT_THRESHOLD):
    """Crea la columna es_gasto_hormiga usando reglas de negocio y un umbral de monto.

    Args:
        df: DataFrame de transacciones ya filtrado y limpiado.
        threshold: Monto máximo para considerar un gasto como posible hormiga.

    Returns:
        DataFrame con la nueva columna booleana/escalar es_gasto_hormiga.
    """

    # Gastos que NO se consideran hormiga (gastos fijos / necesarios o movimientos bancarios genéricos)
    no_hormiga_keywords = [
        # Fijos
        "arriendo", "matricula", "cuota", "seguro",
        "prestamo", "impuesto", "eps", "soat",
        # Movimientos bancarios genéricos / no comercio
        "transferencia", "transf", "nequi", "consignacion", "corresponsal",
        "retiro cajero", "retiro corresponsal", "abono intereses", "ajuste interes",
        "saldo", "cta suc virtual", "cta suc", "ctasuc",
        "cuota manejo", "dev cuota manejo",
        # Transporte
        "transporte", "civica", "recarga de tarjeta civica", "metro", "bus", "taxi",
        # Almuerzos / comidas principales (ejemplos típicos, pero dejamos fuera
        # recanto y universida para que el modelo los decida según monto/frecuencia)
        "frisby", "home food", "bigos", "taco facto", "dogger",
        "chipstatio", "presto", "mcdonald", "dunkin", "restaurante",
        # Farmacias / salud
        "farmacia", "droguer", "comsocial ips"
    ]

    # Gastos típicamente pequeños y frecuentes que sí se pueden considerar hormiga
    hormiga_keywords = [
        "cafe", "tinto", "snack", "pan", "gaseosa",
        "ara", "oxxo",
    ]

    def label(row):
        v = abs(row["valor"])
        desc = row["descripcion"]

        # Si coincide con alguna palabra clave de gastos NO hormiga, se fuerza a 0
        if any(k in desc for k in no_hormiga_keywords):
            return 0

        if v <= threshold and any(k in desc for k in hormiga_keywords):
            return 1

        if v <= threshold:
            return 1

        return 0

    df["es_gasto_hormiga"] = df.apply(label, axis=1)

    return df
