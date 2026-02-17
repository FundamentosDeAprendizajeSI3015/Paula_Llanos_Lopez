from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


def train_model(df):
    """Entrena el modelo de clasificación de gasto hormiga.

    Separa en train/validation/test, construye el pipeline de
    preprocesamiento (numérico + texto) y entrena un RandomForest.

    Returns:
        model: Pipeline entrenado.
        X_val, y_val: conjunto de validación.
        X_test, y_test: conjunto de prueba.
    """

    TARGET = "es_gasto_hormiga"

    features_num = [
        "valor_abs",
        "dia_semana",
        "mes",
        "dia_mes",
        "es_fin_semana",
        "freq_gastos_dia",
        "gasto_diario_acumulado",
        "freq_por_descripcion",
        "gasto_promedio_descripcion",
    ]

    feature_text = "descripcion"

    X = df[features_num + [feature_text]]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    text_transformer = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2)))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, features_num),
            ("txt", text_transformer, feature_text),
        ]
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    return model, X_val, y_val, X_test, y_test
