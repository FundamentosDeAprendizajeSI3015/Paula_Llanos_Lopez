from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from visualizations import plot_confusion_and_roc


def evaluate(model, X, y, name="MODEL", output_dir="../reports"):
    """Evalúa el modelo e imprime métricas de clasificación.

    Además, genera las gráficas de matriz de confusión y curva ROC
    y las guarda en la carpeta de reportes indicada.
    """

    preds = model.predict(X)

    print(f"\n===== {name} =====")
    print(classification_report(y, preds))

    cm = confusion_matrix(y, preds)
    print("\nConfusion Matrix:\n", cm)

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        print("ROC-AUC:", round(auc, 4))

    # Guardar gráficas asociadas a esta evaluación
    plot_confusion_and_roc(y, preds, y_proba=proba, name=name, output_dir=output_dir)
