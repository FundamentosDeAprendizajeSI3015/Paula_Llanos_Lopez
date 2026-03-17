import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def estimate_elbow(k_values: list[int], inertias: list[float]) -> int:
	"""Estimate elbow point using max distance to line method."""
	x = np.array(k_values)
	y = np.array(inertias)

	p1 = np.array([x[0], y[0]])
	p2 = np.array([x[-1], y[-1]])

	distances = []
	for i in range(len(x)):
		p = np.array([x[i], y[i]])
		dist = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
		distances.append(dist)

	return int(x[int(np.argmax(distances))])


def main() -> None:
	random_state = 42
	plt.rc("font", family="serif", size=12)
	output_dir = pathlib.Path(__file__).with_name("Graficas")
	output_dir.mkdir(parents=True, exist_ok=True)

	data_path = pathlib.Path(__file__).with_name("dataset_sintetico_FIRE_UdeA_realista.csv")
	df = pd.read_csv(data_path)

	feature_cols = [c for c in df.columns if c != "label"]
	X = df[feature_cols].copy()

	numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
	categorical_features = [c for c in X.columns if c not in numeric_features]

	preprocessor = ColumnTransformer(
		transformers=[
			(
				"num",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="median")),
						("scaler", StandardScaler()),
					]
				),
				numeric_features,
			),
			(
				"cat",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="most_frequent")),
						("onehot", OneHotEncoder(handle_unknown="ignore")),
					]
				),
				categorical_features,
			),
		],
		remainder="drop",
	)

	x_plot = "liquidez" if "liquidez" in X.columns else numeric_features[0]
	y_plot = "dias_efectivo" if "dias_efectivo" in X.columns else numeric_features[1]

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.scatter(X[x_plot], X[y_plot], alpha=0.75)
	ax.set_title("Dataset sintetico realista: distribucion inicial")
	ax.set_xlabel(x_plot)
	ax.set_ylabel(y_plot)
	fig.savefig(output_dir / "realista_distribucion_inicial.png", dpi=300, bbox_inches="tight")

	clu_kmeans_2 = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("clustering", KMeans(n_clusters=2, random_state=random_state, n_init=10)),
		]
	)
	clu_kmeans_2.fit(X)
	print(f"Con K = 2: inercia = {clu_kmeans_2['clustering'].inertia_:.4f}")

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.scatter(X[x_plot], X[y_plot], c=clu_kmeans_2["clustering"].labels_, cmap="tab10", alpha=0.85)
	ax.set_title("KMeans con K=2")
	ax.set_xlabel(x_plot)
	ax.set_ylabel(y_plot)
	fig.savefig(output_dir / "realista_kmeans_k2.png", dpi=300, bbox_inches="tight")

	inertias = []
	k_values = list(range(1, 11))
	for k in k_values:
		clu_kmeans = Pipeline(
			steps=[
				("preprocessor", preprocessor),
				("clustering", KMeans(n_clusters=k, random_state=random_state, n_init=10)),
			]
		)
		clu_kmeans.fit(X)
		inertias.append(clu_kmeans["clustering"].inertia_)

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.plot(k_values, inertias, marker="o")
	ax.set_title("Metodo del codo")
	ax.set_xlabel("K")
	ax.set_ylabel("Inercia")
	fig.savefig(output_dir / "realista_metodo_codo.png", dpi=300, bbox_inches="tight")

	best_k = estimate_elbow(k_values, inertias)
	print(f"K recomendado por codo: {best_k}")

	clu_kmeans_best = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("clustering", KMeans(n_clusters=best_k, random_state=random_state, n_init=10)),
		]
	)
	clu_kmeans_best.fit(X)
	print(f"Con K = {best_k}: inercia = {clu_kmeans_best['clustering'].inertia_:.4f}")

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.scatter(
		X[x_plot],
		X[y_plot],
		c=clu_kmeans_best["clustering"].labels_,
		cmap="tab10",
		alpha=0.85,
	)
	ax.set_title(f"KMeans con K={best_k}")
	ax.set_xlabel(x_plot)
	ax.set_ylabel(y_plot)
	fig.savefig(output_dir / f"realista_kmeans_k{best_k}.png", dpi=300, bbox_inches="tight")

	X_db = preprocessor.fit_transform(X)
	dbscan = DBSCAN(eps=1.2, min_samples=8)
	db_labels = dbscan.fit_predict(X_db)

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.scatter(X[x_plot], X[y_plot], c=db_labels, cmap="tab10", alpha=0.85)
	ax.set_title("DBSCAN")
	ax.set_xlabel(x_plot)
	ax.set_ylabel(y_plot)
	fig.savefig(output_dir / "realista_dbscan.png", dpi=300, bbox_inches="tight")

	unique_labels, counts = np.unique(db_labels, return_counts=True)
	print("Clusters DBSCAN (label: cantidad):")
	print(dict(zip(unique_labels, counts)))

	plt.tight_layout()


if __name__ == "__main__":
	main()
