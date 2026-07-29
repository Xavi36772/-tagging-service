"""
Análisis de Clustering para entregable académico (Minería de Datos).

Evalúa K-Means, HDBSCAN y agrupamiento jerárquico sobre vectores de tags,
calcula métricas (Silhouette, Davies-Bouldin) y genera visualizaciones.

Uso: python cluster.py [--data-dir dataset] [--taxonomy taxonomy.json]
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_tag_vectors(data_dir: str, taxonomy_path: str):
    """Carga datos de entrenamiento+validación y construye matriz one-hot."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    tag2idx = {t.lower(): i for i, t in enumerate(taxonomy)}

    all_data = []
    for split in ["train.json", "val.json"]:
        path = Path(data_dir) / split
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                all_data.extend(json.load(f))

    # Matriz one-hot
    X = np.zeros((len(all_data), len(taxonomy)), dtype=np.float32)
    synopses = []
    for i, item in enumerate(all_data):
        synopses.append(item["synopsis"])
        for tag in item["tags"]:
            idx = tag2idx.get(tag.lower())
            if idx is not None:
                X[i, idx] = 1.0

    return X, taxonomy, synopses


def evaluate_kmeans(X, k_range=range(2, 21)):
    """Evalúa K-Means para diferentes K y encuentra el óptimo."""
    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)

        # Verificar que tenga al menos 2 clústeres con más de 1 elemento
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2 or any(counts == 1):
            continue

        sil = silhouette_score(X, labels, metric="cosine")
        db = davies_bouldin_score(X, labels)
        inertia = kmeans.inertia_

        results.append({"k": k, "silhouette": sil, "davies_bouldin": db, "inertia": inertia})
        print(f"  K={k:2d} | Silhouette={sil:.4f} | Davies-Bouldin={db:.4f} | Inertia={inertia:.2f}")

    return results


def evaluate_hdbscan(X):
    """Evalúa HDBSCAN con diferentes valores de min_cluster_size."""
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        print("HDBSCAN no disponible en esta versión de scikit-learn. Saltando.")
        return []

    results = []
    for mcs in [3, 5, 8, 10, 15, 20]:
        hdb = HDBSCAN(min_cluster_size=mcs, metric="cosine", cluster_selection_epsilon=0.3)
        labels = hdb.fit_predict(X)

        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        noise_pct = n_noise / len(labels) * 100

        if n_clusters >= 2:
            # Silhouette solo sobre puntos no-noise
            mask = labels != -1
            sil = silhouette_score(X[mask], labels[mask], metric="cosine")
        else:
            sil = None

        results.append({
            "min_cluster_size": mcs,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": f"{noise_pct:.1f}%",
            "silhouette": sil,
        })
        print(f"  min_cluster_size={mcs:2d} | clusters={n_clusters} | noise={noise_pct:.1f}% | Silhouette={sil:.4f}")

    return results


def evaluate_hierarchical(X, k_range=range(2, 15)):
    """Evalúa Agglomerative Clustering (Ward)."""
    results = []
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = agg.fit_predict(X)

        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2 or any(counts == 1):
            continue

        sil = silhouette_score(X, labels, metric="cosine")
        db = davies_bouldin_score(X, labels)
        results.append({"k": k, "silhouette": sil, "davies_bouldin": db})
        print(f"  K={k:2d} | Silhouette={sil:.4f} | Davies-Bouldin={db:.4f}")

    return results


def plot_results(kmeans_results, hdbscan_results, hier_results, output_dir):
    """Genera gráficas de evaluación."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # K-Means: Silhouette vs K
    if kmeans_results:
        ks = [r["k"] for r in kmeans_results]
        sils = [r["silhouette"] for r in kmeans_results]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ks, sils, "bo-", label="Silhouette Score")
        ax.axvline(x=ks[np.argmax(sils)], color="r", linestyle="--",
                   label=f"Óptimo K={ks[np.argmax(sils)]}")
        ax.set_xlabel("Número de Clústeres (K)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("K-Means: Silhouette Score vs K")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "kmeans_silhouette.png", dpi=150)
        plt.close()
        print(f"Gráfica guardada: {output_dir / 'kmeans_silhouette.png'}")

    # Comparativa final
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PCA 2D del mejor K-Means
    if kmeans_results:
        best_k = ks[np.argmax(sils)]
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)

        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            method = "PCA"
        else:
            X_2d = X
            method = "Original"

        axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
        axes[0].scatter(kmeans.cluster_centers_[:, 0] if X.shape[1] <= 2
                        else pca.transform(kmeans.cluster_centers_)[:, 0],
                        kmeans.cluster_centers_[:, 1] if X.shape[1] <= 2
                        else pca.transform(kmeans.cluster_centers_)[:, 1],
                        c="red", marker="x", s=100, linewidths=2, label="Centroides")
        axes[0].set_title(f"K-Means (K={best_k}, Silhouette={sils[ks.index(best_k)]:.3f})")
        axes[0].set_xlabel(f"{method} 1")
        axes[0].set_ylabel(f"{method} 2")
        axes[0].legend()

    # HDBSCAN
    if hdbscan_results and hdbscan_results[0]["silhouette"] is not None:
        best_hdb = max(hdbscan_results, key=lambda r: r["silhouette"] or 0)
        hdb = HDBSCAN(min_cluster_size=best_hdb["min_cluster_size"], metric="cosine",
                      cluster_selection_epsilon=0.3)
        labels = hdb.fit_predict(X)

        if X.shape[1] > 2:
            X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
        sc = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
        axes[1].set_title(f"HDBSCAN (min_cluster={best_hdb['min_cluster_size']}, "
                          f"Silhouette={best_hdb['silhouette']:.3f})")
        axes[1].set_xlabel("PCA 1")
        axes[1].set_ylabel("PCA 2")

    # Jerárquico
    if hier_results:
        hks = [r["k"] for r in hier_results]
        hsils = [r["silhouette"] for r in hier_results]
        best_hk = hks[np.argmax(hsils)]

        agg = AgglomerativeClustering(n_clusters=best_hk, metric="cosine", linkage="average")
        labels = agg.fit_predict(X)

        if X.shape[1] > 2:
            X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
        axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
        axes[2].set_title(f"Jerárquico (K={best_hk}, Silhouette={hsils[hks.index(best_hk)]:.3f})")
        axes[2].set_xlabel("PCA 1")
        axes[2].set_ylabel("PCA 2")

    plt.tight_layout()
    plt.savefig(output_dir / "clustering_comparison.png", dpi=150)
    plt.close()
    print(f"Gráfica guardada: {output_dir / 'clustering_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Análisis de Clustering para Minería de Datos")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Directorio del dataset")
    parser.add_argument("--taxonomy", type=str, default="taxonomy.json", help="Archivo de taxonomía")
    parser.add_argument("--output-dir", type=str, default="cluster_results", help="Directorio de resultados")
    args = parser.parse_args()

    global X
    print("Cargando datos...")
    X, taxonomy, synopses = load_tag_vectors(args.data_dir, args.taxonomy)
    print(f"Matriz cargada: {X.shape} obras × {X.shape[1]} etiquetas")
    print(f"Densidad: {X.sum() / X.size:.4f} ({X.sum():.0f} unos de {X.size} total)")

    # Etiquetas más frecuentes
    freq = X.sum(axis=0)
    top5 = np.argsort(freq)[-5:][::-1]
    print("\nEtiquetas más frecuentes:")
    for i in top5:
        print(f"  {taxonomy[i]}: {int(freq[i])} obras ({freq[i]/X.shape[0]*100:.1f}%)")

    # 1. K-Means
    print("\n=== K-MEANS ===")
    kmeans_results = evaluate_kmeans(X)

    if kmeans_results:
        best = max(kmeans_results, key=lambda r: r["silhouette"])
        print(f"\nMejor K-Means: K={best['k']}, Silhouette={best['silhouette']:.4f}")

    # 2. HDBSCAN
    print("\n=== HDBSCAN ===")
    hdbscan_results = evaluate_hdbscan(X)

    # 3. Jerárquico
    print("\n=== JERÁRQUICO (Agglomerative - Average Linkage) ===")
    hier_results = evaluate_hierarchical(X)

    # 4. Visualizaciones
    print("\n=== GENERANDO VISUALIZACIONES ===")
    plot_results(kmeans_results, hdbscan_results, hier_results, args.output_dir)

    # 5. Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    if kmeans_results:
        best_km = max(kmeans_results, key=lambda r: r["silhouette"])
        print(f"K-Means óptimo: K={best_km['k']}, Silhouette={best_km['silhouette']:.4f}")
    if hdbscan_results:
        valid = [r for r in hdbscan_results if r["silhouette"] is not None]
        if valid:
            best_hdb = max(valid, key=lambda r: r["silhouette"])
            print(f"HDBSCAN óptimo: min_cluster={best_hdb['min_cluster_size']}, "
                  f"Silhouette={best_hdb['silhouette']:.4f}")
    if hier_results:
        best_hier = max(hier_results, key=lambda r: r["silhouette"])
        print(f"Jerárquico óptimo: K={best_hier['k']}, Silhouette={best_hier['silhouette']:.4f}")

    print(f"\nResultados guardados en {args.output_dir}/")


if __name__ == "__main__":
    main()
