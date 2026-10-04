"""
clustering.py
-------------
Custom K-Means implementation and cluster evaluation utilities.
Intentionally built from scratch (without sklearn) to demonstrate
understanding of the underlying mathematics.
"""

import numpy as np
import pandas as pd


def normalise(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Min-Max normalise selected features to [0, 1].
    
    Without normalisation, high-range features like GNI ($1k–$54k)
    would dominate low-range features like HDI (0.49–0.95) purely
    due to scale, not variance — which would bias the clusters.
    """
    df_norm = df.copy()
    for col in features:
        col_min = df[col].min()
        col_max = df[col].max()
        df_norm[col] = (df[col] - col_min) / (col_max - col_min + 1e-9)
    return df_norm


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points in feature space."""
    return np.sqrt(np.sum((a - b) ** 2))


def _single_kmeans_run(X: np.ndarray, k: int, max_iters: int,
                        rng: np.random.RandomState):
    """One K-Means run from a single random initialisation. Returns
    (labels, final_wcss) so the caller can pick the best of several runs."""
    idx = rng.choice(len(X), k, replace=False)
    centroids = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iters):
        new_labels = np.array([
            np.argmin([euclidean_distance(x, c) for c in centroids])
            for x in X
        ])
        new_centroids = np.array([
            X[new_labels == i].mean(axis=0) if np.any(new_labels == i)
            else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            centroids, labels = new_centroids, new_labels
            break
        centroids, labels = new_centroids, new_labels

    final_wcss = sum(
        np.sum((X[labels == i] - centroids[i]) ** 2)
        for i in range(k) if np.any(labels == i)
    )
    return labels, final_wcss


def kmeans(df: pd.DataFrame, features: list, k: int = 3,
           max_iters: int = 100, random_state: int = 42,
           n_init: int = 10) -> pd.Series:
    """
    K-Means clustering from scratch.

    A single random initialisation can land in a poor local optimum —
    this is a known weakness of vanilla K-Means, not a bug. To address
    it (the same way scikit-learn's `n_init` parameter does), this runs
    K-Means `n_init` times from different random starts and keeps the
    run with the lowest within-cluster sum of squares (WCSS). Without
    this, results are unstable and can disagree with sklearn's more
    robust default initialisation (k-means++, n_init=10) on the same
    data — which is exactly what happened during development here and
    is why this fix exists.

    Parameters
    ----------
    df          : DataFrame containing the data
    features    : list of column names to cluster on
    k           : number of clusters
    max_iters   : maximum iterations before stopping, per run
    random_state: seed for reproducible results
    n_init      : number of random restarts; the best (lowest WCSS) is kept

    Returns
    -------
    pd.Series of integer cluster labels (0, 1, 2, ...)
    """
    df_norm = normalise(df, features)
    X = df_norm[features].values

    master_rng = np.random.RandomState(random_state)
    best_labels, best_wcss = None, np.inf

    for run in range(n_init):
        run_rng = np.random.RandomState(master_rng.randint(0, 10_000))
        labels, wcss_val = _single_kmeans_run(X, k, max_iters, run_rng)
        if wcss_val < best_wcss:
            best_labels, best_wcss = labels, wcss_val

    return pd.Series(best_labels, index=df.index, name="cluster")


def wcss(df: pd.DataFrame, features: list, labels: pd.Series) -> float:
    """
    Within-Cluster Sum of Squares — used in the Elbow Method
    to validate the choice of k.
    Lower WCSS = tighter, more compact clusters.
    """
    df_norm = normalise(df, features)
    X = df_norm[features].values
    total = 0.0
    for i in labels.unique():
        cluster_pts = X[labels == i]
        centroid = cluster_pts.mean(axis=0)
        total += np.sum((cluster_pts - centroid) ** 2)
    return total


def silhouette_score_manual(df: pd.DataFrame,
                             features: list,
                             labels: pd.Series) -> float:
    """
    Silhouette Score — measures cluster separation quality.
    Range: -1 (poor) to +1 (excellent). Above 0.5 is good.
    
    a(i) = mean intra-cluster distance
    b(i) = mean nearest-cluster distance
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    """
    df_norm = normalise(df, features)
    X = df_norm[features].values
    n = len(X)
    scores = []

    for i in range(n):
        own_cluster = labels.iloc[i]
        own_pts = X[labels == own_cluster]
        a = np.mean([euclidean_distance(X[i], p)
                     for p in own_pts if not np.array_equal(p, X[i])])

        other_clusters = [c for c in labels.unique() if c != own_cluster]
        b = min(
            np.mean([euclidean_distance(X[i], p)
                     for p in X[labels == c]])
            for c in other_clusters
        )

        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        scores.append(s)

    return float(np.mean(scores))


def gini_coefficient(values: pd.Series) -> float:
    """
    Gini Coefficient for inequality measurement.
    0 = perfect equality, 1 = perfect inequality.
    Applied here to GNI per capita distribution.
    """
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    mean = arr.mean()
    numerator = sum(abs(arr[i] - arr[j])
                    for i in range(n) for j in range(n))
    return numerator / (2 * n * n * mean)


def elbow_curve(df: pd.DataFrame, features: list,
                k_range: range = range(2, 8)) -> dict:
    """
    Compute WCSS for a range of k values to produce the Elbow Plot.
    Returns a dict of {k: wcss_value}.
    """
    results = {}
    for k in k_range:
        labels = kmeans(df, features, k=k)
        results[k] = wcss(df, features, labels)
    return results
