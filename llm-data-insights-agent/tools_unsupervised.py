# tools_unsupervised.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def prepare_unsupervised_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for unsupervised algorithms:

    Steps:
    - Identify categorical and numeric columns.
    - If number of categorical columns > 4:
        Use LabelEncoder on each categorical column separately.
      Else:
        Use a single OrdinalEncoder on all categorical columns.
    - Scale numeric columns using StandardScaler.
    - Return transformed X and list of feature names.
    """
    X = df.copy()

    # Identify categorical and numeric columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(include="number").columns

    # ---------------------------
    # Encode categorical features
    # ---------------------------
    if len(cat_cols) > 4:
        print("[Unsupervised] Using LabelEncoder for categorical features...")
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    else:
        if len(cat_cols) > 0:
            print("[Unsupervised] Using OrdinalEncoder for categorical features...")
            enc = OrdinalEncoder()
            X[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))

    # ---------------------------
    # Scale numeric features
    # ---------------------------
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    feature_names: List[str] = list(X.columns)
    return X, feature_names


def run_pca(df: pd.DataFrame, n_components: int) -> Dict[str, Any]:
    """
    Run PCA and return components + variance info (PCA 'accuracy').
    """
    X, feature_names = prepare_unsupervised_features(df)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    evr = pca.explained_variance_ratio_
    total_var = float(np.round(evr.sum(), 3))

    return {
        "components": X_pca.tolist(),
        "explained_variance_ratio": [round(v, 3) for v in evr],
        "total_explained_variance": total_var,  # like an 'accuracy' for PCA
        "n_components": n_components,
        "feature_names": feature_names,
    }


def _cluster_quality_metrics(X: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
    """
    Helper to compute common clustering quality metrics.
    Returns None for metrics when not applicable (e.g. 1 cluster).
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    # ignore noise label -1 for cluster count (for DBSCAN)
    valid_labels = [lab for lab in unique_labels if lab != -1]
    n_clusters = len(valid_labels)

    metrics: Dict[str, Any] = {
        "n_clusters": n_clusters,
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }

    if n_clusters >= 2 and n_samples > n_clusters:
        try:
            sil = silhouette_score(X, labels)
            cal = calinski_harabasz_score(X, labels)
            dav = davies_bouldin_score(X, labels)
            metrics["silhouette"] = float(np.round(sil, 3))
            metrics["calinski_harabasz"] = float(np.round(cal, 3))
            metrics["davies_bouldin"] = float(np.round(dav, 3))
        except Exception:
            # In weird cases metrics can fail; keep them as None
            pass

    return metrics


def run_kmeans(df: pd.DataFrame, n_clusters: int, random_state: int = 42) -> Dict[str, Any]:
    """
    Run KMeans clustering and return labels + centers + quality metrics.
    """
    X, feature_names = prepare_unsupervised_features(df)
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X)

    quality = _cluster_quality_metrics(X, labels)

    return {
        "algorithm": "kmeans",
        "requested_n_clusters": n_clusters,
        "cluster_labels": labels.tolist(),
        "cluster_centers": km.cluster_centers_.tolist(),
        "inertia": float(np.round(km.inertia_, 3)),
        "metrics": quality,
        "feature_names": feature_names,
    }


def run_dbscan(
    df: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Dict[str, Any]:
    """
    Run DBSCAN clustering and return labels + cluster metrics.
    """
    X, feature_names = prepare_unsupervised_features(df)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    quality = _cluster_quality_metrics(X, labels)
    n_clusters = quality["n_clusters"]

    noise_ratio = float(np.round((labels == -1).mean(), 3))

    return {
        "algorithm": "dbscan",
        "eps": eps,
        "min_samples": min_samples,
        "cluster_labels": labels.tolist(),
        "noise_ratio": noise_ratio,
        "metrics": quality,
        "feature_names": feature_names,
    }


def run_agglomerative(
    df: pd.DataFrame,
    n_clusters: int = 3,
    linkage: str = "ward",
) -> Dict[str, Any]:
    """
    Run Agglomerative (hierarchical) clustering and return labels + metrics.
    """
    X, feature_names = prepare_unsupervised_features(df)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X)

    quality = _cluster_quality_metrics(X, labels)

    return {
        "algorithm": "agglomerative",
        "n_clusters": n_clusters,
        "linkage": linkage,
        "cluster_labels": labels.tolist(),
        "metrics": quality,
        "feature_names": feature_names,
    }
