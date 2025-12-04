# # unsupervised_model_agent.py

# import sys
# import json
# import os

# from eda_agent import run_eda_agent
# from tools_data import load_dataset
# from tools_unsupervised import (
#     run_pca,
#     run_kmeans,
#     run_dbscan,
#     run_agglomerative,
# )
# from llm_local import run_llm

# def build_unsupervised_prompt(
#     file_path: str,
#     algo: str,
#     eda_summary: dict,
#     algo_report: dict,
# ) -> str:
#     """
#     Build a prompt that combines:
#     - EDA summary (with correlations + outliers)
#     - One unsupervised algorithm report (PCA / KMeans / DBSCAN / Agglomerative)
#     and asks the LLM to explain the structure in the data, using all metrics.
#     """
#     return f"""
# You are a senior data scientist specializing in unsupervised learning.

# You are analyzing a dataset from file: {file_path}

# The chosen unsupervised algorithm is: {algo}

# Here is the EDA summary (JSON), which includes correlations and outlier information:
# {json.dumps(eda_summary, indent=2)}

# Here is the unsupervised algorithm report (JSON), which includes quality metrics:
# {json.dumps(algo_report, indent=2)}

# Using ONLY this information, provide a structured explanation.

# You MUST include the following metrics when present:

# --- PCA METRICS ---
# - Explained variance ratio per component.
# - Total explained variance (like accuracy for PCA).

# --- CLUSTERING METRICS ---
# - Silhouette score.
# - Calinski-Harabasz score.
# - Davies-Bouldin score.

# --- ALGORITHM-SPECIFIC METRICS ---
# - For KMeans: inertia and cluster centers.
# - For DBSCAN: noise ratio and number of clusters (excluding noise).
# - For Agglomerative: linkage method and number of clusters.

# Now answer the following:

# 1. Briefly summarize the dataset and which unsupervised method was applied.
# 2. Explain what the metrics indicate about the quality of the PCA or clustering.
# 3. Describe the structure of the data revealed by the unsupervised method
#    (e.g., main PCA directions, or how clusters differ in key features).
# 4. Use EDA info:
#    - Mention any important correlations relevant to the discovered components/clusters.
#    - Comment on outliers and how they might influence the unsupervised results.

# 5. Suggest 3–5 practical applications:
#    - Segmentation, anomaly detection, feature engineering, or other business uses.

# 6. Mention limitations and caveats:
#    - Data quality issues (missing values, outliers, skewed distributions).
#    - Any reasons why the unsupervised patterns should be interpreted carefully.

# Write the answer in clear bullet points and short paragraphs.
#     """



# def run_unsupervised_agent(
#     file_path: str,
#     algo: str,
#     n_components: int | None = None,
#     n_clusters: int | None = None,
#     eps: float | None = None,
#     min_samples: int | None = None,
#     linkage: str | None = None,
# ) -> None:
#     """
#     Full pipeline for the Unsupervised Agent:

#     1. Call EDA agent (no LLM) to get EDA summary.
#     2. Load dataset.
#     3. Run the selected unsupervised algorithm.
#     4. Ask the LLM to explain the results.
#     """

#     algo = algo.lower()
#     print(f"[Unsupervised Agent] Starting unsupervised analysis for: {file_path}")
#     print(f"[Unsupervised Agent] Selected algorithm: {algo}")

#     # 1) Get EDA summary from EDA Agent (without LLM to avoid double explanations)
#     print("[Unsupervised Agent] Calling EDA Agent (no LLM) for summary ...")
#     eda_summary = run_eda_agent(file_path, use_llm=False)

#     # 2) Load dataset
#     print("[Unsupervised Agent] Loading dataset ...")
#     df = load_dataset(file_path)
#     print(f"[Unsupervised Agent] Data shape: {df.shape}")

#     # 3) Run the chosen unsupervised algorithm
#     algo_report: dict

#     if algo == "pca":
#         # Default components if not provided
#         n_components = n_components if n_components is not None else 2
#         print(f"[Unsupervised Agent] Running PCA with n_components={n_components} ...")
#         algo_report = run_pca(df, n_components=n_components)

#     elif algo == "kmeans":
#         # Default clusters if not provided
#         n_clusters = n_clusters if n_clusters is not None else 3
#         print(f"[Unsupervised Agent] Running KMeans with n_clusters={n_clusters} ...")
#         algo_report = run_kmeans(df, n_clusters=n_clusters)

#     elif algo == "dbscan":
#         # Defaults if not provided
#         eps = eps if eps is not None else 0.5
#         min_samples = min_samples if min_samples is not None else 5
#         print(f"[Unsupervised Agent] Running DBSCAN with eps={eps}, min_samples={min_samples} ...")
#         algo_report = run_dbscan(df, eps=eps, min_samples=min_samples)

#     elif algo in ("agg", "agglomerative", "hierarchical"):
#         # Default clusters + linkage
#         n_clusters = n_clusters if n_clusters is not None else 3
#         linkage = linkage if linkage is not None else "ward"
#         print(f"[Unsupervised Agent] Running Agglomerative with n_clusters={n_clusters}, linkage='{linkage}' ...")
#         algo_report = run_agglomerative(df, n_clusters=n_clusters, linkage=linkage)

#     else:
#         print(f"[Unsupervised Agent]  Unknown algorithm '{algo}'.")
#         print("Supported algorithms: pca, kmeans, dbscan, agglomerative")
#         return

#     # 4) Build prompt and ask LLM
#     prompt = build_unsupervised_prompt(
#         file_path=file_path,
#         algo=algo,
#         eda_summary=eda_summary,
#         algo_report=algo_report,
#     )

#     print("[Unsupervised Agent] Querying local LLM (Ollama llama3.2) for unsupervised insights ...")
#     insights = run_llm(prompt)

#     print("\n================= UNSUPERVISED LLM Insights =================\n")
#     print(insights)
#     print("=============================================================\n")

#     os.makedirs("outputs", exist_ok=True)
#     out_path = "outputs/unsupervised_insights.txt"
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.write(insights)

#     print(f"[Unsupervised Agent] Insights saved to {out_path}")


# def _print_usage() -> None:
#     print(
#         """
# Usage:
#     python unsupervised_model_agent.py <path_to_csv> <algorithm> [params]

# Algorithms and optional parameters:

# 1) PCA:
#     python unsupervised_model_agent.py sample_data/cars.csv pca [n_components]
#     - n_components: int, default=2

# 2) KMeans:
#     python unsupervised_model_agent.py sample_data/cars.csv kmeans [n_clusters]
#     - n_clusters: int, default=3

# 3) DBSCAN:
#     python unsupervised_model_agent.py sample_data/cars.csv dbscan [eps] [min_samples]
#     - eps: float, default=0.5
#     - min_samples: int, default=5

# 4) Agglomerative (Hierarchical):
#     python unsupervised_model_agent.py sample_data/cars.csv agglomerative [n_clusters] [linkage]
#     - n_clusters: int, default=3
#     - linkage: one of 'ward', 'complete', 'average', 'single' (default='ward')
# """
#     )


# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         _print_usage()
#         sys.exit(1)

#     file_path = sys.argv[1]
#     algo = sys.argv[2].lower()

#     # Defaults (can be overridden by user args)
#     n_components = None
#     n_clusters = None
#     eps = None
#     min_samples = None
#     linkage = None

#     # Parse extra CLI arguments based on chosen algorithm
#     if algo == "pca":
#         if len(sys.argv) >= 4:
#             n_components = int(sys.argv[3])

#     elif algo == "kmeans":
#         if len(sys.argv) >= 4:
#             n_clusters = int(sys.argv[3])

#     elif algo == "dbscan":
#         if len(sys.argv) >= 4:
#             eps = float(sys.argv[3])
#         if len(sys.argv) >= 5:
#             min_samples = int(sys.argv[4])

#     elif algo in ("agg", "agglomerative", "hierarchical"):
#         if len(sys.argv) >= 4:
#             n_clusters = int(sys.argv[3])
#         if len(sys.argv) >= 5:
#             linkage = sys.argv[4]

#     run_unsupervised_agent(
#         file_path=file_path,
#         algo=algo,
#         n_components=n_components,
#         n_clusters=n_clusters,
#         eps=eps,
#         min_samples=min_samples,
#         linkage=linkage,
#     )

# unsupervised_model_agent.py

import sys
import json
import os

from eda_agent import run_eda_agent
from tools_data import load_dataset, get_dataset_key_from_path
from tools_unsupervised import (
    run_pca,
    run_kmeans,
    run_dbscan,
    run_agglomerative,
)
from llm_local import run_llm


def build_unsupervised_prompt(
    file_path: str,
    algo: str,
    eda_summary: dict,
    algo_report: dict,
) -> str:
    """
    Build a prompt that combines:
    - EDA summary (with correlations + outliers)
    - One unsupervised algorithm report (PCA / KMeans / DBSCAN / Agglomerative)
    and asks the LLM to explain the structure in the data, using all metrics.
    """
    return f"""
You are a senior data scientist specializing in unsupervised learning.

You are analyzing a dataset from file: {file_path}

The chosen unsupervised algorithm is: {algo}

Here is the EDA summary (JSON), which includes correlations and outlier information:
{json.dumps(eda_summary, indent=2)}

Here is the unsupervised algorithm report (JSON), which includes quality metrics:
{json.dumps(algo_report, indent=2)}

Using ONLY this information, provide a structured explanation.

You MUST include the following metrics when present:

--- PCA METRICS ---
- Explained variance ratio per component.
- Total explained variance (like accuracy for PCA).

--- CLUSTERING METRICS ---
- Silhouette score.
- Calinski-Harabasz score.
- Davies-Bouldin score.

--- ALGORITHM-SPECIFIC METRICS ---
- For KMeans: inertia and cluster centers.
- For DBSCAN: noise ratio and number of clusters (excluding noise).
- For Agglomerative: linkage method and number of clusters.

Now answer the following:

1. Briefly summarize the dataset and which unsupervised method was applied.
2. Explain what the metrics indicate about the quality of the PCA or clustering.
3. Describe the structure of the data revealed by the unsupervised method
   (e.g., main PCA directions, or how clusters differ in key features).
4. Use EDA info:
   - Mention any important correlations relevant to the discovered components/clusters.
   - Comment on outliers and how they might influence the unsupervised results.

5. Suggest 3–5 practical applications:
   - Segmentation, anomaly detection, feature engineering, or other business uses.

6. Mention limitations and caveats:
   - Data quality issues (missing values, outliers, skewed distributions).
   - Any reasons why the unsupervised patterns should be interpreted carefully.

Write the answer in clear bullet points and short paragraphs.
    """


def run_unsupervised_agent(
    file_path: str,
    algo: str,
    n_components: int | None = None,
    n_clusters: int | None = None,
    eps: float | None = None,
    min_samples: int | None = None,
    linkage: str | None = None,
) -> None:
    """
    Full pipeline for the Unsupervised Agent:

    1. Call EDA agent (no LLM) to get EDA summary.
    2. Load dataset.
    3. Run the selected unsupervised algorithm.
    4. Ask the LLM to explain the results.
    """

    algo = algo.lower()
    print(f"[Unsupervised Agent] Starting unsupervised analysis for: {file_path}")
    print(f"[Unsupervised Agent] Selected algorithm: {algo}")

    # ----- per-dataset output folder -----
    dataset_key = get_dataset_key_from_path(file_path)   # e.g. 'cars', 'iris'
    base_dir = os.path.join("outputs", dataset_key)
    os.makedirs(base_dir, exist_ok=True)
    print(f"[Unsupervised Agent] Using output folder: {base_dir}")

    # 1) Get EDA summary from EDA Agent (without LLM to avoid double explanations)
    print("[Unsupervised Agent] Calling EDA Agent (no LLM) for summary ...")
    eda_summary = run_eda_agent(file_path, use_llm=False)

    # 2) Load dataset
    print("[Unsupervised Agent] Loading dataset ...")
    df = load_dataset(file_path)
    print(f"[Unsupervised Agent] Data shape: {df.shape}")

    # 3) Run the chosen unsupervised algorithm
    algo_report: dict

    if algo == "pca":
        # Default components if not provided
        n_components = n_components if n_components is not None else 2
        print(f"[Unsupervised Agent] Running PCA with n_components={n_components} ...")
        algo_report = run_pca(df, n_components=n_components)

    elif algo == "kmeans":
        # Default clusters if not provided
        n_clusters = n_clusters if n_clusters is not None else 3
        print(f"[Unsupervised Agent] Running KMeans with n_clusters={n_clusters} ...")
        algo_report = run_kmeans(df, n_clusters=n_clusters)

    elif algo == "dbscan":
        # Defaults if not provided
        eps = eps if eps is not None else 0.5
        min_samples = min_samples if min_samples is not None else 5
        print(f"[Unsupervised Agent] Running DBSCAN with eps={eps}, min_samples={min_samples} ...")
        algo_report = run_dbscan(df, eps=eps, min_samples=min_samples)

    elif algo in ("agg", "agglomerative", "hierarchical"):
        # Default clusters + linkage
        n_clusters = n_clusters if n_clusters is not None else 3
        linkage = linkage if linkage is not None else "ward"
        print(f"[Unsupervised Agent] Running Agglomerative with n_clusters={n_clusters}, linkage='{linkage}' ...")
        algo_report = run_agglomerative(df, n_clusters=n_clusters, linkage=linkage)

    else:
        print(f"[Unsupervised Agent]  Unknown algorithm '{algo}'.")
        print("Supported algorithms: pca, kmeans, dbscan, agglomerative")
        return

    # Save raw algo_report JSON (useful for future reports if needed)
    algo_report_path = os.path.join(base_dir, f"unsupervised_{algo}_report.json")
    with open(algo_report_path, "w", encoding="utf-8") as f:
        json.dump(algo_report, f, indent=2)
    print(f"[Unsupervised Agent] Saved {algo} report to {algo_report_path}")

    # 4) Build prompt and ask LLM
    prompt = build_unsupervised_prompt(
        file_path=file_path,
        algo=algo,
        eda_summary=eda_summary,
        algo_report=algo_report,
    )

    print("[Unsupervised Agent] Querying local LLM (Ollama llama3.2) for unsupervised insights ...")
    insights = run_llm(prompt)

    print("\n================= UNSUPERVISED LLM Insights =================\n")
    print(insights)
    print("=============================================================\n")

    out_path = os.path.join(base_dir, "unsupervised_insights.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(insights)

    print(f"[Unsupervised Agent] Insights saved to {out_path}")


def _print_usage() -> None:
    print(
        """
Usage:
    python unsupervised_model_agent.py <path_to_csv> <algorithm> [params]

Algorithms and optional parameters:

1) PCA:
    python unsupervised_model_agent.py sample_data/cars.csv pca [n_components]
    - n_components: int, default=2

2) KMeans:
    python unsupervised_model_agent.py sample_data/cars.csv kmeans [n_clusters]
    - n_clusters: int, default=3

3) DBSCAN:
    python unsupervised_model_agent.py sample_data/cars.csv dbscan [eps] [min_samples]
    - eps: float, default=0.5
    - min_samples: int, default=5

4) Agglomerative (Hierarchical):
    python unsupervised_model_agent.py sample_data/cars.csv agglomerative [n_clusters] [linkage]
    - n_clusters: int, default=3
    - linkage: one of 'ward', 'complete', 'average', 'single' (default='ward')
"""
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _print_usage()
        sys.exit(1)

    file_path = sys.argv[1]
    algo = sys.argv[2].lower()

    # Defaults (can be overridden by user args)
    n_components = None
    n_clusters = None
    eps = None
    min_samples = None
    linkage = None

    # Parse extra CLI arguments based on chosen algorithm
    if algo == "pca":
        if len(sys.argv) >= 4:
            n_components = int(sys.argv[3])

    elif algo == "kmeans":
        if len(sys.argv) >= 4:
            n_clusters = int(sys.argv[3])

    elif algo == "dbscan":
        if len(sys.argv) >= 4:
            eps = float(sys.argv[3])
        if len(sys.argv) >= 5:
            min_samples = int(sys.argv[4])

    elif algo in ("agg", "agglomerative", "hierarchical"):
        if len(sys.argv) >= 4:
            n_clusters = int(sys.argv[3])
        if len(sys.argv) >= 5:
            linkage = sys.argv[4]

    run_unsupervised_agent(
        file_path=file_path,
        algo=algo,
        n_components=n_components,
        n_clusters=n_clusters,
        eps=eps,
        min_samples=min_samples,
        linkage=linkage,
    )
