
# # eda_agent.py

# import sys
# import json
# import os

# from tools_data import (
#     load_dataset,
#     basic_cleaning,
#     quick_eda_summary,
#     plot_numeric_hist,
#     compute_correlations,
#     plot_corr_heatmap,
#     detect_outliers_iqr,
# )
# from llm_local import run_llm


# def build_eda_prompt(eda_summary: dict, file_path: str) -> str:
#     """
#     Prompt for EDA insights only. This is sent to the local LLM (Ollama).
#     Now also explicitly asks about correlations and outliers.
#     """
#     return f"""
# You are a senior data analyst.

# You are analyzing a dataset from the file: {file_path}

# Below is the EDA summary in JSON format (including correlations and outlier information):
# {json.dumps(eda_summary, indent=2)}

# Using ONLY this information:

# 1. Briefly describe the dataset (rows, columns, and data types).
# 2. Comment on data quality (missing values, outliers, and any potential issues).
# 3. Highlight 3–5 important numeric insights (trends, ranges, and correlations between key variables).
# 4. Mention any notable outliers (which columns they appear in and what they might imply).
# 5. Suggest 3 follow-up analyses or questions that could help explore the data further.

# Provide the answer in clear bullet points.
#     """



# def run_eda_agent(file_path: str, use_llm: bool = True) -> dict:
#     """
#     EDA Agent:
#     1. Load dataset
#     2. Clean data
#     3. Compute EDA summary (+ correlations)
#     4. Generate histograms + correlation heatmap
#     5. (Optional) Ask LLM for EDA insights
#     6. RETURN the EDA summary dict for other agents (e.g., model_agent)
#     """

#     print(f" [EDA Agent] Loading dataset: {file_path}")
#     df_raw = load_dataset(file_path)
#     print(f"[EDA Agent] Loaded dataset with shape: {df_raw.shape}")

#     print("[EDA Agent] Cleaning dataset ...")
#     df_clean = basic_cleaning(df_raw)

#     print("[EDA Agent] Generating EDA summary ...")
#     eda = quick_eda_summary(df_clean)

#     #   compute correlations and add into EDA summary
#     print("[EDA Agent] Computing correlations ...")
#     corr_info = compute_correlations(df_clean)
#     eda["correlations"] = corr_info
    
#     #  detect outliers on numeric columns (IQR-based)
#     print("[EDA Agent] Detecting outliers (IQR method) ...")
#     outlier_info = detect_outliers_iqr(df_clean, multiplier=1.5)
#     eda["outliers"] = outlier_info

#     print("[EDA Agent] Creating histograms ...")
#     os.makedirs("outputs", exist_ok=True)
#     hist_path = plot_numeric_hist(df_clean)
#     if hist_path:
#         print(f"[EDA Agent] Saved histograms to: {hist_path}")
#     else:
#         print("ℹ [EDA Agent] No numeric columns found to plot.")

#     #  NEW: correlation heatmap
#     print("[EDA Agent] Creating correlation heatmap ...")
#     corr_path = plot_corr_heatmap(df_clean)
#     if corr_path:
#         print(f"[EDA Agent] Saved correlation heatmap to: {corr_path}")
#     else:
#         print("ℹ [EDA Agent] No numeric columns found for correlation heatmap.")

#     if use_llm:
#         prompt = build_eda_prompt(eda, file_path)
#         print("[EDA Agent] Querying local LLM (Ollama llama3.2) for EDA insights ...")
#         insights = run_llm(prompt)

#         print("\n================= EDA LLM Insights =================\n")
#         print(insights)
#         print("=======================================================\n")

#         with open("outputs/eda_insights.txt", "w", encoding="utf-8") as f:
#             f.write(insights)
#         print("[EDA Agent] EDA insights saved to outputs/eda_insights.txt")

#     # KEY FOR MULTI-AGENT SETUP:
#     # return the summary so other agents can use it
#     return eda


# if __name__ == "__main__":
#     # Allow running this agent standalone: python eda_agent.py sample_data/cars.csv
#     if len(sys.argv) < 2:
#         print("Usage: python eda_agent.py <path_to_csv>")
#         sys.exit(1)

#     csv_path = sys.argv[1]
#     run_eda_agent(csv_path, use_llm=True)

# eda_agent.py

import sys
import json
import os

from tools_data import (
    load_dataset,
    basic_cleaning,
    quick_eda_summary,
    plot_numeric_hist,
    compute_correlations,
    plot_corr_heatmap,
    detect_outliers_iqr,
    get_dataset_key_from_path,   # ✅ new helper
)
from llm_local import run_llm


def build_eda_prompt(eda_summary: dict, file_path: str) -> str:
    """
    Prompt for EDA insights only. This is sent to the local LLM (Ollama).
    Now also explicitly asks about correlations and outliers.
    """
    return f"""
You are a senior data analyst.

You are analyzing a dataset from the file: {file_path}

Below is the EDA summary in JSON format (including correlations and outlier information):
{json.dumps(eda_summary, indent=2)}

Using ONLY this information:

1. Briefly describe the dataset (rows, columns, and data types).
2. Comment on data quality (missing values, outliers, and any potential issues).
3. Highlight 3–5 important numeric insights (trends, ranges, and correlations between key variables).
4. Mention any notable outliers (which columns they appear in and what they might imply).
5. Suggest 3 follow-up analyses or questions that could help explore the data further.

Provide the answer in clear bullet points.
    """


def run_eda_agent(file_path: str, use_llm: bool = True) -> dict:
    """
    EDA Agent:
    1. Load dataset
    2. Clean data
    3. Compute EDA summary (+ correlations + outliers)
    4. Generate histograms + correlation heatmap
    5. (Optional) Ask LLM for EDA insights
    6. RETURN the EDA summary dict for other agents (e.g., model_agent)
    """

    print(f" [EDA Agent] Loading dataset: {file_path}")
    df_raw = load_dataset(file_path)
    print(f"[EDA Agent] Loaded dataset with shape: {df_raw.shape}")

    # ----- per-dataset output folder -----
    dataset_key = get_dataset_key_from_path(file_path)   # e.g. 'cars', 'iris'
    base_dir = os.path.join("outputs", dataset_key)
    os.makedirs(base_dir, exist_ok=True)
    print(f"[EDA Agent] Using output folder: {base_dir}")

    print("[EDA Agent] Cleaning dataset ...")
    df_clean = basic_cleaning(df_raw)

    print("[EDA Agent] Generating EDA summary ...")
    eda = quick_eda_summary(df_clean)

    # correlations
    print("[EDA Agent] Computing correlations ...")
    corr_info = compute_correlations(df_clean)
    eda["correlations"] = corr_info

    # outliers
    print("[EDA Agent] Detecting outliers (IQR method) ...")
    outlier_info = detect_outliers_iqr(df_clean, multiplier=1.5)
    eda["outliers"] = outlier_info

    # save EDA summary JSON for this dataset
    summary_path = os.path.join(base_dir, "eda_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2)
    print(f"[EDA Agent] EDA summary saved to: {summary_path}")

    # ----- plots -----
    print("[EDA Agent] Creating histograms ...")
    hist_path = plot_numeric_hist(df_clean)  # old function returns a path
    if hist_path:
        dataset_hist_path = os.path.join(base_dir, "histograms.png")
        # move/rename if needed
        if hist_path != dataset_hist_path:
            try:
                os.replace(hist_path, dataset_hist_path)
                hist_path = dataset_hist_path
            except OSError:
                # if move fails, just leave original path
                pass
        print(f"[EDA Agent] Saved histograms to: {hist_path}")
    else:
        print("ℹ [EDA Agent] No numeric columns found to plot.")

    print("[EDA Agent] Creating correlation heatmap ...")
    corr_path = plot_corr_heatmap(df_clean)
    if corr_path:
        dataset_corr_path = os.path.join(base_dir, "corr_heatmap.png")
        if corr_path != dataset_corr_path:
            try:
                os.replace(corr_path, dataset_corr_path)
                corr_path = dataset_corr_path
            except OSError:
                pass
        print(f"[EDA Agent] Saved correlation heatmap to: {corr_path}")
    else:
        print("ℹ [EDA Agent] No numeric columns found for correlation heatmap.")

    # ----- LLM insights -----
    if use_llm:
        prompt = build_eda_prompt(eda, file_path)
        print("[EDA Agent] Querying local LLM (Ollama llama3.2) for EDA insights ...")
        insights = run_llm(prompt)

        print("\n================= EDA LLM Insights =================\n")
        print(insights)
        print("=======================================================\n")

        insights_path = os.path.join(base_dir, "eda_insights.txt")
        with open(insights_path, "w", encoding="utf-8") as f:
            f.write(insights)
        print(f"[EDA Agent] EDA insights saved to {insights_path}")

    # return the summary so other agents can use it
    return eda


if __name__ == "__main__":
    # Allow running this agent standalone: python eda_agent.py sample_data/cars.csv
    if len(sys.argv) < 2:
        print("Usage: python eda_agent.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    run_eda_agent(csv_path, use_llm=True)

