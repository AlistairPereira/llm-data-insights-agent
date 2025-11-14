# eda_agent.py

import sys
import json
import os

from tools_data import (
    load_dataset,
    basic_cleaning,
    quick_eda_summary,
    plot_numeric_hist,
)
from llm_local import run_llm


def build_eda_prompt(eda_summary: dict, file_path: str) -> str:
    """
    Prompt for EDA insights only. This is sent to the local LLM (Ollama).
    """
    return f"""
You are a senior data analyst.

You are analyzing a dataset from the file: {file_path}

Below is the EDA summary in JSON format:
{json.dumps(eda_summary, indent=2)}

Using ONLY this information:

1. Briefly describe the dataset (rows, columns, and data types).
2. Comment on data quality (missing values or issues).
3. Highlight 3–5 important numeric insights (trends, ranges, correlations).
4. Suggest 3 follow-up analyses or questions.

Provide the answer in clear bullet points.
    """


def run_eda_agent(file_path: str, use_llm: bool = True) -> dict:
    """
    EDA Agent:
    1. Load dataset
    2. Clean data
    3. Compute EDA summary
    4. Generate histograms
    5. (Optional) Ask LLM for EDA insights
    6. RETURN the EDA summary dict for other agents (e.g., model_agent)
    """

    print(f" [EDA Agent] Loading dataset: {file_path}")
    df_raw = load_dataset(file_path)
    print(f"[EDA Agent] Loaded dataset with shape: {df_raw.shape}")

    print("🧹 [EDA Agent] Cleaning dataset ...")
    df_clean = basic_cleaning(df_raw)

    print("[EDA Agent] Generating EDA summary ...")
    eda = quick_eda_summary(df_clean)

    print("[EDA Agent] Creating histograms ...")
    os.makedirs("outputs", exist_ok=True)
    hist_path = plot_numeric_hist(df_clean)
    if hist_path:
        print(f"[EDA Agent] Saved histograms to: {hist_path}")
    else:
        print("ℹ [EDA Agent] No numeric columns found to plot.")

    if use_llm:
        prompt = build_eda_prompt(eda, file_path)
        print("[EDA Agent] Querying local LLM (Ollama llama3.2) for EDA insights ...")
        insights = run_llm(prompt)

        print("\n================= EDA LLM Insights =================\n")
        print(insights)
        print("=======================================================\n")

        with open("outputs/eda_insights.txt", "w", encoding="utf-8") as f:
            f.write(insights)
        print("[EDA Agent] EDA insights saved to outputs/eda_insights.txt")

    # KEY FOR MULTI-AGENT SETUP:
    #return the summary so model_agent can use it
    return eda


if __name__ == "__main__":
    # Allow running this agent standalone: python eda_agent.py sample_data/cars.csv
    if len(sys.argv) < 2:
        print("Usage: python eda_agent.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    run_eda_agent(csv_path, use_llm=True)
