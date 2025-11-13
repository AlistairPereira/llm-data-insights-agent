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


def build_analysis_prompt(eda_summary: dict, file_path: str) -> str:
    """
    Create a natural-language prompt that contains the EDA summary in JSON form.
    This is the text that will be sent to the LLM (Ollama llama3.2).
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
4. Suggest 3 follow-up business analyses or questions.

Provide the answer in clear bullet points.
    """


def run_agent(file_path: str) -> None:
    """
    Main EDA agent pipeline:
    1. Load dataset
    2. Clean data
    3. Compute EDA summary
    4. Generate histograms
    5. Build prompt
    6. Ask the local LLM (Ollama)
    7. Display and save insights
    """

    print(f"📂 Loading dataset: {file_path}")
    df_raw = load_dataset(file_path)
    print(f"✅ Loaded dataset with shape: {df_raw.shape}")

    print("🧹 Cleaning dataset ...")
    df_clean = basic_cleaning(df_raw)

    print("📊 Generating EDA summary ...")
    eda = quick_eda_summary(df_clean)

    print("📈 Creating histograms ...")
    os.makedirs("outputs", exist_ok=True)
    hist_path = plot_numeric_hist(df_clean)

    if hist_path:
        print(f"✅ Saved histograms to: {hist_path}")
    else:
        print("ℹ No numeric columns found to plot.")

    # Build the LLM prompt
    prompt = build_analysis_prompt(eda, file_path)

    print("🧠 Querying local LLM (Ollama llama3.2) ...")
    insights = run_llm(prompt)

    print("\n================= 📌 LLM Insights =================\n")
    print(insights)
    print("===================================================\n")

    # Save to file
    with open("outputs/insights.txt", "w", encoding="utf-8") as f:
        f.write(insights)

    print("📝 Insights saved to outputs/insights.txt")


if __name__ == "__main__":
    # Expecting: python agent.py <csv_path>
    if len(sys.argv) < 2:
        print("Usage: python agent.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    run_agent(csv_path)
