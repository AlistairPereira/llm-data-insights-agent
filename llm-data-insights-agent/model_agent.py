# model_agent.py

import sys
import json
import os

from eda_agent import run_eda_agent
from tools_data import load_dataset
from tools_model import (
    choose_target_column,
    prepare_features_and_target,
    train_model,
    evaluate_model,
)
from llm_local import run_llm


def build_model_prompt(
    file_path: str,
    target_col: str,
    eda_summary: dict,
    model_report: dict,
) -> str:
    """
    Build a prompt that combines:
    - EDA summary
    - Model type + metrics + feature importance
    and asks the LLM to explain everything.
    """
    return f"""
You are a senior data scientist.

You are analyzing a dataset from file: {file_path}
The chosen target column for prediction is: {target_col}

Here is the EDA summary (JSON):
{json.dumps(eda_summary, indent=2)}

Here is the model evaluation report (JSON):
{json.dumps(model_report, indent=2)}

Using ONLY this information:

1. Briefly summarize the dataset and the chosen target.
2. Explain whether this is a regression or classification problem and why.
3. Interpret the model performance metrics in simple terms.
4. Comment on the target missing percentage and what it implies.
5. Explain which features appear most important for the model and why that makes sense.
6. Suggest 3 concrete next steps to improve the model or analysis.

Answer in clear bullet points.
    """


def run_model_agent(file_path: str, target_col: str | None = None) -> None:
    """
    Full pipeline for Agent 2:
    1. Call EDA agent to get EDA summary (no LLM here).
    2. Load the dataset.
    3. Choose / validate target column.
    4. Prepare features + target (encoding, scaling).
    5. Train model (regression or classification).
    6. Evaluate model and compute feature importance & target missing %.
    7. Ask LLM for a combined EDA + model explanation.
    """

    print(f"[Model Agent] Starting modeling pipeline for: {file_path}")

    # 1) Get EDA summary from Agent 1 (without calling LLM again)
    print("[Model Agent] Calling EDA agent (no LLM) to get EDA summary ...")
    eda_summary = run_eda_agent(file_path, use_llm=False)

    # 2) Load the dataset again for modeling
    print("[Model Agent] Loading dataset for modeling ...")
    df = load_dataset(file_path)
    print(f"[Model Agent] Data shape: {df.shape}")

    # 3) Choose / validate target column
    target_col_final = choose_target_column(df, target_col)
    print(f"[Model Agent] Using target column: {target_col_final}")

    # 4) Prepare features and target
    print("[Model Agent] Preparing features and target ...")
    X, y, problem_type, ord_encoder, label_encoder, feature_names = prepare_features_and_target(
        df, target_col_final
    )
    print(f"[Model Agent] Inferred problem type: {problem_type}")

    # 5) Train model
    print("[Model Agent] Training model ...")
    model, y_test, y_pred = train_model(X, y, problem_type)

    # 6) Evaluate model
    print("[Model Agent] Evaluating model ...")
    metrics = evaluate_model(y_test, y_pred, problem_type)

    # 6a) Compute real target missing percentage
    target_missing_pct = float(df[target_col_final].isna().mean() * 100.0)

    # 6b) Compute feature importances (top 10) if model supports it
    top_feature_importances: dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        pairs = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        top_pairs = pairs[:10]
        top_feature_importances = {
            name: round(score, 3) for name, score in top_pairs
        }

    # Build a unified model_report dict
    model_report = {
        "problem_type": metrics.get("type", problem_type),
        "target_column": target_col_final,
        "target_missing_percent": round(target_missing_pct, 2),
        "metrics": metrics,
        "top_feature_importances": top_feature_importances,
    }

    # 7) Build prompt and ask LLM for combined explanation
    prompt = build_model_prompt(
        file_path=file_path,
        target_col=target_col_final,
        eda_summary=eda_summary,
        model_report=model_report,
    )

    print("[Model Agent] Querying local LLM (Ollama llama3.2) for model insights ...")
    insights = run_llm(prompt)

    print("\n=================  MODEL LLM Insights =================\n")
    print(insights)
    print("=========================================================\n")

    # Save to file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model_insights.txt", "w", encoding="utf-8") as f:
        f.write(insights)

    print("[Model Agent] Model insights saved to outputs/model_insights.txt")


if __name__ == "__main__":
    # Usage:
    #   python model_agent.py <path_to_csv> [target_column]
    if len(sys.argv) < 2:
        print("Usage: python model_agent.py <path_to_csv> [target_column]")
        sys.exit(1)

    csv_path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) >= 3 else None

    run_model_agent(csv_path, target_col=target)
