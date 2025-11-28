

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
    target_column: str | None,
    problem_type: str,
    eda_summary: dict,
    model_report: dict,
) -> str:
    """
    Prompt for supervised model explanation.
    """
    return f"""
You are a senior data scientist.

You are analyzing a dataset from the file: {file_path}
The supervised learning task uses target column: {target_column}
The problem type is: {problem_type} (regression or classification).

Here is the EDA summary in JSON (includes correlations and outlier info):
{json.dumps(eda_summary, indent=2)}

Here is the model report in JSON (includes metrics, feature importance, and target missing percentage):
{json.dumps(model_report, indent=2)}

Using ONLY this information:

1. Briefly summarize:
   - What the dataset looks like (rows, columns, key data types).
   - What the target column represents and why this is a {problem_type} problem.

2. Explain the model performance:
   - For regression, clearly report metrics like RMSE and R².
   - For classification, clearly report metrics like accuracy, precision/recall/F1 if present.
   - Comment on whether the performance seems good, moderate, or poor.

3. Use EDA info (correlations + outliers) to interpret the model:
   - Mention which features are most strongly correlated with the target.
   - Discuss feature importances from the model and compare them with the correlations.
   - Comment on outliers in the target or key features and how they might affect metrics.

4. Provide practical insights:
   - 3–5 concrete interpretations of what the model is telling us about the data.

5. Suggest next steps:
   - Feature engineering ideas.
   - Handling outliers or skewed features.
   - Model improvements (hyperparameter tuning, more data, different algorithms).

Answer in clear bullet points and short paragraphs so a non-expert data stakeholder can understand.
    """


def run_model_agent(file_path: str, target_col: str | None = None, algo: str = "random_forest") -> None:
    """
    Full pipeline for Agent 2.
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
    X, y, problem_type, cat_encoder, y_encoder, feature_names = prepare_features_and_target(
        df, target_col_final
    )
    print(f"[Model Agent] Inferred problem type: {problem_type}")

    # 5) Train model
    print(f"[Model Agent] Training model with algo = {algo} ...")
    model, y_test, y_pred = train_model(X, y, problem_type, algo=algo)

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
            reverse=True,
        )
        top_pairs = pairs[:10]
        top_feature_importances = {
            name: round(score, 3) for name, score in top_pairs
        }

    # Build unified model_report
    model_report = {
        "problem_type": metrics.get("type", problem_type),
        "target_column": target_col_final,
        "target_missing_percent": round(target_missing_pct, 2),
        "metrics": metrics,
        "top_feature_importances": top_feature_importances,
        "algorithm": algo,
    }

    # 7) Build prompt and ask LLM
    prompt = build_model_prompt(
        file_path=file_path,
        target_column=target_col_final,
        problem_type=problem_type,
        eda_summary=eda_summary,
        model_report=model_report,
    )

    print("[Model Agent] Querying local LLM (Ollama llama3.2) for model insights ...")
    insights = run_llm(prompt)

    print("\n================= MODEL LLM Insights =================\n")
    print(insights)
    print("=========================================================\n")

    # Save to file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model_insights.txt", "w", encoding="utf-8") as f:
        f.write(insights)

    print("[Model Agent] Model insights saved to outputs/model_insights.txt")


if __name__ == "__main__":
    # Usage:
    #   python model_agent.py <path_to_csv> [target_column] [algo]
    if len(sys.argv) < 2:
        print("Usage: python model_agent.py <path_to_csv> [target_column] [algo]")
        sys.exit(1)

    csv_path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) >= 3 else None
    algo = sys.argv[3] if len(sys.argv) >= 4 else "random_forest"

    run_model_agent(csv_path, target_col=target, algo=algo)
