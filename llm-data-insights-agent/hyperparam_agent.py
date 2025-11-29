# hyperparam_agent.py

import sys
import json
import os
from typing import List, Optional, Dict, Any

from tools_data import load_dataset
from tools_model import choose_target_column, prepare_features_and_target
from tools_hyperparam import run_hyperparam_experiments
from llm_local import run_llm


def build_hyperparam_prompt(
    file_path: str,
    target_column: str,
    problem_type: str,
    search_results: dict,
) -> str:
    return f"""
You are a senior machine learning engineer.

We ran hyperparameter search experiments on dataset: {file_path}
Target column: {target_column}
Problem type: {problem_type}

Here are the hyperparameter search results in JSON:
{json.dumps(search_results, indent=2)}

Using ONLY this information:

1. Summarize:
   - Which algorithms were tried.
   - Which algorithm performed best and what its key hyperparameters are.

2. Explain:
   - What the reported metric means (RMSE for regression, accuracy for classification).
   - How good the best metric value is in practical terms.

3. Interpret hyperparameters:
   - For tree-based models (Random Forest / Gradient Boosting), explain how n_estimators, max_depth, etc. affect performance and overfitting.
   - For linear models (Ridge, Lasso, Logistic Regression), explain the role of regularization (alpha or C).

4. Recommend:
   - Which model configuration you would deploy and why.
   - 3–5 ideas for future experiments (e.g., try more trees, different learning_rates, more data, feature engineering).

Answer in clear bullet points and short paragraphs so a non-expert data stakeholder can understand.
    """


def _load_last_model_report(path: str = "outputs/model_report.json") -> Optional[Dict[str, Any]]:
    """Load the last model_report.json written by model_agent."""
    if not os.path.exists(path):
        print(f"[Hyperparam Agent] No model_report found at {path}.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Hyperparam Agent] Loaded last model report from {path}")
        return data
    except Exception as e:
        print(f"[Hyperparam Agent] Failed to load model_report: {e}")
        return None


def _infer_algos_from_model_report(
    model_report: Dict[str, Any],
    problem_type: str,
) -> Optional[List[str]]:
    """
    Given last model_report and problem_type, choose which algorithms to tune.

    Regression examples:
      - 'linear'         → ['ridge', 'lasso', 'elasticnet']
      - 'random_forest'  → ['rf']
      - 'gbr'            → ['gbr']
      - 'knn'            → ['knn']

    Classification examples:
      - 'logreg'         → ['logreg']
      - 'random_forest'  → ['rf']
      - 'gboost'/'gbr'   → ['gboost']
      - 'knn'            → ['knn']
    """
    base_algo = str(model_report.get("algorithm", "")).lower()
    print(f"[Hyperparam Agent] Base algorithm from last model: '{base_algo}'")

    if problem_type == "regression":
        if base_algo in ("linear", "linreg", "lr"):
            return ["ridge", "lasso", "elasticnet"]
        if base_algo in ("random_forest", "rf", "randomforest"):
            return ["rf"]
        if base_algo in ("gbr", "gb", "gradient_boosting"):
            return ["gbr"]
        if base_algo in ("knn", "kneighbors", "k_neighbors"):
            return ["knn"]
        # fallback mix
        return ["rf", "gbr", "ridge", "lasso"]

    else:  # classification
        if base_algo in ("logreg", "logistic", "lr"):
            return ["logreg"]
        if base_algo in ("random_forest", "rf", "randomforest"):
            return ["rf"]
        if base_algo in ("gboost", "gbr", "gradient_boosting"):
            return ["gboost"]
        if base_algo in ("knn", "kneighbors", "k_neighbors"):
            return ["knn"]
        # fallback mix
        return ["rf", "gboost", "logreg", "knn"]


def run_hyperparam_agent(
    file_path: str,
    target_col: str | None = None,
    algo_list: List[str] | None = None,
    n_iter: int = 15,
    cv: int = 3,
) -> None:
    """
    Full hyperparameter tuning pipeline:

    1. Load dataset.
    2. Choose / validate target column.
    3. Prepare features + target (same as modeling agent).
    4. If algo_list is not given, read last model_report.json to infer which family to tune.
    5. Run hyperparameter experiments.
    6. Save raw JSON results.
    7. Ask LLM to explain results.
    """

    print(f"[Hyperparam Agent] Starting hyperparameter search for: {file_path}")

    # 1) Load dataset
    df = load_dataset(file_path)
    print(f"[Hyperparam Agent] Data shape: {df.shape}")

    # 2) Choose / validate target
    target_col_final = choose_target_column(df, target_col)
    print(f"[Hyperparam Agent] Using target column: {target_col_final}")

    # 3) Prepare features and target
    print("[Hyperparam Agent] Preparing features and target ...")
    X, y, problem_type, cat_encoder, y_encoder, feature_names = prepare_features_and_target(
        df, target_col_final
    )
    print(f"[Hyperparam Agent] Detected problem type: {problem_type}")

    # 4) If no algo_list → infer from last model_report.json
    if algo_list is None:
        print("[Hyperparam Agent] No algo_list provided. Trying to infer from last model_agent run...")
        model_report = _load_last_model_report()
        if model_report is not None:
            inferred = _infer_algos_from_model_report(model_report, str(problem_type))
            if inferred:
                algo_list = inferred
                print(f"[Hyperparam Agent] Inferring algorithms to tune based on last model: {algo_list}")
            else:
                print("[Hyperparam Agent] Could not infer algorithms, will fall back to defaults.")
        else:
            print("[Hyperparam Agent] No usable model_report found, will fall back to defaults.")

    # 5) Run hyperparameter experiments
    print("[Hyperparam Agent] Running hyperparameter experiments ...")
    search_results = run_hyperparam_experiments(
        X,
        y,
        problem_type=problem_type,  # type: ignore[arg-type]
        algo_list=algo_list,
        n_iter=n_iter,
        cv=cv,
    )

    # 6) Save JSON results
    os.makedirs("outputs", exist_ok=True)
    results_path = "outputs/hyperparam_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(search_results, f, indent=2)
    print(f"[Hyperparam Agent] Raw hyperparameter results saved to {results_path}")

    # 7) Ask LLM to explain results
    prompt = build_hyperparam_prompt(
        file_path=file_path,
        target_column=target_col_final,
        problem_type=str(problem_type),
        search_results=search_results,
    )

    print("[Hyperparam Agent] Querying local LLM (Ollama llama3.2) for hyperparam insights ...")
    insights = run_llm(prompt)

    print("\n================= HYPERPARAM LLM Insights =================\n")
    print(insights)
    print("============================================================\n")

    insights_path = "outputs/hyperparam_insights.txt"
    with open(insights_path, "w", encoding="utf-8") as f:
        f.write(insights)

    print(f"[Hyperparam Agent] Hyperparameter insights saved to {insights_path}")


if __name__ == "__main__":
    # Usually you will call this via router,
    # but you can still call directly:
    #   python hyperparam_agent.py sample_data/cars.csv price
    if len(sys.argv) < 2:
        print("Usage: python hyperparam_agent.py <path_to_csv> [target_column]")
        sys.exit(1)

    csv_path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) >= 3 else None

    run_hyperparam_agent(csv_path, target_col=target, algo_list=None)
