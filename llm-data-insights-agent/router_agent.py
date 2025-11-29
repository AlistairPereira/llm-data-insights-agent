#router_agent.py
import sys
import json

from llm_local import run_llm
from eda_agent import run_eda_agent
from model_agent import run_model_agent
from unsupervised_model_agent import run_unsupervised_agent
from hyperparam_agent import run_hyperparam_agent


def build_router_prompt(user_message: str) -> str:
    """
    Ask the LLM to decide which action to take based on the user request.
    It must respond with STRICT JSON.
    """
    return f"""
You are a routing assistant for a data analysis system.

The user will describe what they want to do with a dataset.

You have access to four high-level tools:

1) "eda"          - Run exploratory data analysis (EDA) on a CSV file.
2) "model"        - Train and evaluate a supervised ML model on a CSV file (needs target column).
3) "unsupervised" - Run an unsupervised analysis (PCA or clustering) on a CSV file.
4) "hyperparam"   - Train a supervised model AND then run hyperparameter tuning for it on a CSV file.
                    Use this when the user says things like "tune", "hyperparameters",
                    "optimize the model", "improve the model", or "find best parameters".

For supervised modeling ("model" or "hyperparam"):
- If the user mentions an algorithm, set "model_algo" accordingly:
    * "linear model", "linear regression"           → "linear"
    * "random forest", "forest"                     → "random_forest"
    * "gradient boosting", "boosting", "gboost"     → "gbr"
    * "logistic regression", "logistic model"       → "logreg"
    * "knn", "k-nearest neighbors", "k nearest"     → "knn"
- If the user does NOT clearly mention an algorithm, set "model_algo" to null.
  The system will then use "random_forest" as the default.

For unsupervised analysis:
- Infer which algorithm to use from the user's message:
    * "pca", "principal components", "dimensionality reduction" → algorithm = "pca"
    * "kmeans", "k-means", "cluster into X groups", "segment into X clusters" → algorithm = "kmeans"
    * "dbscan" or "density-based clustering" → algorithm = "dbscan"
    * "hierarchical", "agglomerative" → algorithm = "agglomerative"
- If user mentions number of components (e.g. "2D", "2 components") → set n_components.
- If user mentions number of clusters (e.g. "3 clusters", "cluster into 3 groups") → set n_clusters.
- If user mentions eps or radius for DBSCAN → set eps.
- If user mentions min_samples for DBSCAN → set min_samples.
- If user mentions linkage for hierarchical ("ward", "complete", "average", "single") → set linkage.
- If something is not mentioned, set it to null.

Your job:
- Read the user message.
- Decide which single action to take: "eda", "model", "unsupervised", or "hyperparam".
- Extract the file path mentioned by the user (e.g. "sample_data/cars.csv").
- If the user clearly mentions a target column (like "price" or "Species") for supervised modeling, set "target_column" accordingly.
- If the user does not mention any target and the action is "model" or "hyperparam", set "target_column" to null.
- For supervised modeling, also set "model_algo" as described above (or null if not specified).

Return ONLY a single JSON object with this structure:

{{
  "action": "eda" | "model" | "unsupervised" | "hyperparam",
  "file_path": "<path_to_csv>",
  "target_column": "<name or null>",

  "model_algo": "<linear | random_forest | gbr | logreg | knn | null>",

  "algorithm": "<pca | kmeans | dbscan | agglomerative | null>",
  "n_components": <int or null>,
  "n_clusters": <int or null>,
  "eps": <float or null>,
  "min_samples": <int or null>,
  "linkage": "<string or null>"
}}

User message:
\"\"\"{user_message}\"\"\"  
    """


def route_user_request(user_message: str) -> None:
    prompt = build_router_prompt(user_message)
    print("[Router] Asking LLM to route the request ...")
    router_response = run_llm(prompt)

    print("\n[Router] Raw LLM routing response:")
    print(router_response)
    print()

    # Parse JSON from LLM output
    try:
        start = router_response.find("{")
        end = router_response.rfind("}") + 1
        json_str = router_response[start:end]
        route = json.loads(json_str)
    except Exception as e:
        print("[Router]  Failed to parse routing JSON:", e)
        print("[Router] Aborting.")
        return

    action = route.get("action")
    file_path = route.get("file_path")
    target_col = route.get("target_column")
    model_algo = route.get("model_algo")

    algorithm = route.get("algorithm")
    n_components = route.get("n_components")
    n_clusters = route.get("n_clusters")
    eps = route.get("eps")
    min_samples = route.get("min_samples")
    linkage = route.get("linkage")

    print(f"[Router] Parsed action: {action}")
    print(f"[Router] File path: {file_path}")
    print(f"[Router] Target column: {target_col}")
    print(f"[Router] Model algo: {model_algo}")
    print(f"[Router] Algorithm: {algorithm}")
    print(f"[Router] n_components: {n_components}")
    print(f"[Router] n_clusters: {n_clusters}")
    print(f"[Router] eps: {eps}")
    print(f"[Router] min_samples: {min_samples}")
    print(f"[Router] linkage: {linkage}")

    if not file_path:
        print("[Router] No file_path provided in routing output. Aborting.")
        return

    # --------- helpers to clean values ---------
    def _normalize(value):
        if isinstance(value, str) and value.lower() in ("null", "none", ""):
            return None
        return value

    target_col = _normalize(target_col)
    model_algo = _normalize(model_algo)
    algorithm = _normalize(algorithm)
    n_components = _normalize(n_components)
    n_clusters = _normalize(n_clusters)
    eps = _normalize(eps)
    min_samples = _normalize(min_samples)
    linkage = _normalize(linkage)

    def _to_int(val):
        try:
            return int(val)
        except Exception:
            return None

    def _to_float(val):
        try:
            return float(val)
        except Exception:
            return None

    n_components = _to_int(n_components) if n_components is not None else None
    n_clusters = _to_int(n_clusters) if n_clusters is not None else None
    eps = _to_float(eps) if eps is not None else None
    min_samples = _to_int(min_samples) if min_samples is not None else None

    # default supervised algo if user didn't specify
    if model_algo is None:
        model_algo = "random_forest"

    # ---------------- ROUTING ----------------
    if action == "eda":
        print("[Router] Calling EDA Agent ...")
        run_eda_agent(file_path, use_llm=True)

    elif action == "model":
        print("[Router] Calling Model Agent ...")
        run_model_agent(file_path, target_col=target_col, algo=model_algo)

    elif action == "unsupervised":
        print("[Router] Calling Unsupervised Agent ...")

        if not algorithm:
            print("[Router] No algorithm specified for unsupervised analysis. Defaulting to 'kmeans'.")
            algorithm = "kmeans"

        run_unsupervised_agent(
            file_path=file_path,
            algo=algorithm,
            n_components=n_components,
            n_clusters=n_clusters,
            eps=eps,
            min_samples=min_samples,
            linkage=linkage,
        )

    elif action == "hyperparam":
        print("[Router] First calling Model Agent for a baseline ...")
        # Baseline: user-chosen algo (if present) or default RF
        run_model_agent(file_path, target_col=target_col, algo=model_algo)

        print("[Router] Now calling Hyperparameter Agent ...")
        # Hyperparam agent will read model_report.json and tune the right family
        run_hyperparam_agent(
            file_path=file_path,
            target_col=target_col,
            algo_list=None,
        )

    else:
        print(f"[Router] Unknown action '{action}'. Aborting.")


if __name__ == "__main__":
    # Examples:
    #   python router_agent.py "run EDA on sample_data/cars.csv"
    #   python router_agent.py "train a linear model on sample_data/cars.csv to predict price"
    #   python router_agent.py "run kmeans with 3 clusters on sample_data/cars.csv"
    #   python router_agent.py "train and tune a model on sample_data/cars.csv to predict price"
    if len(sys.argv) < 2:
        print('Usage: python router_agent.py "<your_request_here>"')
        sys.exit(1)

    user_msg = " ".join(sys.argv[1:])
    route_user_request(user_msg)

